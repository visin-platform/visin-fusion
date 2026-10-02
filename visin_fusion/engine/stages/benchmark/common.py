#!/usr/bin/env python3
"""Benchmark any model: parameters, FLOPs, inference time and memory, per config and device.

The model is built from the config through visin_fusion/models/registry.py with random weights: speed and
memory do not depend on the weights, and nothing has to be downloaded. Results are written to
<Log.logdir>/benchmark/ and reported to Visin, linked to the run's latest epoch.

    python -m visin_fusion.engine.stages.benchmark.common -c <config.json> [<config.json> ...] [--single --device cpu]
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import GPUtil
import numpy as np
import pandas as pd
import psutil
import torch

from visin_fusion.config.config import load_config
from visin_fusion.engine.callbacks import Benchmark, configured_callbacks
from visin_fusion.logging_setup import configure_logging
from visin_fusion.models.registry import from_config, model_section

logger = logging.getLogger(__name__)

try:
    from thop import profile
except ImportError:
    profile = None
try:
    from fvcore.nn import FlopCountAnalysis
except ImportError:
    FlopCountAnalysis = None


def find_config_files(paths):
    """Config files among ``paths`` (files, or directories searched recursively)."""
    found = []
    for path in paths:
        if os.path.isfile(path) and path.endswith(".json"):
            found.append(path)
        elif os.path.isdir(path):
            found += sorted(glob.glob(os.path.join(path, "**", "*.json"), recursive=True))
    return found


def setup_device(device):
    """The torch device for ``auto``, ``cpu`` or ``cuda[:N]``; CUDA falls back to the CPU when unavailable."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.info("CUDA not available, falling back to CPU")
        return torch.device("cpu")
    if device == "cpu" or device.startswith("cuda"):
        return torch.device(device)
    raise ValueError(f"Unsupported device: {device}")


def latest_epoch(logdir):
    """(epoch, epoch UUID, training UUID) of the newest epoch log: this run's last epoch, even when an
    earlier run's logs with higher epoch numbers remain in the same directory."""
    files = glob.glob(os.path.join(logdir, "epochs", "epoch_*.json"))
    if not files:
        logger.warning("No epoch logs in %s/epochs; the benchmark is linked to no epoch", logdir)
        return None, None, None
    return epoch_info(max(files, key=os.path.getmtime))


def epoch_info(epoch_file):
    """``(epoch, epoch UUID, training UUID)`` recorded in an epoch log file."""
    with open(epoch_file) as f:
        data = json.load(f)
    logger.info("Using epoch file: %s", epoch_file)
    return data.get("epoch"), data.get("epoch_uuid"), data.get("training_uuid")


def count_parameters(model):
    """Total and trainable parameter counts of ``model``, as a dict."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "total_parameters_m": total / 1e6,
        "trainable_parameters_m": trainable / 1e6,
    }


def count_flops(model, rgb, lidar):
    """FLOPs of one forward pass, with thop, or fvcore if thop fails."""
    info = {"flops_available": False, "total_flops": None, "flops_giga": None, "flops_method": None}
    attempts = []
    if profile is not None:
        attempts.append(("thop", lambda: profile(model, inputs=(rgb, lidar), verbose=False)[0]))
    if FlopCountAnalysis is not None:
        attempts.append(("fvcore", lambda: FlopCountAnalysis(model, (rgb, lidar)).total()))
    for method, count in attempts:
        try:
            with torch.no_grad():
                flops = count()
        except Exception as e:  # profilers fail on some operations; the next one may not
            logger.error("%s profiling failed: %s", method, e)
            continue
        info.update(flops_available=True, total_flops=flops, flops_giga=flops / 1e9, flops_method=method)
        break
    return info


def measure_inference(model, rgb, lidar, device, num_runs, warmup_runs):
    """Time ``num_runs`` forward passes after ``warmup_runs``, and the memory in use.

    gpu_memory_* sample memory still allocated after each pass (mostly the weights); the peak
    during the passes, activations included, is gpu_memory_peak_mb.
    """
    model.eval()  # inference mode for timing, whatever ran before
    cuda = device.type == "cuda"
    baseline_gpu = torch.cuda.memory_allocated() / 2**20 if cuda else 0
    baseline_ram = psutil.Process().memory_info().rss / 2**20
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(rgb, lidar)
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        times, gpu, ram = [], [], []
        for _ in range(num_runs):
            start = time.perf_counter()
            model(rgb, lidar)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            if cuda:
                gpu.append(torch.cuda.memory_allocated() / 2**20)
            else:
                ram.append(psutil.Process().memory_info().rss / 2**20)

    times = np.array(times)
    result = {
        "mean_time_ms": times.mean() * 1000,
        "std_time_ms": times.std() * 1000,
        "min_time_ms": times.min() * 1000,
        "max_time_ms": times.max() * 1000,
        "fps": 1.0 / times.mean(),
        "num_runs": num_runs,
        "baseline_gpu_memory_mb": baseline_gpu,
        "baseline_ram_memory_mb": baseline_ram,
    }
    for name, samples in (("gpu", gpu), ("ram", ram)):
        if samples:
            samples = np.array(samples)
            result.update(
                {
                    f"{name}_memory_mean_mb": samples.mean(),
                    f"{name}_memory_std_mb": samples.std(),
                    f"{name}_memory_min_mb": samples.min(),
                    f"{name}_memory_max_mb": samples.max(),
                }
            )
    if cuda:
        result["gpu_memory_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return result


def system_info():
    """CPU, memory, Python, PyTorch and GPU details of this machine, as reported with the benchmark."""
    info = {
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "memory_total_gb": psutil.virtual_memory().total / 2**30,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        gpus = GPUtil.getGPUs()
        info["gpu_info"] = [
            {
                "name": gpus[i].name,
                "memory_total_mb": gpus[i].memoryTotal,
                "memory_free_mb": gpus[i].memoryFree,
                "temperature": gpus[i].temperature,
            }
            if i < len(gpus)
            else {
                "name": torch.cuda.get_device_name(i),
                "memory_total_mb": torch.cuda.get_device_properties(i).total_memory / 2**20,
                "memory_free_mb": None,
                "temperature": None,
            }
            for i in range(torch.cuda.device_count())
        ]
    return info


def benchmark_config(config_path, device, num_runs=100, warmup_runs=10):
    """The measurements of one config on one device."""
    config = load_config(config_path)
    mode = config["CLI"]["mode"]
    logger.info(
        "\n%s\nBenchmarking %s (%s, %s) on %s\n%s",
        "=" * 60,
        os.path.basename(config_path),
        config["CLI"]["backbone"],
        mode,
        device,
        "=" * 60,
    )
    model = from_config(config, pretrained=False).to(device)
    model.eval()
    size = config["Dataset"]["transforms"]["resize"]
    rgb, lidar = torch.randn(1, 3, size, size, device=device), torch.randn(1, 3, size, size, device=device)

    result = {
        "modality": mode,
        **count_parameters(model),
        "model_name": config.get("Summary") or "Unknown",
        "backbone": config["CLI"]["backbone"],
        "dataset": config["Dataset"]["name"],
        "image_size": size,
        "pretrained": model_section(config).get("pretrained", True),
        **count_flops(model, rgb, lidar),
        **measure_inference(model, rgb, lidar, device, num_runs, warmup_runs),
        "config_path": config_path,
        "device": str(device),
        "device_type": device.type,
    }
    logger.info("  Parameters: %.1fM", result["total_parameters_m"])
    if result["flops_available"]:
        logger.info("  FLOPs: %.2fG (%s)", result["flops_giga"], result["flops_method"])
    logger.info("  Inference: %.2f±%.2f ms, %.1f FPS", result["mean_time_ms"], result["std_time_ms"], result["fps"])
    if "gpu_memory_peak_mb" in result:
        logger.info(
            "  GPU memory: %.1f MB after a pass, %.1f MB peak",
            result["gpu_memory_mean_mb"],
            result["gpu_memory_peak_mb"],
        )
    return result


def save_results(results, logdir, epoch, epoch_uuid, training_uuid, output_path=None, config=None):
    """Write the results (JSON and a CSV summary) and report them to the callbacks ``config`` names."""
    if output_path is None:
        os.makedirs(os.path.join(logdir, "benchmark"), exist_ok=True)
        output_path = os.path.join(logdir, "benchmark", f"benchmark_results_{time.strftime('%Y%m%d_%H%M%S')}.json")
    info = system_info()
    with open(output_path, "w") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": epoch,
                "epoch_uuid": epoch_uuid,
                "training_uuid": training_uuid,
                "system_info": info,
                "results": results,
            },
            f,
            indent=2,
            default=float,
        )
    pd.DataFrame(results).to_csv(output_path.replace(".json", "_summary.csv"), index=False)
    logger.info("\nBenchmark results saved to: %s", output_path)

    configured_callbacks(config).emit(
        Benchmark(results=results, system_info=info, training_uuid=training_uuid, epoch=epoch, epoch_uuid=epoch_uuid)
    )
    return output_path


def main(argv=None):
    """Entry point of the benchmark stage (``python -m visin_fusion.engine.stages.benchmark.common``)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", nargs="+", required=True, help="config files or directories of them")
    parser.add_argument("-d", "--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument(
        "--single",
        action="store_true",
        help="benchmark on --device only (default: on both CPU and GPU when there is a GPU)",
    )
    parser.add_argument("--epoch-file", help="epoch log to link the results to (default: the newest in Log.logdir)")
    parser.add_argument("--output", help="results file (default: <Log.logdir>/benchmark/benchmark_results_<time>.json)")
    parser.add_argument("--num-runs", type=int, default=100, help="timed forward passes (default: 100)")
    parser.add_argument("--warmup-runs", type=int, default=10, help="untimed passes first (default: 10)")
    args = parser.parse_args(argv)
    configure_logging()

    configs = find_config_files(args.config)
    if not configs:
        sys.exit("No config files found")
    devices = ["cpu", "cuda"] if not args.single and torch.cuda.is_available() else [args.device]

    results, failed = [], []
    for device in map(setup_device, devices):
        for config_path in configs:
            try:
                results.append(benchmark_config(config_path, device, args.num_runs, args.warmup_runs))
            except Exception as e:  # one broken config should not stop the others; it still fails the run
                logger.error("Benchmarking %s on %s failed: %s: %s", config_path, device, type(e).__name__, e)
                failed.append(config_path)

    config = load_config(configs[0])
    logdir = config["Log"]["logdir"]
    epoch, epoch_uuid, training_uuid = epoch_info(args.epoch_file) if args.epoch_file else latest_epoch(logdir)
    if results:
        save_results(results, logdir, epoch, epoch_uuid, training_uuid, args.output, config)
    if failed:
        sys.exit(
            f"Benchmarking failed for {len(failed)} of {len(configs) * len(devices)} config/device pairs: "
            f"{sorted(set(failed))}"
        )


if __name__ == "__main__":
    main()
