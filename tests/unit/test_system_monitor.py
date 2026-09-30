"""System snapshots handle CPU hosts, GPU hosts, and optional NVML failures."""

import sys
from types import SimpleNamespace

from visin_fusion.utils import system_monitor as monitor

GIB = 1024**3


def fake_resources(monkeypatch, cuda=False):
    memory = SimpleNamespace(total=16 * GIB, available=8 * GIB, used=8 * GIB, percent=50.0)
    process = SimpleNamespace(
        cpu_percent=lambda **_: 3.0, memory_info=lambda: SimpleNamespace(rss=2 * GIB), num_threads=lambda: 4
    )
    monkeypatch.setattr(monitor.psutil, "virtual_memory", lambda: memory)
    monkeypatch.setattr(monitor.psutil, "Process", lambda: process)
    monkeypatch.setattr(monitor.psutil, "cpu_count", lambda logical=True: 8 if logical else 4)
    monkeypatch.setattr(monitor.psutil, "cpu_percent", lambda **_: [1.0, 2.0] if _.get("percpu") else 2.0)
    monkeypatch.setattr(monitor.torch.cuda, "is_available", lambda: cuda)
    if cuda:
        props = SimpleNamespace(total_memory=8 * GIB, major=8, minor=0, multi_processor_count=12)
        monkeypatch.setattr(monitor.torch.cuda, "device_count", lambda: 1)
        monkeypatch.setattr(monitor.torch.cuda, "get_device_name", lambda _: "Fake GPU")
        monkeypatch.setattr(monitor.torch.cuda, "get_device_properties", lambda _: props)
        monkeypatch.setattr(monitor.torch.cuda, "memory_allocated", lambda _: 2 * GIB)
        monkeypatch.setattr(monitor.torch.cuda, "memory_reserved", lambda _: 3 * GIB)


def test_cpu_snapshots_and_print(monkeypatch, caplog):
    fake_resources(monkeypatch)
    info = monitor.get_system_info()
    snapshot = monitor.get_epoch_system_snapshot()
    assert info["gpu"]["available"] is False
    assert info["memory"]["total_gb"] == 16
    assert snapshot["process"]["threads"] == 4
    assert "gpu" not in snapshot
    monitor.print_system_info(info)
    assert "GPU: Not available" in caplog.text
    monkeypatch.setattr(monitor, "get_system_info", lambda: info)
    monitor.print_system_info()
    assert "Physical cores: 4" in caplog.text


def test_gpu_information_and_nvml_snapshot(monkeypatch, caplog):
    fake_resources(monkeypatch, cuda=True)
    info = monitor.get_system_info()
    assert info["gpu"]["gpu_0"]["free_memory_gb"] == 6
    assert info["gpu"]["gpu_0"]["compute_capability"] == "8.0"
    monitor.print_system_info(info)
    assert "Fake GPU" in caplog.text
    calls = []
    nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        NVML_CLOCK_SM=1,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: calls.append("init"),
        nvmlShutdown=lambda: calls.append("shutdown"),
        nvmlDeviceGetHandleByIndex=lambda _: "gpu",
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=75, memory=35),
        nvmlDeviceGetTemperature=lambda *args: 42,
        nvmlDeviceGetPowerUsage=lambda _: 100000,
        nvmlDeviceGetPowerManagementLimit=lambda _: 200000,
        nvmlDeviceGetClockInfo=lambda _, kind: 1200 if kind == 1 else 800,
        nvmlDeviceGetFanSpeed=lambda _: 55,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    snapshot = monitor.get_epoch_system_snapshot()
    assert snapshot["gpu"]["gpu_0"]["memory_utilization_percent"] == 25
    assert snapshot["gpu"]["gpu_0"]["gpu_utilization_percent"] == 75
    assert snapshot["gpu"]["gpu_0"]["power_percent"] == 50
    assert snapshot["gpu"]["gpu_0"]["fan_speed_percent"] == 55
    assert calls == ["init", "shutdown"]


def test_snapshot_without_nvml_and_with_partial_nvml_failures(monkeypatch):
    fake_resources(monkeypatch, cuda=True)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    snapshot = monitor.get_epoch_system_snapshot()
    assert "gpu_0" in snapshot["gpu"]
    assert "gpu_utilization_percent" not in snapshot["gpu"]["gpu_0"]

    def broken(*args):
        raise RuntimeError("unavailable")

    nvml = SimpleNamespace(
        NVML_TEMPERATURE_GPU=0,
        NVML_CLOCK_SM=1,
        NVML_CLOCK_MEM=2,
        nvmlInit=lambda: None,
        nvmlShutdown=broken,
        nvmlDeviceGetHandleByIndex=lambda _: "gpu",
        nvmlDeviceGetUtilizationRates=lambda _: SimpleNamespace(gpu=1, memory=2),
        nvmlDeviceGetTemperature=lambda *args: 40,
        nvmlDeviceGetPowerUsage=broken,
        nvmlDeviceGetClockInfo=broken,
        nvmlDeviceGetFanSpeed=broken,
    )
    monkeypatch.setitem(sys.modules, "pynvml", nvml)
    snapshot = monitor.get_epoch_system_snapshot()
    assert snapshot["gpu"]["gpu_0"]["temperature_celsius"] == 40
    assert "power_watts" not in snapshot["gpu"]["gpu_0"]
    assert "fan_speed_percent" not in snapshot["gpu"]["gpu_0"]
    nvml.nvmlDeviceGetHandleByIndex = broken
    assert "nvml_error" in monitor.get_epoch_system_snapshot()["gpu"]["gpu_0"]
