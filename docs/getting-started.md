# Getting started

## Install

```bash
git clone https://github.com/visin-platform/visin-fusion.git
cd visin-fusion
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Or use the Docker image (see [Running](running.md#docker)).

## Quick start

The repository includes a 28-frame sample of ZOD in `tests/data/zod_sample`. Train SwinFusion on it
for two short epochs, then test, visualize and benchmark:

```bash
python run.py -c configs/quickstart.json
```

Everything the run writes goes to `logs/quickstart/`: epoch logs, checkpoints, test results,
visualizations, benchmark results and TensorBoard logs.

## Report to Visin

With a Visin project token, every stage is reported to [Visin](https://app.visin.eu): the run and its
config, each epoch, the test results, the visualizations and the benchmark.

```bash
cp integrations/.env.example integrations/.env   # then set VISIN_TOKEN
```

Without a token, everything runs and nothing is reported. See `integrations/README.md` for
offline nodes (`VISIN_MODE=offline`, then `visin sync`).

## Next

- [Configs](configs.md): write a config for your own run
- [Datasets](datasets.md): prepare your own data
- [Running](running.md): stages, Docker, the cluster
