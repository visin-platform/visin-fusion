# Install and connect

## Install

Run these commands once:

```bash title="Install Fusion and the Visin SDK"
git clone https://github.com/visin-platform/visin-fusion.git
cd visin-fusion
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,visin]'
```

Use Python 3.10 or newer. Run the remaining examples from this checkout.
Training uses CUDA when available and falls back to CPU.

## Optional: connect to Visin

Create a project pipeline key and save this as `.env`, replacing the token and
using your deployment's URL:

```dotenv title=".env"
VISIN_URL=https://vision-api.visin.eu
VISIN_TOKEN=replace-with-your-project-pipeline-key
VISIN_DIR=.visin
```

Fusion loads `.env` automatically. Without a token, outputs remain local.
Keep the file out of version control. Exported settings take precedence.

To check the connection, export the settings for the SDK CLI:

```bash title="Check reporting"
set -a
source .env
set +a
visin check --write
```

The check creates and deletes a small test run. Export your credential before
downloading if a dataset is private. See [Visin reporting](../visin.md) for more.

## Next: prepare a dataset

- [ZOD](datasets/zod.md)
- [Waymo](datasets/waymo.md)
- [Iseauto](datasets/iseauto.md)

For a smaller local check, use the [28-frame sample](../getting-started.md#run-the-sample-pipeline).
