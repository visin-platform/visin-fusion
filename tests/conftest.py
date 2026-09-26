"""Shared pytest options for the visin-fusion tests."""
import pytest
import torch


def pytest_addoption(parser):
    group = parser.getgroup("visin-fusion")
    group.addoption(
        "--visin",
        choices=["offline", "online", "disabled"],
        default="offline",
        help="How the e2e tests report to Visin: offline (default) keeps reports on disk and checks "
             "them; online sends them with the VISIN_TOKEN from the environment or "
             "integrations/.env and reads them back; disabled reports nothing.",
    )
    group.addoption(
        "--device",
        default=None,
        help="Device for the e2e runs (default: cuda:0 if available, else cpu).",
    )
    group.addoption(
        "--models",
        default="clft,swin,maskformer,mask2former,deeplab",
        help="Comma-separated models for the e2e tests (default: all).",
    )
    group.addoption(
        "--modes",
        default="rgb,lidar,fusion",
        help="Comma-separated modes to run per model (default: rgb,lidar,fusion).",
    )


@pytest.fixture(scope="session")
def visin_mode(request):
    return request.config.getoption("--visin")


@pytest.fixture(scope="session")
def device(request):
    return request.config.getoption("--device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
