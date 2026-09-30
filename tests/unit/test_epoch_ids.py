"""Training and epoch IDs are chosen the same way with or without Visin."""

import json
import uuid

import pytest

from visin_fusion.engine.epoch_ids import epoch_uuid_for, training_uuid_for_run


def earlier_run(logdir, epoch=3):
    """A logs directory holding one epoch log and checkpoint of an earlier training."""
    training_uuid = str(uuid.uuid4())
    epoch_uuid = epoch_uuid_for(training_uuid, epoch)
    (logdir / "epochs").mkdir(parents=True)
    (logdir / "checkpoints").mkdir()
    (logdir / "epochs" / f"epoch_{epoch}_{epoch_uuid}.json").write_text(
        json.dumps({"training_uuid": training_uuid, "epoch_uuid": epoch_uuid, "epoch": epoch})
    )
    (logdir / "checkpoints" / f"epoch_{epoch}_{epoch_uuid}.pth").write_bytes(b"")
    return training_uuid


def config_for(logdir, **general):
    return {"General": {"resume_training": True, "model_path": "", **general}, "Log": {"logdir": str(logdir)}}


def test_matches_visin():
    visin = pytest.importorskip("visin")
    for training_uuid in (str(uuid.uuid4()) for _ in range(5)):
        for epoch in (0, 1, 299):
            assert epoch_uuid_for(training_uuid, epoch) == visin.epoch_uuid_for(training_uuid, epoch)


def test_resume_continues_the_earlier_run(tmp_path):
    earlier = earlier_run(tmp_path)
    assert training_uuid_for_run(config_for(tmp_path)) == (earlier, True)


@pytest.mark.parametrize("flag", ["reset_lr", "transfer_learning", "create_new_training"])
def test_fresh_flags_start_a_new_run(tmp_path, flag):
    earlier = earlier_run(tmp_path)
    training_uuid, resumed = training_uuid_for_run(config_for(tmp_path, **{flag: True}))
    assert (training_uuid != earlier, resumed) == (True, False)


def test_no_resume_starts_a_new_run(tmp_path):
    earlier = earlier_run(tmp_path)
    training_uuid, resumed = training_uuid_for_run(config_for(tmp_path, resume_training=False))
    assert (training_uuid != earlier, resumed) == (True, False)
