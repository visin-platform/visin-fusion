"""Test results recorded as an evaluation on a Visin suite: what is sent, and what never stops a test stage."""

import json
import logging
import types

import pytest

from visin_fusion import __version__
from visin_fusion.engine import test_aggregator
from visin_fusion.engine.callbacks import TestEnd
from visin_fusion.engine.stages.test import common
from visin_fusion.engine.test_aggregator import TestResults
from visin_fusion.integrations import callback as visin_callback
from visin_fusion.integrations import visin as integration

RESULTS = {"day": {"overall": {"mIoU_foreground": 0.8}}, "night": {"overall": {"mIoU_foreground": 0.6}}}


class FakeVisinError(Exception):
    pass


class FakeVisin:
    """Stands in for the ``visin`` package, so these tests do not depend on which version is installed."""

    VisinError = FakeVisinError

    def __init__(self, refuse=None):
        self.calls = []
        self.refuse = refuse
        self.splits = {"splits/day.txt": ["b.png", "a.png"], "splits/night.txt": ["c.png"]}
        self.suites = {
            "suites/road.json": {"slug": "road-test", "version": 1, "protocol": {"data": {"kind": "external"}}},
            "suites/hub.json": {"slug": "hub-test", "version": 1, "protocol": {"data": {"kind": "hf"}}},
            "suites/visin.json": {"slug": "visin-test", "version": 1, "protocol": {"data": {"kind": "visin"}}},
        }

    def local_checkpoint(self, path, label=None):
        return {"kind": "local", "path": str(path), "label": label}

    def read_split(self, path):
        if path not in self.splits:
            raise FakeVisinError(f"cannot read the split file {path}")
        return self.splits[path]

    def manifest_digest(self, conditions):
        return "digest:" + ";".join(f"{name}={','.join(sorted(frames))}" for name, frames in sorted(conditions.items()))

    def load_suite(self, path):
        if path not in self.suites:
            raise FakeVisinError(f"cannot read the suite file {path}")
        return self.suites[path]

    def evaluate(self, results, **kwargs):
        self.calls.append({"results": results, **kwargs})
        if self.refuse:
            raise self.refuse
        return types.SimpleNamespace(verdict="eligible", stored=True, queued=False)


@pytest.fixture
def fake(monkeypatch):
    visin = FakeVisin()
    monkeypatch.setattr(integration, "visin", visin)
    monkeypatch.setattr(integration, "prepare_environment", lambda: None)
    monkeypatch.setattr(integration, "training_uuid_for_epoch", lambda logdir, epoch_uuid: "run-uuid")
    return visin


def config(suite="road-test@1", suite_file="suites/road.json"):
    general = {key: value for key, value in (("suite", suite), ("suite_file", suite_file)) if value}
    return {"General": general, "Log": {"logdir": "logs"}, "CLI": {"backbone": "clftv2"}}


def report(cfg, **overrides):
    arguments = {
        "epoch": 40,
        "epoch_uuid": "e-40",
        "results": RESULTS,
        "test_uuid": "t-1",
        "checkpoint_path": "logs/checkpoints/epoch_40_e-40.pth",
        "sample_counts": {"day": 1200, "night": 800},
        "splits": {"day": "splits/day.txt", "night": "splits/night.txt"},
    }
    return integration.report_evaluation(cfg, **{**arguments, **overrides})


class TestTestResults:
    def test_is_a_plain_dict_to_everything_that_saves_it_and_remembers_the_counts_beside_the_scores(self):
        results = TestResults(RESULTS, sample_counts={"day": 3})
        assert results == RESULTS and json.loads(json.dumps(results)) == RESULTS
        assert results.sample_counts == {"day": 3} and "sample_counts" not in results
        assert TestResults().sample_counts == {} and TestResults().splits == {}
        assert TestResults(splits={"day": "d.txt"}).splits == {"day": "d.txt"}

    def test_each_test_set_is_counted_as_it_is_scored(self):
        results = TestResults()
        results["day"] = {"overall": {}}
        results.sample_counts["day"] = 10
        assert dict(results) == {"day": {"overall": {}}} and results.sample_counts == {"day": 10}


class TestTheStageHandsTheCheckpointAndCountsToTheEvent:
    def test_the_event_carries_the_tested_file_and_how_many_frames_each_test_set_scored(self, monkeypatch, tmp_path):
        events = []
        monkeypatch.setattr(
            test_aggregator, "configured_callbacks", lambda _config: types.SimpleNamespace(emit=events.append)
        )
        cfg = {"Log": {"logdir": str(tmp_path)}}
        path = "checkpoints/epoch_7_0a1b2c.pth"
        test_aggregator.test_checkpoint_and_save(
            path,
            lambda *_args: TestResults(RESULTS, sample_counts={"day": 12, "night": 8}, splits={"day": "d.txt"}),
            cfg,
        )
        [event] = events
        assert isinstance(event, TestEnd)
        assert event.checkpoint_path == path and event.sample_counts == {"day": 12, "night": 8}
        assert event.splits == {"day": "d.txt"}
        assert event.epoch == 7 and dict(event.results) == RESULTS

    def test_a_plain_dict_of_results_gives_no_counts(self, monkeypatch, tmp_path):
        events = []
        monkeypatch.setattr(
            test_aggregator, "configured_callbacks", lambda _config: types.SimpleNamespace(emit=events.append)
        )
        test_aggregator.test_checkpoint_and_save(
            "epoch_1_ab.pth", lambda *_a: dict(RESULTS), {"Log": {"logdir": str(tmp_path)}}
        )
        assert events[0].sample_counts is None and events[0].splits is None

    def test_the_suite_flag_sets_general_suite_for_that_run(self, monkeypatch):
        seen = {}
        loaded = {"General": {"seed": 0}, "Log": {"logdir": "logs"}}
        monkeypatch.setattr(common, "load_config", lambda _path: loaded)
        monkeypatch.setattr(common, "configure_logging", lambda: None)
        monkeypatch.setattr(common, "set_seed", lambda _seed: None)
        monkeypatch.setattr(common, "get_device", lambda _config: "cpu")
        monkeypatch.setattr(common, "get_checkpoint_path_with_fallback", lambda _config: "best.pth")
        monkeypatch.setattr(common, "test_checkpoint_and_save", lambda *args: seen.update(config=args[2]))
        common.main(["-c", "cfg.json", "--suite", "road-test@2"])
        assert seen["config"]["General"]["suite"] == "road-test@2"
        loaded["General"].pop("suite")
        common.main(["-c", "cfg.json"])
        assert "suite" not in seen["config"]["General"]
        common.main(["-c", "cfg.json", "--suite-file", "suites/road.json"])
        assert seen["config"]["General"]["suite_file"] == "suites/road.json"


class TestReportEvaluation:
    def test_records_the_results_on_the_suite_naming_the_weights_the_run_and_the_evaluator(self, fake):
        report(config())
        [call] = fake.calls
        assert call["suite"] == "road-test@1" and call["results"] == RESULTS
        assert call["checkpoint"] == {
            "kind": "local",
            "path": "logs/checkpoints/epoch_40_e-40.pth",
            "label": "clftv2-epoch-40",
        }
        assert call["sample_counts"] == {"day": 1200, "night": 800}
        assert (call["run"], call["epoch"], call["epoch_uuid"]) == ("run-uuid", 40, "e-40")
        assert call["evaluator"] == {"package": "visin-fusion", "version": __version__}

    def test_the_evaluation_is_named_by_the_test_uuid_so_a_repeat_is_harmless(self, fake):
        report(config())
        report(config())
        assert [call["uuid"] for call in fake.calls] == ["t-1", "t-1"]

    def test_without_a_suite_nothing_is_recorded(self, fake):
        report(config(suite=None, suite_file=None))
        report({"Log": {"logdir": "logs"}})
        assert fake.calls == []

    def test_an_epoch_with_no_uuid_names_no_run(self, fake):
        report(config(), epoch_uuid=None)
        assert fake.calls[0]["run"] is None

    def test_a_refusal_is_logged_and_does_not_fail_the_stage(self, fake, caplog):
        fake.refuse = FakeVisinError("Suite not found")
        with caplog.at_level(logging.WARNING):
            report(config())
        assert "did not record the evaluation on road-test@1: Suite not found" in caplog.text

    def test_a_result_kept_for_sync_says_so(self, fake, caplog, monkeypatch):
        monkeypatch.setattr(
            fake, "evaluate", lambda *_a, **_k: types.SimpleNamespace(verdict=None, stored=False, queued=True)
        )
        with caplog.at_level(logging.INFO):
            report(config())
        assert "kept to send with `visin sync`" in caplog.text

    def test_the_verdict_is_logged(self, fake, caplog):
        with caplog.at_level(logging.INFO):
            report(config())
        assert "Visin evaluation on road-test@1: eligible" in caplog.text

    def test_an_older_visin_that_cannot_record_evaluations_is_told_what_to_do(self, monkeypatch, caplog):
        monkeypatch.setattr(integration, "visin", types.SimpleNamespace())
        with caplog.at_level(logging.WARNING):
            report(config())
        assert "pip install -U visin" in caplog.text

    def test_an_unknown_checkpoint_cannot_be_named_and_is_not_guessed(self, fake, caplog):
        with caplog.at_level(logging.WARNING):
            report(config(), checkpoint_path=None)
        assert fake.calls == [] and "checkpoint is not known" in caplog.text


class TestWhatRan:
    def test_the_suite_file_is_the_protocol_that_ran_and_the_frame_lists_are_the_data_that_was_scored(self, fake):
        report(config())
        [call] = fake.calls
        assert call["protocol"] == "suites/road.json"
        assert call["data"] == {"kind": "external", "manifestSha256": "digest:day=a.png,b.png;night=c.png"}

    def test_a_changed_frame_list_gives_another_digest_even_with_the_same_counts(self, fake):
        report(config())
        fake.splits["splits/day.txt"] = ["a.png", "z.png"]
        report(config())
        first, second = (call["data"]["manifestSha256"] for call in fake.calls)
        assert first != second

    def test_a_suite_pinned_to_a_hub_dataset_gets_the_repo_and_commit_the_dataset_root_resolved_to(self, fake):
        commit = "3f2a1c9d8e7b6a5f4e3d2c1b0a99887766554433"
        root = f"/home/u/.cache/huggingface/hub/datasets--acme--zod-png/snapshots/{commit}"
        cfg = {**config(suite=None, suite_file="suites/hub.json"), "Dataset": {"dataset_root": root}}
        report(cfg)
        assert fake.calls[0]["data"] == {"kind": "hf", "repo": "acme/zod-png", "commit": commit}

    def test_a_hub_suite_with_a_dataset_that_is_not_a_hub_snapshot_sends_no_data_rather_than_the_wrong_kind(self, fake):
        cfg = {**config(suite=None, suite_file="suites/hub.json"), "Dataset": {"dataset_root": "/data/zod"}}
        report(cfg)
        assert fake.calls[0]["data"] is None

    def test_a_suite_pinned_to_a_visin_dataset_sends_no_data_because_the_archive_digest_is_not_known_here(self, fake):
        report(config(suite=None, suite_file="suites/visin.json"))
        assert fake.calls[0]["data"] is None

    def test_without_a_suite_file_the_kind_is_unknown_so_no_data_is_sent(self, fake):
        report(config(suite_file=None))
        assert fake.calls[0]["data"] is None

    def test_the_suite_comes_from_the_file_when_the_config_names_none(self, fake):
        report(config(suite=None))
        assert fake.calls[0]["suite"] == "road-test@1" and fake.calls[0]["protocol"] == "suites/road.json"

    def test_a_suite_with_no_file_is_recorded_and_the_log_says_it_is_reported_not_observed(self, fake, caplog):
        with caplog.at_level(logging.WARNING):
            report(config(suite_file=None))
        assert fake.calls[0]["protocol"] is None and "suite_file is not set" in caplog.text
        assert "ranked as reported" in caplog.text and "not ranked" not in caplog.text

    def test_a_suite_file_that_cannot_be_read_names_no_suite_and_records_nothing(self, fake, caplog):
        with caplog.at_level(logging.WARNING):
            report(config(suite=None, suite_file="missing.json"))
        assert fake.calls == [] and "could not be read" in caplog.text

    def test_a_file_without_a_slug_names_no_suite(self, fake):
        fake.suites["suites/bare.json"] = {"protocol": {}}
        report(config(suite=None, suite_file="suites/bare.json"))
        assert fake.calls == []

    def test_frame_lists_that_cannot_be_read_are_not_guessed_and_do_not_stop_the_evaluation(self, fake, caplog):
        fake.splits.pop("splits/night.txt")
        with caplog.at_level(logging.WARNING):
            report(config())
        assert fake.calls[0]["data"] is None and "frame lists could not be read" in caplog.text

    def test_no_splits_report_no_data(self, fake):
        report(config(), splits=None)
        assert fake.calls[0]["data"] is None

    def test_an_older_visin_without_suite_helpers_still_records_what_it_can(self, monkeypatch):
        calls = []
        older = types.SimpleNamespace(
            VisinError=FakeVisinError,
            local_checkpoint=lambda path, label=None: {"label": label},
            evaluate=lambda results, **kwargs: (
                calls.append(kwargs) or types.SimpleNamespace(verdict="incomplete", stored=True, queued=False)
            ),
        )
        monkeypatch.setattr(integration, "visin", older)
        monkeypatch.setattr(integration, "prepare_environment", lambda: None)
        monkeypatch.setattr(integration, "training_uuid_for_epoch", lambda *_a: None)
        report(config(suite=None))
        assert calls == []
        report(config())
        assert calls[0]["data"] is None


class TestReportEvaluationSaysWhetherItRecorded:
    def test_it_recorded_when_the_evaluation_was_stored_or_kept_for_sync(self, fake, monkeypatch):
        assert report(config()) is True
        monkeypatch.setattr(
            fake, "evaluate", lambda *_a, **_k: types.SimpleNamespace(verdict=None, stored=False, queued=True)
        )
        assert report(config()) is True

    def test_it_did_not_when_nothing_was_configured_to_record_it(self, fake, monkeypatch):
        monkeypatch.setattr(
            fake, "evaluate", lambda *_a, **_k: types.SimpleNamespace(verdict=None, stored=False, queued=False)
        )
        assert report(config()) is False

    def test_it_did_not_without_a_suite_a_checkpoint_or_a_capable_visin_or_after_a_refusal(self, fake, monkeypatch):
        assert report(config(suite=None, suite_file=None)) is False
        assert report(config(), checkpoint_path=None) is False
        fake.refuse = FakeVisinError("Suite not found")
        assert report(config()) is False
        monkeypatch.setattr(integration, "visin", types.SimpleNamespace())
        assert report(config()) is False


class TestTheCallback:
    def test_a_test_end_on_a_suite_reports_the_evaluation_and_not_also_a_test_result(self, monkeypatch):
        calls = []
        monkeypatch.setattr(visin_callback.service, "prepare_environment", lambda: None)
        monkeypatch.setattr(visin_callback.service, "report_test_results", lambda *a, **k: calls.append(("test", a, k)))
        monkeypatch.setattr(
            visin_callback.service, "report_evaluation", lambda *a, **k: calls.append(("evaluation", a, k)) or True
        )
        callback = visin_callback.VisinCallback(config())
        callback.on_test_end(
            TestEnd(
                config=config(),
                epoch=3,
                epoch_uuid="e-3",
                results=RESULTS,
                test_uuid="t-3",
                checkpoint_path="epoch_3_e-3.pth",
                sample_counts={"day": 5},
                splits={"day": "splits/day.txt"},
            )
        )
        assert [kind for kind, *_ in calls] == ["evaluation"]
        _, _, arguments = calls[0]
        assert arguments["test_uuid"] == "t-3" and arguments["checkpoint_path"] == "epoch_3_e-3.pth"
        assert arguments["sample_counts"] == {"day": 5} and arguments["results"] == RESULTS
        assert arguments["splits"] == {"day": "splits/day.txt"}

    def test_a_test_end_that_could_not_be_recorded_on_a_suite_falls_back_to_the_test_result(self, monkeypatch):
        calls = []
        monkeypatch.setattr(visin_callback.service, "prepare_environment", lambda: None)
        monkeypatch.setattr(visin_callback.service, "report_test_results", lambda *a, **k: calls.append((a, k)))
        monkeypatch.setattr(visin_callback.service, "report_evaluation", lambda *a, **k: False)
        visin_callback.VisinCallback(config()).on_test_end(
            TestEnd(config=config(), epoch=3, epoch_uuid="e-3", results=RESULTS, test_uuid="t-3")
        )
        [(arguments, keywords)] = calls
        assert arguments[1:4] == (3, "e-3", RESULTS) and keywords == {"test_uuid": "t-3"}

    def test_an_event_without_counts_passes_none(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(visin_callback.service, "prepare_environment", lambda: None)
        monkeypatch.setattr(visin_callback.service, "report_test_results", lambda *a, **k: None)
        monkeypatch.setattr(visin_callback.service, "report_evaluation", lambda *a, **k: seen.update(k) or True)
        visin_callback.VisinCallback(config()).on_test_end(
            TestEnd(config=config(), epoch=1, epoch_uuid=None, results=RESULTS)
        )
        assert seen["sample_counts"] is None and seen["checkpoint_path"] is None and seen["splits"] is None
