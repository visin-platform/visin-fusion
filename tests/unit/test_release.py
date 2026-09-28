"""tools/release.py: the version it picks and the changelog it writes."""

import importlib.util
import re
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "release.py"
spec = importlib.util.spec_from_file_location("release", SCRIPT)
release = importlib.util.module_from_spec(spec)
sys.modules["release"] = release  # dataclasses look their module up here
spec.loader.exec_module(release)


def commits(*messages):
    return [c for c in (release.parse_commit(m) for m in messages) if c]


def test_conventional_commits_are_parsed():
    commit = release.parse_commit("feat(sync)!: resume interrupted syncs\n\nBREAKING CHANGE: new file layout")
    assert (commit.type, commit.scope, commit.subject, commit.breaking) == (
        "feat",
        "sync",
        "resume interrupted syncs",
        True,
    )
    assert commit.breaking_note == "new file layout"
    assert release.parse_commit("Merge branch 'main'") is None


@pytest.mark.parametrize(
    ("current", "messages", "release_as", "expected"),
    [
        ("1.2.3", ["fix: a"], None, "1.2.4"),
        ("1.2.3", ["fix: a", "feat: b"], None, "1.3.0"),
        ("1.2.3", ["feat!: b"], None, "2.0.0"),
        ("0.4.1", ["feat!: b"], None, "0.5.0"),  # before 1.0, breaking is a minor
        ("1.2.3", ["docs: a"], None, "1.2.4"),
        ("1.2.3", ["feat: a"], "patch", "1.2.4"),
        ("1.2.3", [], "major", "2.0.0"),
        ("1.2.3", [], "v1.5.0", "1.5.0"),
    ],
)
def test_the_version_follows_the_commits(current, messages, release_as, expected):
    assert release.next_version(current, commits(*messages), release_as) == expected


def test_a_nonsense_release_as_is_refused():
    with pytest.raises(ValueError):
        release.next_version("1.0.0", [], "huge")


CHANGELOG = """# Changelog

Intro.

## [Unreleased]

### Fixed

- A hand-written note.

## [1.0.0] - 2026-01-01

### Added

- The first thing.

[Unreleased]: https://github.com/visin-platform/visin-fusion/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/visin-platform/visin-fusion/releases/tag/v1.0.0
"""


def test_the_changelog_entry_keeps_hand_written_notes_and_adds_commits():
    generated = release.render_entry(
        commits("feat(api): epochs_frame", "fix: retry 429", "chore: tidy", "feat!: rename init")
    )
    text = release.update_changelog(CHANGELOG, "1.1.0", "1.0.0", generated, "2026-09-25")
    notes = release.release_notes(text, "1.1.0")
    # Breaking changes first, then the hand-written sections, then the ones only commits produced
    assert notes.index("### Breaking changes") < notes.index("### Fixed") < notes.index("### Added")
    assert "- **api:** epochs_frame" in notes
    assert "- A hand-written note." in notes and "- retry 429" in notes
    assert "tidy" not in notes
    assert "## [Unreleased]\n\n## [1.1.0] - 2026-09-25" in text
    assert "[Unreleased]: https://github.com/visin-platform/visin-fusion/compare/v1.1.0...HEAD" in text
    assert "[1.1.0]: https://github.com/visin-platform/visin-fusion/compare/v1.0.0...v1.1.0" in text
    assert "## [1.0.0] - 2026-01-01" in text
    assert text.count("[Unreleased]:") == 1


def test_the_first_release_links_to_its_tag():
    text = release.update_changelog(
        "# Changelog\n\n## [Unreleased]\n\n- Everything.\n", "0.1.0", None, {}, "2026-09-25"
    )
    assert "[0.1.0]: https://github.com/visin-platform/visin-fusion/releases/tag/v0.1.0" in text
    assert release.release_notes(text, "0.1.0") == "- Everything.\n"


def test_a_release_with_nothing_to_say_says_so():
    text = release.update_changelog("# Changelog\n\n## [Unreleased]\n", "1.0.1", "1.0.0", {}, "2026-09-25")
    assert release.release_notes(text, "1.0.1") == "No user-facing changes.\n"


def test_notes_for_a_missing_version_are_an_error():
    with pytest.raises(ValueError):
        release.release_notes(CHANGELOG, "9.9.9")


def test_hand_written_paragraphs_and_section_order_are_kept():
    unreleased = "- A.\n- B.\n\nA paragraph.\n\n### Changes to results\n\n- C.\n\n### Fixed\n\n- D."
    text = release.update_changelog(f"# Changelog\n\n## [Unreleased]\n\n{unreleased}\n", "0.1.0", None,
                                    release.render_entry(commits("feat: E")), "2026-09-25")
    assert release.release_notes(text, "0.1.0") == unreleased + "\n\n### Added\n\n- E\n"


def test_the_real_changelog_can_be_released():
    """The next release, made from the changelog as it stands: what is under Unreleased becomes its
    entry, word for word, and any versions already released are left as they were."""
    text = (SCRIPT.parents[1] / "CHANGELOG.md").read_text()
    current = release.read_version()
    released_before = f"## [{current}]" in text
    upcoming = release.next_version(current, [], "minor") if released_before else current
    unreleased = re.search(r"^## \[Unreleased\]\n(.*?)(?=^## \[|^\[[^\]]+\]: |\Z)", text, re.M | re.S).group(1).strip()
    released = release.update_changelog(text, upcoming, current if released_before else None, {}, "2026-09-25")
    assert release.release_notes(released, upcoming).strip() == (unreleased or "No user-facing changes.")
    if released_before:
        assert release.release_notes(released, current) == release.release_notes(text, current)
