"""The suite must not write into the developer's home directory.

Agent binaries, wheels, node and ripgrep resolve their download directory through
`appdirs.package_cache_dir`, which `mkdir(parents=True)`s eagerly — so merely *resolving*
a path creates it. Tests that mock an installer but not the resolution still wrote
`~/.cache/inspect_swe/...`, and `minisweagent` created `~/.config/mini-swe-agent` at
import. `tests/conftest.py` redirects both; these tests fail if either redirect is
removed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import minisweagent
from inspect_swe._util.appdirs import package_cache_dir, package_data_dir

# One known offender is enough for the subprocess check: it exercises the eager mkdir in
# a real interpreter, including the import-time directory creation a fixture cannot
# reach. Running the entire suite in a subprocess would double its runtime to re-prove
# what the conftest redirect already covers for every module.
_REPRESENTATIVE_TEST_FILE = "tests/test_codex_agentbinary.py"

# The live catalog drift check makes a real network request and is independently
# runnable in the outer suite; it has nothing to do with home-directory isolation, and
# a network failure/skip here must not be conflated with a home-isolation regression.
_LIVE_DRIFT_TEST = (
    f"{_REPRESENTATIVE_TEST_FILE}::test_bundled_catalog_tracks_live_latest"
)


def _is_under(path: Path, parent: Path) -> bool:
    return parent.resolve() in (path.resolve(), *path.resolve().parents)


def test_package_dirs_never_resolve_into_the_real_home() -> None:
    """The cache/data roots are redirected, so resolving one cannot touch $HOME."""
    home = Path.home()
    for resolved in (package_cache_dir("probe"), package_data_dir("probe")):
        assert resolved.exists(), f"{resolved} should have been created eagerly"
        assert not _is_under(resolved, home), (
            f"{resolved} is inside {home}: the conftest redirect is not in force, and "
            "running the suite writes into the developer's home directory"
        )


def test_mini_swe_agent_global_config_is_redirected_out_of_the_real_home() -> None:
    """`minisweagent` mkdir()s its config dir at import; it must not land in $HOME."""
    config_dir = Path(minisweagent.global_config_dir)
    assert not _is_under(config_dir, Path.home()), (
        f"{config_dir} is inside {Path.home()}: MSWEA_GLOBAL_CONFIG_DIR must be set in "
        "pytest_configure, before collection imports minisweagent"
    )


def test_a_real_pytest_run_writes_nothing_into_a_clean_home() -> None:
    """End-to-end: a fresh interpreter running real tests leaves an empty $HOME."""
    repository_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="inspect-swe-home-probe-") as clean_home:
        environment = dict(os.environ)
        environment["HOME"] = clean_home
        # Drop every redirect so the subprocess must re-establish isolation itself;
        # inheriting them would prove nothing about the conftest under test.
        for inherited in ("XDG_CACHE_HOME", "XDG_DATA_HOME", "MSWEA_GLOBAL_CONFIG_DIR"):
            environment.pop(inherited, None)

        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                _REPRESENTATIVE_TEST_FILE,
                "--deselect",
                _LIVE_DRIFT_TEST,
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=repository_root,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

        written = sorted(
            path.relative_to(clean_home).as_posix()
            for path in Path(clean_home).rglob("*")
        )
        assert not written, (
            f"{_REPRESENTATIVE_TEST_FILE} wrote into a clean home directory: {written}"
        )
