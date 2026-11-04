"""Tests for the dependency cooldown in pyproject.toml and the CI jobs around it.

pyproject.toml holds back releases younger than seven days, or six hours for a
list of model provider SDKs and Inspect packages. Dependabot applies the seven
days to its uv.lock bumps and leaves that list to uv. CI installs from uv.lock,
and the build toolchain from scripts/build-dist/requirements.txt, pinned with
hashes.
Testing against the newest releases happens in meridianlabs-ai/canaries,
outside this repository.
"""

import pathlib
import re
import tomllib
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import yaml

REPO = pathlib.Path(__file__).parents[1]
GITHUB = REPO / ".github"
WORKFLOWS = GITHUB / "workflows"
SHARED_ACTIONS = "meridianlabs-ai/actions/"
SHARED_INSTALLERS = ("setup-uv@", "install-python-dependencies@")
EXEMPT_COOLDOWN = "6 hours"
BUILD_REQUIREMENTS = "scripts/build-dist/requirements.txt"


def uv_config() -> dict[str, Any]:
    config: dict[str, Any] = tomllib.loads((REPO / "pyproject.toml").read_text())[
        "tool"
    ]["uv"]
    return config


def normalize(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package).lower()


def direct_dependencies() -> set[str]:
    """Packages in the pyproject.toml `dependencies` or the `dev` extra."""
    project = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    requirements = project["dependencies"] + project["optional-dependencies"]["dev"]
    return {
        normalize(re.split(r"[^A-Za-z0-9._-]", requirement, maxsplit=1)[0])
        for requirement in requirements
    }


def exempt_packages(config: dict[str, Any]) -> list[str]:
    return sorted(
        package
        for package, cooldown in config["exclude-newer-package"].items()
        if cooldown == EXEMPT_COOLDOWN
    )


def dependabot_uv() -> dict[str, Any]:
    updates = yaml.safe_load((GITHUB / "dependabot.yml").read_text())["updates"]
    root = [
        u for u in updates if u["package-ecosystem"] == "uv" and u["directory"] == "/"
    ]
    assert len(root) == 1, "expected one uv entry for directory / in dependabot.yml"
    entry: dict[str, Any] = root[0]
    return entry


def jobs() -> list[tuple[str, dict[str, Any], Any]]:
    """Every job of every workflow, with its location and its permissions."""
    return [
        (
            f"{path.name}:{name}",
            job,
            job.get("permissions", workflow.get("permissions")),
        )
        for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
        for workflow in [yaml.safe_load(path.read_text())]
        for name, job in workflow["jobs"].items()
    ]


def steps() -> list[tuple[str, dict[str, Any]]]:
    """Every step of every workflow, with its location."""
    return [(where, step) for where, job, _ in jobs() for step in job.get("steps", [])]


def run_lines() -> list[tuple[str, str]]:
    # Joins backslash continuations so a command's flags stay on one line.
    return [
        (where, line.strip())
        for where, step in steps()
        for line in step.get("run", "").replace("\\\n", " ").splitlines()
        if line.strip()
    ]


def verify_exclude_newer_package(
    config: dict[str, Any], direct: set[str], now: datetime
) -> None:
    assert config["exclude-newer"] == "7 days", config["exclude-newer"]
    # Each entry is "6 hours" or a dated entry younger than the cooldown.
    for package, cooldown in config["exclude-newer-package"].items():
        assert normalize(package) in direct, f"{package} is not a direct dependency"
        if cooldown != EXEMPT_COOLDOWN:
            age = now - datetime.fromisoformat(cooldown)
            assert timedelta(0) <= age < timedelta(days=7), (
                f'dated entry {package} = "{cooldown}" is in the future or stale'
            )


def test_exclude_newer_package_entries() -> None:
    verify_exclude_newer_package(uv_config(), direct_dependencies(), datetime.now(UTC))


def test_non_direct_entry_is_caught() -> None:
    config = uv_config()
    config["exclude-newer-package"]["certifi"] = EXEMPT_COOLDOWN
    with pytest.raises(AssertionError, match="not a direct dependency"):
        verify_exclude_newer_package(config, direct_dependencies(), datetime.now(UTC))


@pytest.mark.parametrize(
    "age,caught",
    [(timedelta(days=6), False), (timedelta(days=7), True), (-timedelta(days=1), True)],
    ids=["fresh", "stale", "future"],
)
def test_dated_entry_age(age: timedelta, caught: bool) -> None:
    config = uv_config()
    now = datetime.now(UTC)
    config["exclude-newer-package"]["httpx"] = (now - age).isoformat()
    if caught:
        with pytest.raises(AssertionError, match="dated entry"):
            verify_exclude_newer_package(config, direct_dependencies(), now)
    else:
        verify_exclude_newer_package(config, direct_dependencies(), now)


def test_dependabot_cooldown_matches_pyproject() -> None:
    dependabot = dependabot_uv()
    assert dependabot["cooldown"]["default-days"] == 7
    # Dependabot's cooldown counts whole days, so it leaves the "6 hours"
    # entries to uv. Dated entries are locked by hand.
    exempt = exempt_packages(uv_config())
    assert sorted(dependabot["cooldown"]["exclude"]) == exempt
    # The exempt packages have their own group, listed before the catch-all.
    groups = dependabot["groups"]
    assert list(groups) == ["six-hour-cooldown", "python-packages"]
    assert sorted(groups["six-hour-cooldown"]["patterns"]) == exempt
    assert groups["python-packages"]["patterns"] == ["*"]


def test_python_comes_from_the_shared_actions() -> None:
    direct = [
        (where, step["uses"])
        for where, step in steps()
        if step.get("uses", "").startswith(
            ("astral-sh/setup-uv@", "actions/setup-python@")
        )
    ]
    assert direct == [], "use meridianlabs-ai/actions/install-python-dependencies"


def test_shared_actions_share_one_pinned_commit_and_pin_versions() -> None:
    # One commit for every shared action means one uv.
    shared = [
        (where, step)
        for where, step in steps()
        if step.get("uses", "").startswith(
            tuple(SHARED_ACTIONS + name for name in SHARED_INSTALLERS)
        )
    ]
    assert shared, "no workflow uses the shared install actions"
    refs = {step["uses"].split("@")[1].split()[0] for _, step in shared}
    assert len(refs) == 1, f"shared actions pin different commits: {refs}"
    assert re.fullmatch(r"[0-9a-f]{40}", refs.pop()), "pin a 40-character SHA"
    unlocked = [
        where
        for where, step in shared
        if "install-python-dependencies@" in step["uses"]
        and step["with"]["versions"] not in ("locked", "hashed-requirements")
    ]
    assert unlocked == [], f"use versions: locked or hashed-requirements in {unlocked}"


def test_jobs_that_can_write_use_no_cache() -> None:
    # A cache that a write-capable job restores could run code from a less
    # trusted run. Jobs without permissions get the read-only repo default.
    cached = [
        where
        for where, job, permissions in jobs()
        if permissions == "write-all"
        or "write" in (permissions.values() if isinstance(permissions, dict) else [])
        for step in job.get("steps", [])
        if step.get("uses", "").startswith(
            SHARED_ACTIONS + "install-python-dependencies@"
        )
        and step["with"].get("actions-cache") != "none"
    ]
    assert cached == [], f"use actions-cache: none in {cached}"


def test_ruff_comes_from_uv_lock() -> None:
    # ruff-action installs its own ruff, outside the shared action.
    ruff = [
        w for w, s in steps() if s.get("uses", "").startswith("astral-sh/ruff-action@")
    ]
    assert ruff == [], f"run ruff with uv run --no-sync in {ruff}"


def test_build_toolchain_is_hashed_wheels() -> None:
    hashed = [
        (where, step["with"])
        for where, step in steps()
        if step.get("uses", "").startswith(
            SHARED_ACTIONS + "install-python-dependencies@"
        )
        and step["with"]["versions"] == "hashed-requirements"
    ]
    assert hashed, "no workflow installs the hashed build toolchain"
    wrong = [
        where
        for where, inputs in hashed
        if inputs.get("package-format") != "wheels-only"
        or inputs.get("requirements-file") != BUILD_REQUIREMENTS
    ]
    assert wrong == [], f"use wheels-only and {BUILD_REQUIREMENTS} in {wrong}"


def test_builds_use_the_installed_toolchain() -> None:
    # An isolated build resolves its own build backend, outside the hashes.
    builds = [(w, line) for w, line in run_lines() if re.search(r"-m build\b", line)]
    assert builds, "no workflow builds the package"
    isolated = [(w, line) for w, line in builds if "--no-isolation" not in line.split()]
    assert isolated == [], f"build with --no-isolation in {isolated}"
    uv_build = [
        (w, line) for w, line in run_lines() if re.search(r"\buv build\b", line)
    ]
    assert uv_build == [], f"uv build resolves the build backend in {uv_build}"


def test_build_toolchain_keeps_the_hatchling_cap() -> None:
    def hatchling(requirements: list[str]) -> str:
        [cap] = [r for r in requirements if normalize(r).startswith("hatchling")]
        return cap

    pinned = (REPO / "scripts/build-dist/requirements.in").read_text().splitlines()
    build_system = tomllib.loads((REPO / "pyproject.toml").read_text())["build-system"]
    assert hatchling(pinned) == hatchling(build_system["requires"])


# pip installs skip uv.lock; installs go through the shared action.
def test_no_workflow_runs_pip_install() -> None:
    pip = [
        (where, line)
        for where, line in run_lines()
        if re.search(r"\bpip3? install\b", line)
    ]
    assert pip == [], "pip installs skip uv.lock; use the shared action"


def test_no_workflow_opts_out_of_the_cooldown() -> None:
    floating = [
        (where, line)
        for where, line in run_lines()
        if re.search(r"--exclude-newer(=|\s+)false", line)
    ]
    assert floating == []
