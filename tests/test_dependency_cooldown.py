"""Tests for the dependency cooldown in pyproject.toml and the CI jobs around it.

pyproject.toml holds back releases younger than seven days, or six hours for a
list of model provider SDKs and Inspect packages. Dependabot applies the seven
days to its uv.lock bumps and leaves that list to uv. CI installs from uv.lock,
except that inspect-ai moves to inspect_ai main's latest commit first, and the
build toolchain from scripts/build-dist/requirements.txt, pinned with hashes.
Testing against the newest releases happens in meridianlabs-ai/canaries,
outside this repository.
"""

import importlib.util
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
UV_SYNC_ACTION = "./.github/actions/uv-sync"
SHARED_ACTIONS = "meridianlabs-ai/actions/"
SHARED_INSTALLERS = ("setup-uv@", "install-python-dependencies@")
EXEMPT_COOLDOWN = "6 hours"
BUILD_REQUIREMENTS = "scripts/build-dist/requirements.txt"
SHARED_BUILD = SHARED_ACTIONS + ".github/workflows/build-python-dist.yml@"
PUBLISH_STEPS = ("actions/download-artifact@", "pypa/gh-action-pypi-publish@")
DOWNLOAD_INPUTS = {
    "artifact-ids": "${{ needs.build.outputs.artifact-id }}",
    "path": "dist/",
}
RELEASE_BRANCH = "release-please--branches--main"
INSPECT_AI_MAIN = {
    "git": "https://github.com/UKGovernmentBEIS/inspect_ai",
    "branch": "main",
}
MATRIX_PYTHON = "${{ matrix.python-version }}"
LOCKED_JOBS = {
    "py-ruff": {
        "python-version": MATRIX_PYTHON,
        "project-install": "skip",
        "extras": "dev",
    },
    "py-mypy": {"python-version": MATRIX_PYTHON, "extras": "dev"},
    "py-test": {"python-version": MATRIX_PYTHON, "extras": "dev"},
    "py-build": {"python-version": MATRIX_PYTHON, "project-install": "skip"},
}

spec = importlib.util.spec_from_file_location(
    "inspect_ai_floor", GITHUB / "scripts" / "inspect_ai_floor.py"
)
assert spec and spec.loader
inspect_ai_floor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspect_ai_floor)


def pyproject() -> dict[str, Any]:
    return tomllib.loads((REPO / "pyproject.toml").read_text())


def uv_config() -> dict[str, Any]:
    config: dict[str, Any] = pyproject()["tool"]["uv"]
    return config


def normalize(package: str) -> str:
    return re.sub(r"[-_.]+", "-", package).lower()


def direct_dependencies() -> set[str]:
    """Packages in the pyproject.toml `dependencies` or the `dev` extra."""
    project = pyproject()["project"]
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


def local_actions() -> list[pathlib.Path]:
    return sorted((GITHUB / "actions").glob("*/action.y*ml"))


def steps() -> list[tuple[str, dict[str, Any]]]:
    """Every step of every workflow and local composite action, with its location."""
    return [
        (where, step) for where, job, _ in jobs() for step in job.get("steps", [])
    ] + [
        (path.parent.name, step)
        for path in local_actions()
        for step in yaml.safe_load(path.read_text())["runs"].get("steps", [])
    ]


def step_lines(steps: list[dict[str, Any]]) -> list[str]:
    # Joins backslash continuations so a command's flags stay on one line.
    return [
        line.strip()
        for step in steps
        for line in step.get("run", "").replace("\\\n", " ").splitlines()
        if line.strip()
    ]


def run_lines() -> list[tuple[str, str]]:
    return [(where, line) for where, step in steps() for line in step_lines([step])]


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
    # Dependabot resolves against the inspect-ai commit in uv.lock, so it must
    # be able to update it.
    assert "ignore" not in dependabot
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
        if step.get("uses") == UV_SYNC_ACTION
        or (
            step.get("uses", "").startswith(
                SHARED_ACTIONS + "install-python-dependencies@"
            )
            and step["with"].get("actions-cache") != "none"
        )
    ]
    assert cached == [], (
        f"use actions-cache: none (not {UV_SYNC_ACTION}, which always uses "
        f"the cache) in {cached}"
    )


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


def test_publish_jobs_only_upload_the_shared_build() -> None:
    # A job with `id-token: write` only downloads the dist and publishes it.
    # The build runs in the shared workflow, in a job with no publishing
    # credential. A job that calls a shared workflow, such as the docs deploy,
    # has no steps here.
    publishers = [
        (where, job)
        for where, job, permissions in jobs()
        if isinstance(permissions, dict)
        and permissions.get("id-token") == "write"
        and "uses" not in job
    ]
    assert publishers, "no job holds id-token: write"
    wrong = [
        where
        for where, job in publishers
        if len(job["steps"]) != len(PUBLISH_STEPS)
        or not all(
            step.get("uses", "").startswith(prefix)
            for step, prefix in zip(job["steps"], PUBLISH_STEPS, strict=True)
        )
    ]
    assert wrong == [], f"only download the dist and publish it in {wrong}"
    unlinked = [
        where
        for where, job in publishers
        for needs in [job.get("needs", [])]
        if "build" not in ([needs] if isinstance(needs, str) else needs)
        or job["steps"][0].get("with") != DOWNLOAD_INPUTS
    ]
    assert unlinked == [], f"download only the shared build artifact in {unlinked}"
    build = yaml.safe_load((WORKFLOWS / "publish.yml").read_text())["jobs"]["build"]
    assert build["uses"].startswith(SHARED_BUILD), build["uses"]
    ref = build["uses"].split("@")[1].split()[0]
    assert re.fullmatch(r"[0-9a-f]{40}", ref), "pin a 40-character SHA"


def test_build_toolchain_keeps_the_hatchling_cap() -> None:
    def hatchling(requirements: list[str]) -> str:
        [cap] = [r for r in requirements if normalize(r).startswith("hatchling")]
        return cap

    pinned = (REPO / "scripts/build-dist/requirements.in").read_text().splitlines()
    build_system = pyproject()["build-system"]
    assert hatchling(pinned) == hatchling(build_system["requires"])


# pip installs skip uv.lock, so they may only add local packages without their
# dependencies, or install a hashed lock (`-r` or `uv pip sync`).
def test_pip_installs_are_dependency_free_or_hashed() -> None:
    unlocked = [
        (where, line)
        for where, line in run_lines()
        if (re.search(r"\bpip3? install\b", line) and "--no-deps" not in line.split())
        or (
            re.search(r"\bpip sync\b|\bpip3? install\b.* -r\b", line)
            and "--require-hashes" not in line.split()
        )
    ]
    assert unlocked == []


def test_no_workflow_opts_out_of_the_cooldown() -> None:
    floating = [
        (where, line)
        for where, line in run_lines()
        if re.search(r"--exclude-newer(=|\s+)false", line)
    ]
    assert floating == []


def test_pyproject_tracks_inspect_ai_main_with_one_floor() -> None:
    project = pyproject()
    assert project["tool"]["uv"]["sources"] == {"inspect-ai": INSPECT_AI_MAIN}
    # release-pin-deps.yml rewrites this one requirement with sed.
    floors = [d for d in project["project"]["dependencies"] if "inspect-ai>=" in d]
    assert len(floors) == 1, floors


@pytest.mark.parametrize("name,inputs", LOCKED_JOBS.items())
def test_locked_jobs_install_from_the_lockfile(
    name: str, inputs: dict[str, str]
) -> None:
    job = yaml.safe_load((WORKFLOWS / "build.yaml").read_text())["jobs"][name]
    installs = [
        step.get("with", {})
        for step in job["steps"]
        if step.get("uses") == UV_SYNC_ACTION
    ]
    assert installs == [inputs]
    assert not [line for line in step_lines(job["steps"]) if line.startswith("uv sync")]


def test_uv_sync_action_moves_only_inspect_ai() -> None:
    # Moving inspect-ai re-locks, which would also absorb a pyproject.toml
    # change nobody locked, so the committed lock is checked first. The
    # release PR resolves without sources, and later steps see the committed
    # uv.lock.
    action = yaml.safe_load((GITHUB / "actions" / "uv-sync" / "action.yml").read_text())
    setup, move, install, restore = action["runs"]["steps"]
    assert setup["uses"].startswith(SHARED_ACTIONS + "setup-uv@")
    assert setup["with"] == {"python-version": "${{ inputs.python-version }}"}
    assert move["env"]["HEAD_REF"] == "${{ github.head_ref }}"
    assert step_lines([move]) == [
        "uv lock --check",
        f'if [ "$HEAD_REF" = "{RELEASE_BRANCH}" ]; then',
        "export UV_NO_SOURCES=1",
        'echo "UV_NO_SOURCES=1" >> "$GITHUB_ENV"',
        "fi",
        "uv lock --upgrade-package inspect-ai",
    ]
    assert install["uses"].startswith(SHARED_ACTIONS + "install-python-dependencies@")
    assert install["with"] == {
        "versions": "locked",
        "python-version": "${{ inputs.python-version }}",
        "extras": "${{ inputs.extras }}",
        "project-install": "${{ inputs.project-install }}",
    }
    assert step_lines([restore]) == ["git checkout -- uv.lock"]


def test_release_branch_name_matches_everywhere() -> None:
    # A misspelt name silently runs the release PR as a dev PR.
    names = {
        where: set(re.findall(r"release-please--branches--[\w-]+", text))
        for where, text in [
            *[(path.name, path.read_text()) for path in WORKFLOWS.glob("*.y*ml")],
            ("uv-sync", (GITHUB / "actions" / "uv-sync" / "action.yml").read_text()),
        ]
    }
    assert set().union(*names.values()) == {RELEASE_BRANCH}
    assert names["uv-sync"] and names["release-pin-deps.yml"]


# A bare `uv run` or `uv sync` re-resolves an out-of-date uv.lock and installs
# the result. `uv run --no-project` ignores the project and uv.lock. `--with`
# installs unlocked packages whatever the other flags.
def test_uv_never_relocks() -> None:
    unlocked = [
        (where, line)
        for where, line in run_lines()
        if re.search(r"\buv (run|sync)\b", line)
        and (
            "--with" in line
            or not {"--locked", "--no-sync", "--no-project"} & set(line.split())
        )
    ]
    assert unlocked == []


def test_floor_script_requires_a_released_inspect_ai() -> None:
    lock = """
[[package]]
name = "inspect-ai"
version = "0.3.287.dev30"
source = { git = "https://github.com/UKGovernmentBEIS/inspect_ai?branch=main#abc" }
"""
    with pytest.raises(ValueError, match="not locked from PyPI"):
        inspect_ai_floor.locked_inspect_ai(lock)
    released = lock.replace('version = "0.3.287.dev30"', 'version = "0.3.287"').replace(
        'git = "https://github.com/UKGovernmentBEIS/inspect_ai?branch=main#abc"',
        'registry = "https://pypi.org/simple"',
    )
    assert inspect_ai_floor.locked_inspect_ai(released) == "0.3.287"
