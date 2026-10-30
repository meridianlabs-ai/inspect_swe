import importlib
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Callable, List, Literal, TypeVar, cast
from unittest.mock import MagicMock

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog, EvalSample
from inspect_ai.model import ChatMessageAssistant
from inspect_swe._util import appdirs


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runapi", action="store_true", default=False, help="run API tests"
    )
    parser.addoption(
        "--runflaky", action="store_true", default=False, help="run flaky tests"
    )


_HOME_ISOLATION_ROOT: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    # `minisweagent/__init__.py` mkdir()s its global config directory AT IMPORT, so a
    # fixture cannot get there first -- collection imports the test modules. It honours
    # MSWEA_GLOBAL_CONFIG_DIR, so redirect before anything imports it.
    global _HOME_ISOLATION_ROOT
    _HOME_ISOLATION_ROOT = Path(tempfile.mkdtemp(prefix="inspect-swe-test-home-"))
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(_HOME_ISOLATION_ROOT / "mini-swe-agent")

    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "api: mark test as requiring API access")
    config.addinivalue_line("markers", "flaky: mark test as flaky/unreliable")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--runapi"):
        skip_api = pytest.mark.skip(reason="need --runapi option to run")
        for item in items:
            if "api" in item.keywords:
                item.add_marker(skip_api)

    if not config.getoption("--runflaky"):
        skip_flaky = pytest.mark.skip(reason="need --runflaky option to run")
        for item in items:
            if "flaky" in item.keywords:
                item.add_marker(skip_flaky)


def skip_if_env_var(var: str, exists: bool = True) -> pytest.MarkDecorator:
    """Pytest mark to skip the test if the var environment variable is not defined."""
    condition = (var in os.environ.keys()) if exists else (var not in os.environ.keys())
    return pytest.mark.skipif(
        condition,
        reason=f"Test doesn't work without {var} environment variable defined.",
    )


F = TypeVar("F", bound=Callable[..., Any])


def skip_if_no_openai(func: F) -> F:
    return cast(
        F,
        pytest.mark.skipif(
            importlib.util.find_spec("openai") is None
            or os.environ.get("OPENAI_API_KEY") is None,
            reason="Test requires both OpenAI package and OPENAI_API_KEY environment variable",
        )(func),
    )


def skip_if_no_anthropic(func: F) -> F:
    return cast(F, skip_if_env_var("ANTHROPIC_API_KEY", exists=False)(func))


def skip_if_no_google(func: F) -> F:
    return cast(F, skip_if_env_var("GOOGLE_API_KEY", exists=False)(func))


def skip_if_github_action(func: F) -> F:
    return cast(F, skip_if_env_var("GITHUB_ACTIONS", exists=True)(func))


def is_docker_available() -> bool:
    """Check if Docker is available on the system."""
    try:
        return (
            subprocess.run(
                ["docker", "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ).returncode
            == 0
        )
    except FileNotFoundError:
        return False


def skip_if_no_docker(func: F) -> F:
    return cast(
        F,
        pytest.mark.skipif(
            not is_docker_available(),
            reason="Test doesn't work without Docker installed.",
        )(func),
    )


def is_k8s_available() -> bool:
    """Check if Kubernetes is available on the system.

    Detects Kubernetes by checking if kubectl can connect to a cluster
    by running 'kubectl version --client=false'.
    """
    try:
        return (
            subprocess.run(
                ["kubectl", "version", "--client=false"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5,  # Add timeout to prevent hanging
            ).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def skip_if_no_k8s(func: F) -> F:
    """Skip test if we don't have access to a Kubernetes cluster."""
    return cast(
        F,
        pytest.mark.skipif(
            not is_k8s_available(),
            reason="Test requires a connection to a Kubernetes cluster.",
        )(func),
    )


def get_available_sandboxes() -> List[Literal["docker", "k8s"]]:
    """Return a list of available sandbox environments.

    This function checks if docker and/or kubernetes are available
    on the system and returns a list of available sandbox types.
    """
    available_sandboxes: list[Literal["docker", "k8s"]] = []

    # Check if Docker is available
    if is_docker_available():
        available_sandboxes.append("docker")

    # Check if Kubernetes is available
    if is_k8s_available():
        available_sandboxes.append("k8s")

    return available_sandboxes


def run_example(
    example: str,
    agent: Literal[
        "claude_code",
        "codex_cli",
        "gemini_cli",
        "kimi_code",
        "mini_swe_agent",
        "opencode",
        "pi",
    ],
    model: str,
    sandbox: str | None = None,
    *,
    assert_completed: bool = True,
) -> list[EvalLog]:
    example_file = os.path.join("examples", example)
    task_args: dict[str, str] = {
        "agent": agent,
    }

    if sandbox is not None:
        task_args["sandbox"] = sandbox

    # Bound the agent. These examples are trivial by construction -- guess a
    # number without calling tools, answer "what is 2+2" -- so a run that
    # reaches either limit is a runaway CLI loop, not slow honest work.
    # Nothing else bounds them: the tasks set no limits and the agents'
    # sbox.exec_remote() calls pass no timeout, so the only backstop is the
    # pytest timeout, which reports as an opaque stall (and, under
    # --timeout-method=thread, as a bare "worker crashed"). Seen in
    # meridianlabs-ai/actions run 31968805891, where gemini_cli looped for
    # 9m21s and 1.68M tokens on multiple_attempts before pytest killed it.
    #
    # Both limits earn their place: time catches a wedged process that has
    # stopped generating, tokens cap the cost of one that has not.
    #
    # The token ceiling is deliberately loose. It is shared by every example
    # (system_explorer and web_search are genuinely multi-turn), and cached
    # reads count toward it -- they were 1.4M of that 1.68M -- so a tight cap
    # would clip honest runs. 500k still trips a runaway inside ~3 minutes,
    # ahead of the time limit, while sitting well clear of real usage.
    #
    # gemini_cli and mini_swe_agent get double the time. gemini_cli's healthy
    # multi-turn runs (multi_call, skills) already take ~300s, so any burst of
    # Gemini API retries pushes it over the limit and fails the run with a
    # truncated transcript rather than an obvious timeout (e.g.
    # meridianlabs-ai/actions run 33373083766, where 8 HTTP retries left only
    # 1 of the expected 4+ user messages). mini_swe_agent reached the same
    # regime with gpt-5-mini: it drives ~10 sequential reasoning calls for the
    # four multi_call questions, healthy runs range 95-200s, and run
    # 34455110720 hit 300s with the same 8-retry / 1-user-message signature.
    # The allowance is keyed on the agent, not the example, so it deliberately
    # widens the runaway window for every mini_swe_agent example (as it
    # already does for gemini_cli).
    #
    # 600s leaves ~300s under the nightly's 900s pytest --timeout, but the
    # margin is thinner than it looks: Inspect starts the sample clock after
    # sandbox init, so docker pull/start/teardown come out of that 300s, and
    # scoring gets a further time_limit / 2. That scoring grant is moot today
    # (no example defines a scorer), but adding one would put the worst case
    # at exactly 900s -- revisit the limits if that happens.
    logs = eval(
        example_file,
        model=model,
        limit=1,
        task_args=task_args,
        time_limit=600 if agent in ("gemini_cli", "mini_swe_agent") else 300,
        token_limit=500_000,
    )

    # Every caller gets the completion check, because a run that did not
    # actually finish is never a result worth asserting on: a sample cut short
    # by a limit or an error, a task that failed outright, a sample the agent
    # never answered (the two helpers below say what each check is for). Left
    # to the individual tests it either surfaces as a misleading downstream
    # assertion (the multi_call solver only copies the agent's messages back
    # after all four turns, so a truncated sample reports a bare
    # "assert 1 >= 4") or as a vacuous pass. On 2026-09-15
    # test_gemini_cli_web_search ran 604.81s against its 600s time limit and
    # still reported green, because the one search call it asserts on had
    # already happened before the limit hit.
    # A test that legitimately expects a limit passes assert_completed=False.
    if assert_completed:
        for log in logs:
            for sample in assert_eval_completed(log):
                _assert_agent_turn(sample)

    return logs


def _assert_agent_turn(sample: EvalSample) -> None:
    """Fail loudly if the agent never answered.

    Every example is an agent turn -- the solver is the CLI agent itself, or
    (multi_call, image_input) a solver that runs it -- so a sample with no
    assistant message is a run where the agent never took one. None of them
    expects zero turns. Inspect records such a sample as a plain success, with
    no limit and no error, so a test that checks only `log.samples` passes on
    it.
    """
    assert any(
        isinstance(message, ChatMessageAssistant) for message in sample.messages
    ), "sample has no assistant message: the agent never took a turn"


def assert_eval_completed(log: EvalLog) -> list[EvalSample]:
    """Fail loudly unless the eval and every sample in it ran to completion.

    Returns the samples, so callers can keep asserting on them.

    The three checks catch three different failures, and none of them covers
    another:

    - `log.status` is the *task* verdict. It is "error" when the task itself
      raised or when enough samples errored to trip `fail_on_error`
      (`_should_eval_fail` in `inspect_ai/_eval/task/error.py`, applied in
      `_eval/task/run.py`); these tests never pass `fail_on_error`, so the
      default applies and a single errored sample is enough. A task
      failure that happens before the log is opened (unresolvable model, a
      sandbox that will not build) raises out of `eval()` instead, so it
      already fails loudly.
    - `log.samples` can be empty on a non-success log, which used to reach the
      caller as a bare `assert log.samples` with no explanation of what broke.
    - Neither status check sees a *limit*. A sample cut short by its time or
      token limit is recorded with `sample.limit` set, no error, and a task
      status of "success" -- that is the false pass this suite kept hitting.
    """
    # `log.error` carries the traceback and an ANSI-rendered copy of it as
    # well; only the message belongs in an assertion, the rest is in the log.
    assert log.status == "success", f"eval did not complete: status={log.status}" + (
        f", error={log.error.message}" if log.error is not None else ""
    )
    assert log.samples, f"eval reported {log.status} with no samples"
    for sample in log.samples:
        assert sample.limit is None, (
            f"sample hit a {sample.limit.type} limit ({sample.limit.limit})"
        )
        assert sample.error is None, (
            f"sample errored: {sample.error.message}\n{sample.error.traceback}"
        )
    return log.samples


# --- Wheels cache utilities ---


@pytest.fixture
def wheels_cache_cleanup() -> Any:
    """Fixture that redirects wheels cache to a temp directory for test isolation.

    Tests using this fixture will have their cache operations isolated from the
    real cache directory. The temp directory is automatically cleaned up after
    the test completes (even if it fails).
    """
    import shutil
    import tempfile
    from pathlib import Path
    from unittest.mock import patch

    # Create temp directory for test cache
    temp_dir = Path(tempfile.mkdtemp(prefix="wheels_cache_test_"))

    def mock_cache_dir(package_name: str) -> Path:
        safe_name = package_name.replace("-", "_").replace(".", "_")
        cache_path = temp_dir / f"{safe_name}-wheels"
        cache_path.mkdir(parents=True, exist_ok=True)
        return cache_path

    with patch("inspect_swe._util.agentwheel._wheels_cache_dir", mock_cache_dir):
        yield temp_dir

    # Cleanup temp directory (runs even if test fails)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_pip_download_failure() -> Any:
    """Fixture to mock pip download network failures.

    Mocks both the cache read (to force download) and subprocess.run (to simulate failure).
    """
    from unittest.mock import patch

    # Mock cache to return None (force download path)
    with (
        patch("inspect_swe._util.agentwheel.read_cached_wheels", return_value=None),
        patch(
            "inspect_swe._util.agentwheel.importlib.util.find_spec",
            return_value=MagicMock(),
        ),
    ):
        # Mock subprocess.run to simulate pip download failure
        with patch("inspect_swe._util.agentwheel.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stderr="ERROR: Could not find a version that satisfies the requirement (network error)",
            )
            yield mock_run


@pytest.fixture(autouse=True, scope="session")
def _package_dirs_stay_out_of_the_real_home() -> Iterator[None]:
    """Redirect the package cache/data roots so the suite never writes into $HOME.

    Agent binaries, wheels, node and ripgrep all resolve their download directory
    through `appdirs.package_cache_dir`, which `mkdir(parents=True)`s eagerly -- so
    merely *resolving* a path creates it. A test that mocks the installer but not the
    resolution still writes `~/.cache/inspect_swe/...` on the developer's machine.

    The patch goes on `user_cache_path`/`user_data_path` as imported into
    `_util.appdirs`, NOT on `package_cache_dir` itself: eleven modules do
    `from .._util.appdirs import package_cache_dir`, so each holds its own reference and
    patching that name would miss every one of them. Patching platformdirs' entry points
    inside the one module that calls them covers all callers however they imported.

    Setting `XDG_CACHE_HOME` would also work on Linux, and only on Linux: platformdirs
    resolves macOS caches to `~/Library/Caches` and ignores XDG, so that variant would
    look green in CI while still writing to a developer's home.
    """
    root = Path(tempfile.mkdtemp(prefix="inspect-swe-test-dirs-"))
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(appdirs, "user_cache_path", lambda _package: root / "cache")
    monkeypatch.setattr(appdirs, "user_data_path", lambda _package: root / "data")
    try:
        yield
    finally:
        monkeypatch.undo()
        shutil.rmtree(root, ignore_errors=True)


def pytest_unconfigure(config: pytest.Config) -> None:
    if _HOME_ISOLATION_ROOT is not None:
        shutil.rmtree(_HOME_ISOLATION_ROOT, ignore_errors=True)
