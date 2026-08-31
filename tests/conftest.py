import importlib
import os
import subprocess
from typing import Any, Callable, List, Literal, TypeVar, cast
from unittest.mock import MagicMock

import pytest
from inspect_ai import eval
from inspect_ai.log import EvalLog


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


def pytest_configure(config: pytest.Config) -> None:
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
    ],
    model: str,
    sandbox: str | None = None,
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
    # gemini_cli gets double the time: its healthy multi-turn runs (multi_call,
    # skills) already take ~300s, so any burst of Gemini API retries pushes it
    # over the limit and fails the run with a truncated transcript rather than
    # an obvious timeout (e.g. meridianlabs-ai/actions run 33373083766, where 8
    # HTTP retries left only 1 of the expected 4+ user messages). 600s still
    # sits well under the 900s pytest --timeout backstop in the nightly.
    return eval(
        example_file,
        model=model,
        limit=1,
        task_args=task_args,
        time_limit=600 if agent == "gemini_cli" else 300,
        token_limit=500_000,
    )


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
