import importlib
import os
import subprocess
from typing import Any, Callable, List, Literal, TypeVar, cast

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
    agent: Literal["claude_code", "codex_cli", "mini_swe_agent"],
    model: str,
    sandbox: str | None = None,
) -> list[EvalLog]:
    example_file = os.path.join("examples", example)
    task_args: dict[str, str] = {
        "agent": agent,
    }

    if sandbox is not None:
        task_args["sandbox"] = sandbox
    return eval(example_file, model=model, limit=1, task_args=task_args)
