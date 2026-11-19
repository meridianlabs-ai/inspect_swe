from inspect_swe._pi.config import pi_command, pi_environment


def test_command_resumes_isolated_session_with_real_model() -> None:
    command = pi_command(
        binary="/opt/pi/pi",
        directory="/tmp/pi dir",
        model="gpt-5-mini",
        thinking="high",
    )
    assert command[:7] == [
        "/opt/pi/pi",
        "--provider",
        "inspect",
        "--model",
        "gpt-5-mini",
        "--thinking",
        "high",
    ]
    assert command[7:9] == ["--session", "/tmp/pi dir/session.jsonl"]
    assert command[9:] == [
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-approve",
    ]
    assert "--no-session" not in command


def test_environment_reserves_isolation_and_both_idle_timeout_controls() -> None:
    supplied = {
        "CUSTOM_VALUE": "kept",
        "PI_CODING_AGENT_DIR": "/unsafe",
        "PI_OFFLINE": "0",
        "PI_TELEMETRY": "1",
        "BUN_CONFIG_HTTP_IDLE_TIMEOUT": "300",
    }
    env = pi_environment(directory="/tmp/isolated-pi", env=supplied)
    assert env == {
        "CUSTOM_VALUE": "kept",
        "PI_CODING_AGENT_DIR": "/tmp/isolated-pi",
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        "BUN_CONFIG_HTTP_IDLE_TIMEOUT": "86400",
    }
    assert supplied["PI_CODING_AGENT_DIR"] == "/unsafe"
    assert "API_KEY" not in " ".join(pi_environment(directory="/tmp/pi", env=None))
