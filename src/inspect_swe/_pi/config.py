import json
from datetime import datetime, timezone
from uuid import uuid4


def pi_command(*, binary: str, directory: str, model: str, thinking: str) -> list[str]:
    return [
        binary,
        "--provider",
        "inspect",
        "--model",
        model,
        "--thinking",
        thinking,
        "--session",
        f"{directory}/session.jsonl",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-themes",
        "--no-approve",
    ]


def pi_environment(*, directory: str, env: dict[str, str] | None) -> dict[str, str]:
    return (env or {}) | {
        "PI_CODING_AGENT_DIR": directory,
        "PI_OFFLINE": "1",
        "PI_TELEMETRY": "0",
        # Bun otherwise drops idle requests after 300s, even when Pi's own
        # httpIdleTimeoutMs is disabled. Inspect enforces model call timeouts.
        "BUN_CONFIG_HTTP_IDLE_TIMEOUT": "86400",
    }


def pi_session_header(*, cwd: str) -> str:
    return (
        json.dumps(
            {
                "type": "session",
                "version": 3,
                "id": str(uuid4()),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cwd": cwd,
            }
        )
        + "\n"
    )


def pi_models_json(
    *, port: int, model: str, context_size: int, max_tokens: int, reasoning: bool
) -> str:
    """Configure only a dummy-key provider pointing at the Inspect bridge."""
    return json.dumps(
        {
            "providers": {
                "inspect": {
                    "baseUrl": f"http://localhost:{port}/v1",
                    "api": "openai-completions",
                    "apiKey": "inspect-bridge",
                    "compat": {"supportsStore": False, "supportsDeveloperRole": False},
                    "models": [
                        {
                            "id": model,
                            "name": model,
                            "reasoning": reasoning,
                            "thinkingLevelMap": {"xhigh": "xhigh", "max": "max"},
                            "input": ["text", "image"],
                            "contextWindow": context_size,
                            "maxTokens": max_tokens,
                        }
                    ],
                }
            }
        }
    )


def pi_result_error(*, stdout: str) -> str | None:
    """Check Pi's final turn; an exit code alone can miss model errors."""
    final_message = None
    ended = False
    for line in stdout.split("\n"):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            # Startup diagnostics may share stdout with JSON events. Require a
            # complete terminal event below rather than treating them as success.
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "agent_start":
            ended = False
            final_message = None
        if event.get("type") == "agent_end":
            ended = True
        if event.get("type") == "message_end":
            message = event.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                final_message = message
    if final_message is not None and final_message.get("stopReason") in (
        "error",
        "aborted",
    ):
        return str(final_message.get("errorMessage") or final_message["stopReason"])
    if (
        not ended
        or final_message is None
        or final_message.get("stopReason") not in ("stop", "length")
    ):
        return "Pi exited without a completed assistant turn"
    return None
