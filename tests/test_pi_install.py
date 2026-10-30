import json
from pathlib import Path

import anyio
import pytest
from inspect_ai._eval.task.sandbox import sandboxenv_context
from inspect_ai._util.logger import init_logger
from inspect_ai.dataset import Sample
from inspect_ai.util import SandboxEnvironmentSpec, sandbox
from inspect_swe._pi.agentbinary import pi_binary_source
from inspect_swe._pi.config import pi_models_json, pi_result_error
from inspect_swe._util.agentbinary import ensure_agent_binary_installed

from tests.conftest import skip_if_no_docker


@pytest.mark.slow
@skip_if_no_docker
def test_pi_offline_sandbox_install_and_cli(tmp_path: Path) -> None:
    """Run the real standalone release in a network-isolated, non-root sandbox."""
    init_logger(log_level="warning")
    compose = tmp_path / "compose.yaml"
    compose.write_text(
        data="""services:
  default:
    image: python:3.12-slim
    command: sleep infinity
    network_mode: none
    working_dir: /tmp
"""
    )

    async def check() -> None:
        async with sandboxenv_context(
            task_name="pi-install-test",
            sandbox=SandboxEnvironmentSpec(type="docker", config=str(compose)),
            max_sandboxes=None,
            cleanup=True,
            sample=Sample(input="Install Pi"),
        ):
            sbox = sandbox()
            binary = await ensure_agent_binary_installed(
                source=pi_binary_source(), version="latest", sandbox=sbox
            )
            result = await sbox.exec(cmd=[binary, "--version"])
            assert result.success, result.stderr
            version = result.stdout.strip()
            assert version
            cached = await ensure_agent_binary_installed(
                source=pi_binary_source(), version=version, sandbox=sbox
            )
            assert cached == binary
            for mode in ("auto", "sandbox"):
                linked = await sbox.exec(cmd=["ln", "-sf", binary, "/usr/local/bin/pi"])
                assert linked.success, linked.stderr
                assert (
                    await ensure_agent_binary_installed(
                        source=pi_binary_source(),
                        version=mode,
                        user="nobody",
                        sandbox=sbox,
                    )
                    == "/usr/local/bin/pi"
                )
            setup = await sbox.exec(
                cmd=["mktemp", "-d", "/tmp/inspect-pi-XXXXXXXX"], user="nobody"
            )
            assert setup.success, setup.stderr
            pi_dir = setup.stdout.strip()
            await sbox.write_file(
                file=f"{pi_dir}/models.json",
                contents=pi_models_json(
                    port=1,
                    model="openai/gpt-5-mini",
                    context_size=400000,
                    max_tokens=4096,
                    reasoning=False,
                ),
            )
            await sbox.write_file(
                file=f"{pi_dir}/settings.json",
                contents=json.dumps({"retry": {"enabled": False}}),
            )
            env = {
                "PI_CODING_AGENT_DIR": pi_dir,
                "PI_OFFLINE": "1",
                "PI_TELEMETRY": "0",
            }
            result = await sbox.exec(
                cmd=[binary, "--list-models", "inspect"],
                env=env,
                user="nobody",
                cwd=pi_dir,
                timeout=30,
            )
            assert result.success, result.stderr
            assert "openai/gpt-5-mini" in result.stdout
            prompt = "@nonexistent-pi-file\nSay hello"
            await sbox.write_file(file=f"{pi_dir}/prompt.txt", contents=prompt)
            result = await sbox.exec(
                cmd=[
                    "bash",
                    "-c",
                    'prompt=$1; shift; exec "$@" < "$prompt"',
                    "bash",
                    f"{pi_dir}/prompt.txt",
                    binary,
                    "--append-system-prompt",
                    "pi-test-system-marker",
                    "--provider",
                    "inspect",
                    "--model",
                    "openai/gpt-5-mini",
                    "--thinking",
                    "off",
                    "--session",
                    f"{pi_dir}/session.jsonl",
                    "--no-extensions",
                    "--no-skills",
                    "--no-prompt-templates",
                    "--no-themes",
                    "--no-approve",
                    "--print",
                    "--mode",
                    "json",
                ],
                env=env,
                user="nobody",
                cwd=pi_dir,
                timeout=60,
            )
            # No server is listening on port 1. This checks actual Pi failure events,
            # not a fabricated CLI transcript or a replacement model implementation.
            assert pi_result_error(stdout=result.stdout) is not None
            assert '"stopReason":"error"' in result.stdout, (
                result.stdout + result.stderr
            )
            session_text = await sbox.read_file(file=f"{pi_dir}/session.jsonl")
            entries = [json.loads(line) for line in session_text.split("\n") if line]
            messages = [
                entry["message"] for entry in entries if entry["type"] == "message"
            ]
            assert messages[0]["role"] == "user"
            assert messages[0]["content"][0]["text"] == prompt
            # Pi stores conversation messages, not the system prompt. The wrapper
            # must supply its original append-system-prompt again when resuming.
            assert "pi-test-system-marker" not in session_text

    anyio.run(check)
