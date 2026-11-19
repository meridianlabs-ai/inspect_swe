import importlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
from inspect_ai.agent import AgentState
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.model._model import init_active_model
from inspect_ai.tool import Skill
from inspect_ai.util import SandboxEnvironment, Store
from inspect_ai.util._sandbox import ExecRemoteAwaitableOptions
from inspect_ai.util._store import init_subtask_store

pi_module = importlib.import_module("inspect_swe._pi.pi")
DIRECTORY = "/tmp/inspect-pi-test"
PROMPT = '@literal "quoted"\n\nDo not expand $HOME or `echo hi`.'


class Result:
    success = True
    returncode = 0
    stderr = ""

    def __init__(self, stdout: str = "") -> None:
        self.stdout = stdout


class RecordingSandbox(SandboxEnvironment):
    def __init__(self) -> None:
        self.files: dict[str, str | bytes] = {}
        self.writes: list[tuple[str, str | bytes]] = []
        self.commands: list[tuple[list[str], dict[str, Any]]] = []
        self.remote: list[
            tuple[list[str], ExecRemoteAwaitableOptions, str | bytes]
        ] = []

    async def exec(self, cmd: list[str], *args: Any, **kwargs: Any) -> Any:
        self.commands.append((cmd, kwargs))
        if cmd[0] == "mktemp":
            return Result(f"{DIRECTORY}\n")
        if "pwd" in cmd[-1]:
            return Result("/workspace/project\n")
        return Result()

    async def write_file(self, file: str, contents: str | bytes, **kwargs: Any) -> None:
        self.files[file] = contents
        self.writes.append((file, contents))

    async def read_file(self, file: str, *args: Any, **kwargs: Any) -> Any:
        return self.files[file]

    async def exec_remote(
        self, cmd: list[str], options: Any = None, **kwargs: Any
    ) -> Any:
        assert isinstance(options, ExecRemoteAwaitableOptions)
        assert kwargs == {"stream": False}
        assert all(
            f"{DIRECTORY}/{name}" in self.files
            for name in ("models.json", "settings.json", "session.jsonl", "system.txt")
        )
        self.remote.append((cmd, options, self.files[cmd[4]]))
        return Result(
            '{"type":"message_end","message":{"role":"assistant","stopReason":"stop"}}\n{"type":"agent_end"}\n'
        )

    @classmethod
    async def sample_cleanup(cls, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.anyio
@pytest.mark.parametrize("optional", [False, True])
async def test_pi_executes_and_resumes(
    monkeypatch: pytest.MonkeyPatch, optional: bool
) -> None:
    sbox = RecordingSandbox()
    bridge_calls: list[dict[str, Any]] = []
    installs: list[dict[str, Any]] = []
    sandbox_names: list[str | None] = []

    def sandbox_env(name: str | None = None) -> RecordingSandbox:
        sandbox_names.append(name)
        return sbox

    @asynccontextmanager
    async def bridge(state: AgentState, **kwargs: Any) -> AsyncIterator[Any]:
        bridge_calls.append(kwargs)
        yield type("Bridge", (), {"port": 4321, "state": state})()

    async def install(**kwargs: Any) -> str:
        installs.append(kwargs)
        return "/opt/pi"

    monkeypatch.setattr(pi_module, "sandbox_env", sandbox_env)
    monkeypatch.setattr(pi_module, "sandbox_agent_bridge", bridge)
    monkeypatch.setattr(pi_module, "ensure_agent_binary_installed", install)

    model = get_model(
        model="openai/gpt-5-mini",
        api_key="dummy",
        base_url="http://localhost:1/v1",
        config=GenerateConfig(max_tokens=1234),
    )
    init_active_model(model, GenerateConfig())
    init_subtask_store(Store())
    kwargs: dict[str, Any] = {}
    if optional:
        kwargs = {
            "model": "fast",
            "model_aliases": {"fast": model},
            "max_context_size": 8000,
            "thinking": "high",
            "cwd": "project",
            "user": "worker",
            "sandbox": "named",
            "version": "1.2.3",
            "retry_refusals": 2,
            "env": {
                "CUSTOM": "value",
                "PI_OFFLINE": "0",
                "PI_CODING_AGENT_DIR": "bad",
                "PI_TELEMETRY": "1",
                "BUN_CONFIG_HTTP_IDLE_TIMEOUT": "1",
            },
            "system_prompt": "Extra instruction.",
            "agents_md": "Project rules.",
            "skills": [
                Skill(
                    name="example",
                    description="Example skill",
                    instructions="Do the thing.",
                )
            ],
        }
    agent = pi_module.pi(**kwargs)
    state = AgentState(
        messages=[
            ChatMessageSystem(content="Original instruction."),
            ChatMessageUser(content=PROMPT),
        ]
    )
    assert await agent(state) is state
    state.messages.extend(
        [
            ChatMessageAssistant(content="First answer"),
            ChatMessageSystem(content="Changed instruction."),
            ChatMessageUser(content="Next @literal\n'quoted'"),
        ]
    )
    assert await agent(state) is state

    assert sandbox_names == (["named", "named"] if optional else [None, None])
    assert len(installs) == 2
    assert all(
        item["sandbox"] is sbox
        and item["user"] == ("worker" if optional else None)
        and item["version"] == ("1.2.3" if optional else "auto")
        and item["source"].binary == item["source"].agent == "pi"
        for item in installs
    )
    assert [call["port"] for call in bridge_calls] == [3201, 3202]
    assert all(
        call["sandbox"] == ("named" if optional else None)
        and call["forward_generation_config"] is True
        and call["retry_refusals"] == (2 if optional else None)
        and callable(call["filter"])
        for call in bridge_calls
    )
    assert all(
        call["model"] == ("inspect/fast" if optional else "inspect")
        for call in bridge_calls
    )
    assert all(call["model_aliases"][model.name] is model for call in bridge_calls)
    if optional:
        assert all(call["model_aliases"]["fast"] is model for call in bridge_calls)

    provider = json.loads(sbox.files[f"{DIRECTORY}/models.json"])["providers"][
        "inspect"
    ]
    assert provider["baseUrl"] == "http://localhost:4321/v1"
    assert provider["apiKey"] == "inspect-bridge"
    model_config = provider["models"][0]
    assert model_config["id"] == model_config["name"] == model.name
    assert model_config["maxTokens"] == 1234
    assert model_config["reasoning"] is optional
    if optional:
        assert model_config["contextWindow"] == 8000
    else:
        assert model_config["contextWindow"] > 8000
    assert json.loads(sbox.files[f"{DIRECTORY}/settings.json"]) == {
        "retry": {"enabled": False},
        "httpIdleTimeoutMs": 0,
        "compaction": {"reserveTokens": min(16384, model_config["contextWindow"] // 4)},
    }
    assert (
        json.loads(sbox.files[f"{DIRECTORY}/session.jsonl"])["cwd"]
        == "/workspace/project"
    )
    assert sum(path.endswith("/session.jsonl") for path, _ in sbox.writes) == 1
    instruction = (
        "Original instruction.\n\nExtra instruction."
        if optional
        else "Original instruction."
    )
    assert [
        contents for path, contents in sbox.writes if path == f"{DIRECTORY}/system.txt"
    ] == [instruction, instruction]
    pwd_calls = [(cmd, opts) for cmd, opts in sbox.commands if "pwd" in cmd[-1]]
    assert len(pwd_calls) == 2
    assert all(
        opts["user"] == ("worker" if optional else None)
        and opts["cwd"] == ("project" if optional else None)
        for _, opts in pwd_calls
    )
    assert sum(cmd[0] == "mktemp" for cmd, _ in sbox.commands) == 1
    if optional:
        assert sbox.files[f"{DIRECTORY}/AGENTS.md"] == "Project rules."
        skill = sbox.files[f"{DIRECTORY}/skills/example/SKILL.md"]
        assert (
            isinstance(skill, str)
            and "Example skill" in skill
            and "Do the thing." in skill
        )
    else:
        assert f"{DIRECTORY}/AGENTS.md" not in sbox.files

    assert len(sbox.remote) == 2
    for (cmd, options, prompt), expected in zip(
        sbox.remote, [PROMPT, "Next @literal\n'quoted'"], strict=True
    ):
        assert cmd[:5] == [
            "bash",
            "-c",
            'prompt=$1; shift; exec "$@" < "$prompt"',
            "bash",
            f"{DIRECTORY}/prompt.txt",
        ]
        assert prompt == expected
        assert all(expected not in arg for arg in cmd)
        assert cmd[5:11] == [
            "/opt/pi",
            "--provider",
            "inspect",
            "--model",
            model.name,
            "--thinking",
        ]
        assert cmd[11] == ("high" if optional else "off")
        assert cmd[cmd.index("--session") + 1] == f"{DIRECTORY}/session.jsonl"
        assert cmd[cmd.index("--append-system-prompt") + 1] == f"{DIRECTORY}/system.txt"
        assert ("--skill" in cmd) is optional
        if optional:
            assert cmd[cmd.index("--skill") + 1] == f"{DIRECTORY}/skills"
        assert cmd[-3:] == ["--print", "--mode", "json"]
        assert {
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-themes",
            "--no-approve",
        }.issubset(cmd)
        assert options.cwd == "/workspace/project" and options.user == (
            "worker" if optional else None
        )
        assert options.concurrency is False
        assert options.env == {
            "PI_CODING_AGENT_DIR": DIRECTORY,
            "PI_OFFLINE": "1",
            "PI_TELEMETRY": "0",
            "BUN_CONFIG_HTTP_IDLE_TIMEOUT": "86400",
        } | ({"CUSTOM": "value"} if optional else {})
