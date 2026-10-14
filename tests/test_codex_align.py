"""End-to-end tests that Codex's prompt/tooling is aligned to the real model.

These run Codex CLI in a real sandbox and capture its first bridged request via
a ``GenerateFilter``. The bridge hands the filter Codex's chosen instructions as
the leading system message and the tool set Codex offered, so we can assert that
the catalog-driven mapping took effect:

- an OpenAI model in Codex's catalog maps to its native entry -> ``apply_patch``
  and ``tool_search`` are offered;
- an OpenAI model *not* in the catalog and not a frontier version (e.g. an older
  ``gpt-5``) passes through to Codex's generic prompt -> no ``apply_patch`` and,
  importantly, no ``tool_search`` (which such models reject);
- a non-OpenAI model falls back to Codex's generic prompt -> no ``apply_patch``;
- with a custom ``model_catalog_json``, the slug comes from that catalog.

Slow: requires Docker + a live model API (mirrors ``tests/test_mcp.py``). The
custom catalog tests answer every request in the filter, so they need Docker
but no API key.
"""

import copy
import json
from pathlib import Path
from typing import Any

import anyio
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    GenerateConfig,
    Model,
    ModelOutput,
)
from inspect_ai.tool import ToolChoice, ToolInfo
from inspect_swe import codex_cli
from inspect_swe._codex_cli.agentbinary import codex_version_models_catalog

from tests.conftest import (
    skip_if_no_anthropic,
    skip_if_no_docker,
    skip_if_no_openai,
)

# Dockerfile with the prerequisites for running an agent in a sandbox.
_DOCKERFILE = str(Path(__file__).parent.parent / "examples" / "mcp" / "Dockerfile")


class _CaptureFirstRequest:
    """A bridge ``GenerateFilter`` that records the first request and passes through."""

    def __init__(self, respond: bool = False) -> None:
        self.respond = respond
        self.system_prompt: str | None = None
        self.tool_names: list[str] | None = None

    async def __call__(
        self,
        model: Model,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice | None,
        config: GenerateConfig,
    ) -> ModelOutput | None:
        if self.tool_names is None:
            self.tool_names = [t.name for t in tools]
            self.system_prompt = next(
                (m.text for m in messages if isinstance(m, ChatMessageSystem)), None
            )
        if self.respond:
            return ModelOutput.from_content(str(model), "DONE")
        # passthrough: let the real model handle generation
        return None


def _run_codex(
    model: str, custom_catalog: dict[str, Any] | None = None
) -> _CaptureFirstRequest:
    capture = _CaptureFirstRequest(respond=custom_catalog is not None)
    files = config_overrides = None
    version = "auto"
    if custom_catalog is not None:
        files = {_CATALOG_FILE: json.dumps(custom_catalog)}
        config_overrides = {"model_catalog_json": f'"{_CATALOG_FILE}"'}
        version = _CODEX_VERSION
    task = Task(
        dataset=[Sample(input="Print the word DONE and then stop.", files=files)],
        solver=codex_cli(
            model=model,
            version=version,
            config_overrides=config_overrides,
            filter=capture,
        ),
        sandbox=("docker", _DOCKERFILE),
    )
    eval(task, model=model, limit=1)
    assert capture.tool_names is not None, "Codex made no bridged request"
    return capture


def _offers_apply_patch(tool_names: list[str]) -> bool:
    return any("apply_patch" in name for name in tool_names)


def _offers_tool_search(tool_names: list[str]) -> bool:
    return any("tool_search" in name for name in tool_names)


@skip_if_no_openai
@skip_if_no_docker
def test_codex_align_catalog_openai_gets_native_tools() -> None:
    # gpt-5.5 is in Codex's catalog -> native prompt + tools (apply_patch).
    capture = _run_codex("openai/gpt-5.5")
    assert capture.system_prompt, "expected a non-empty system prompt"
    assert _offers_apply_patch(capture.tool_names or []), (
        f"expected apply_patch for a catalog OpenAI model, got: {capture.tool_names}"
    )


@skip_if_no_openai
@skip_if_no_docker
def test_codex_align_older_openai_passes_through() -> None:
    # gpt-5 is recognized by Inspect but no longer in Codex's catalog and not a
    # frontier version -> generic fallback: no apply_patch and, critically, no
    # tool_search (which the real gpt-5 rejects with a 400). Guards the fix that
    # stopped older models from being aliased up to the latest catalog slug.
    capture = _run_codex("openai/gpt-5")
    assert capture.system_prompt, "expected a non-empty system prompt"
    assert not _offers_tool_search(capture.tool_names or []), (
        f"gpt-5 must not be offered tool_search, got: {capture.tool_names}"
    )
    assert not _offers_apply_patch(capture.tool_names or []), (
        f"expected generic fallback (no apply_patch) for gpt-5, got: {capture.tool_names}"
    )


@skip_if_no_anthropic
@skip_if_no_docker
def test_codex_align_non_openai_uses_generic_fallback() -> None:
    capture = _run_codex("anthropic/claude-sonnet-4-5")
    assert capture.system_prompt, "expected a non-empty system prompt"
    assert not _offers_apply_patch(capture.tool_names or []), (
        f"expected no apply_patch for a non-OpenAI model, got tools: {capture.tool_names}"
    )


# Codex release whose catalog the custom catalog is built from
_CODEX_VERSION = "0.160.1"
_CATALOG_FILE = "/tmp/codex-models.json"
_CUSTOM_MARKER = "CUSTOM-CATALOG-GPT-9.9"


def _custom_catalog() -> dict[str, Any]:
    """A catalog whose only entry is `gpt-9.9`, with marked instructions.

    The entry is a copy of the release's latest one. The release catalog has
    no `gpt-9.9`, and Codex's generic fallback has no marker.
    """
    release = anyio.run(codex_version_models_catalog, _CODEX_VERSION)
    if release is None:
        pytest.skip(f"Codex {_CODEX_VERSION} catalog unavailable")
    entry = copy.deepcopy(
        next(m for m in release["models"] if m["slug"] == "gpt-6.1-sol")
    )
    # direct, so its tools (apply_patch) are offered to the model itself
    entry.update(slug="gpt-9.9", tool_mode="direct")
    messages = entry["model_messages"]
    messages["instructions_template"] = (
        f"{_CUSTOM_MARKER}\n{messages['instructions_template']}"
    )
    return {**release, "models": [entry]}


@pytest.mark.slow
@skip_if_no_docker
@pytest.mark.parametrize(
    "model",
    [
        # only the custom catalog has it: its own entry
        "openai/gpt-9.9",
        # only the release catalog has it: aliased to the custom catalog's latest
        "openai/gpt-5.5",
    ],
)
def test_codex_align_custom_catalog(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    capture = _run_codex(model, _custom_catalog())
    assert capture.system_prompt and _CUSTOM_MARKER in capture.system_prompt, (
        f"expected the custom catalog entry's instructions, got: {capture.system_prompt}"
    )
    assert _offers_apply_patch(capture.tool_names or []), (
        f"expected the custom entry's apply_patch, got: {capture.tool_names}"
    )
