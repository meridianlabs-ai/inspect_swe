"""Drift check: the dsh strings `deepseek_harness()` classifies and places by.

`_deepseek_harness/_events/detection.py` reads who sent a request, which
children settled and what is a compaction call from text dsh's own runtime
writes. `"auto"` and `"latest"` install whatever dsh release is newest, so a
reworded string would silently turn sub-agent requests into `"unknown"` ones
(or compaction calls into root turns). This fetches the sources that write
those strings from dsh's default branch and fails loudly instead.

Skips when offline; github-action-gated so an upstream hiccup can't block
unrelated CI.
"""

import anyio
import pytest
from inspect_swe._deepseek_harness._events.detection import (
    COMPACTION_INSTRUCTION_MARKER,
    COMPACTION_INSTRUCTION_PREAMBLE,
    DELEGATION_TOOLS,
    ROOT_AGENT_NAME,
    WEB_SEARCH_REQUEST_PREFIX,
)
from inspect_swe._deepseek_harness.deepseek_harness import (
    DSH_SEARCH_MODEL,
    DSH_SEARCH_PROVIDER,
)

from tests.conftest import skip_if_github_action

_RAW = "https://raw.githubusercontent.com/deepseek-ai/deepseek-harness/master/"

# source file → the text detection.py depends on, as that file writes it
_EXPECTED: dict[str, list[str]] = {
    # the session reminder every agent's identity is read from
    "packages/core/agent-loop/src/runtime-context.ts": [
        "'The following information describes your current session:'",
        "'The following information describes your current session. It replaces",
        "`<system-reminder>\\n${intro}\\n\\n${body}\\n</system-reminder>`",
    ],
    "packages/subagent/tool-subagent-control/src/index.ts": [
        "`You are the root agent. Your name is ${JSON.stringify(names.self)}.`",
        "`You are a subagent delegated by agent ${JSON.stringify(names.parent)}.\\n`",
        "`Your name is ${JSON.stringify(names.self)}.\\n`",
    ],
    "packages/subagent/subagent/src/names.ts": [
        f"ROOT_AGENT_NAME = '{ROOT_AGENT_NAME}'",
    ],
    # a delegation call's launch acknowledgement
    "packages/subagent/tool-subagent/src/index.ts": [
        "`started subagent ${value.name}`",
    ],
    # the settlement notice that ends a child's span, in every ending
    "packages/subagent/subagent/src/continuation-messages.ts": [
        "`Background subagent ${childName}`",
        "`${subject} finished and will do no further work unless you send it more.`",
        "`${subject} finished. It cannot receive follow-up messages.`",
        "`${subject} was stopped before it finished.`",
        "`${subject} ran out of room before it finished.`",
        "`${subject} declined the task.`",
        "`${subject} failed before it finished.`",
        "`${subject} ended abnormally (${String(stopReason)}) before it finished.`",
    ],
    "packages/compaction/compaction-basic/src/summarizer.ts": [
        COMPACTION_INSTRUCTION_PREAMBLE,
        COMPACTION_INSTRUCTION_MARKER,
    ],
    # the model request behind web_search, and the provider the agent pins
    "packages/web/web-search-deepseek/src/provider.ts": [
        "`" + WEB_SEARCH_REQUEST_PREFIX + "${request.query}`",
        f"DEEPSEEK_PROVIDER_ID = '{DSH_SEARCH_PROVIDER}'",
        f"DEEPSEEK_DEFAULT_MODEL = '{DSH_SEARCH_MODEL}'",
        "type: 'web_search_20250305'",
    ],
    # the delegation tools the headless profile mounts, and the rows the
    # agent's profile patch addresses by id
    "packages/bundle/base/cordis.patch.yml": [
        *(f"toolName: {tool}" for tool in DELEGATION_TOOLS),
        *(
            f"    - id: {row}\n"
            for row in (
                "agent-default-model",
                "llm-deepseek",
                "web",
                "tool-goal",
                "goal-round-driver",
                "session-log-deepseek",
                "plugin-package-inventory-deepseek",
                "session-title-llm",
                "session-telemetry-otel",
            )
        ),
    ],
}


@skip_if_github_action
@pytest.mark.parametrize("path", sorted(_EXPECTED))
def test_live_dsh_source_still_writes_what_detection_reads(path: str) -> None:
    from inspect_swe._util.download import download_text_file

    try:
        source = anyio.run(download_text_file, _RAW + path)
    except Exception as ex:
        pytest.skip(f"live deepseek-harness source unavailable: {ex}")

    missing = [text for text in _EXPECTED[path] if text not in source]
    assert not missing, (
        f"dsh's {path} no longer contains text that "
        "_deepseek_harness/_events/detection.py matches requests by; re-derive "
        f"the patterns there from the current source. Missing: {missing}"
    )
