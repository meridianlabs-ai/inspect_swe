"""opencode receives its prompt on stdin, not as a positional argument.

``opencode run`` quote-wraps a positional message that contains spaces and
backslash-escapes the double quotes inside it (``packages/opencode/src/cli/cmd/
run.ts``), so the prompt the model saw -- and the user message crossing the
agent bridge -- differed from the task input. The bridge anchors main-thread
tracking on the task input, and for prompts containing ``"`` the mismatch let
opencode's session-title generation call be surfaced as the sample's final
answer (GAIA level 1, opencode 1.18.29 + gpt-5.5: 4 of 10 samples). Piped stdin
is used verbatim, so the prompt is delivered that way instead.
"""

import subprocess
from pathlib import Path

from inspect_swe._opencode.opencode import opencode_stdin_prompt_cmd

# embedded double quotes, blank lines, and no trailing newline
PROMPT = 'Write the opposite of the word "left".\n\nAnswer with one word'
OPENCODE_CMD = ["opencode", "run", "--model", "openai/gpt-5.5", "--format", "json"]


def test_prompt_is_not_a_positional_argument(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROMPT)

    cmd = opencode_stdin_prompt_cmd(OPENCODE_CMD, str(prompt_file))

    assert PROMPT not in cmd
    assert str(prompt_file) in cmd
    # opencode's own argv is preserved intact at the end (no message appended)
    assert cmd[-len(OPENCODE_CMD) :] == OPENCODE_CMD


def test_prompt_reaches_the_agent_verbatim_on_stdin(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(PROMPT)

    # stand in for opencode with a command that echoes its stdin
    cmd = opencode_stdin_prompt_cmd(["cat"], str(prompt_file))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert result.stdout == PROMPT


def test_stdin_redirection_tolerates_spaces_in_the_prompt_path(
    tmp_path: Path,
) -> None:
    prompt_file = tmp_path / "dir with spaces" / "prompt file.txt"
    prompt_file.parent.mkdir()
    prompt_file.write_text(PROMPT)

    cmd = opencode_stdin_prompt_cmd(["cat"], str(prompt_file))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    assert result.stdout == PROMPT
