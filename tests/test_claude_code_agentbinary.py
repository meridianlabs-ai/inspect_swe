from unittest.mock import AsyncMock

import pytest
from inspect_swe._claude_code import agentbinary


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("assignment", "expected_url"),
    [
        (
            "export DOWNLOAD_BASE_URL='https://downloads.claude.ai/claude-code-releases'",
            "https://downloads.claude.ai/claude-code-releases",
        ),
        (
            'GCS_BUCKET="https://storage.googleapis.com/legacy-claude-code"',
            "https://storage.googleapis.com/legacy-claude-code",
        ),
    ],
)
async def test_download_base_url_supports_current_and_legacy_install_scripts(
    monkeypatch: pytest.MonkeyPatch, assignment: str, expected_url: str
) -> None:
    # Given
    manifest = (
        '{"version":"2.1.205","platforms":{"linux-x64":'
        '{"checksum":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        '"size":1}}}'
    )
    monkeypatch.setattr(
        agentbinary,
        "download_text_file",
        AsyncMock(side_effect=[f"#!/bin/bash\n{assignment}\n", "2.1.205", manifest]),
    )

    # When
    resolved = await agentbinary.claude_code_binary_source().resolve_version(
        "stable", "linux-x64"
    )

    # Then
    assert resolved.version == "2.1.205"
    assert resolved.expected_checksum == "a" * 64
    assert resolved.download_url == f"{expected_url}/2.1.205/linux-x64/claude"
