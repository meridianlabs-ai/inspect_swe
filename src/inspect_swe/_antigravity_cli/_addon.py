"""mitmproxy addon: route agy's Vertex :streamGenerateContent to the Inspect
bridge model_proxy (dynamic port). Everything else passes through to Google.

Uses ``requests`` (bundled with the mitmproxy standalone binary) run off the
event loop via ``run_in_executor``, not ``httpx``: the standalone binary's
embedded Python environment does not ship httpx (or anyio), only requests.
"""

import asyncio  # noqa: ANYIO_OK -- anyio is not present in mitmdump's embedded env
import functools
import json
import os
import re
from typing import TypeAlias

import requests
from mitmproxy import http

BRIDGE_PORT = int(os.environ.get("ANTIGRAVITY_BRIDGE_PORT", "13131"))
_MODEL_RE = re.compile(r"models/([^/:]+)")

# Vertex/Gemini's Schema proto uses SCREAMING_SNAKE_CASE type names for the
# JSON Schema "type" keyword in tool declarations; the bridge's ToolParams
# model expects lowercase JSON Schema type strings.
_SCHEMA_TYPE_NAMES = frozenset(
    {"STRING", "NUMBER", "INTEGER", "BOOLEAN", "ARRAY", "OBJECT", "NULL"}
)

JsonValue: TypeAlias = (
    "dict[str, JsonValue] | list[JsonValue] | str | int | float | bool | None"
)


def _lowercase_schema_type(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return value.lower() if value in _SCHEMA_TYPE_NAMES else value
    if isinstance(value, list):
        return [_lowercase_schema_type(item) for item in value]
    return value


def _lowercase_schema_types(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: _lowercase_schema_type(inner)
            if key == "type"
            else _lowercase_schema_types(inner)
            for key, inner in value.items()
        }
    if isinstance(value, list):
        return [_lowercase_schema_types(item) for item in value]
    return value


def _lowercase_tool_schema_types(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return body

    normalized_tools: list[JsonValue] = []
    for tool in tools:
        if not isinstance(tool, dict):
            normalized_tools.append(tool)
            continue
        declarations = tool.get("functionDeclarations")
        if not isinstance(declarations, list):
            normalized_tools.append(tool)
            continue

        normalized_declarations: list[JsonValue] = []
        for declaration in declarations:
            if not isinstance(declaration, dict):
                normalized_declarations.append(declaration)
                continue
            normalized_declaration = declaration.copy()
            for schema_key in ("parameters", "parametersJsonSchema"):
                schema = declaration.get(schema_key)
                if isinstance(schema, (dict, list)):
                    normalized_declaration[schema_key] = _lowercase_schema_types(schema)
            normalized_declarations.append(normalized_declaration)

        normalized_tool = tool.copy()
        normalized_tool["functionDeclarations"] = normalized_declarations
        normalized_tools.append(normalized_tool)

    if normalized_tools == tools:
        return body
    normalized_body = body.copy()
    normalized_body["tools"] = normalized_tools
    return normalized_body

# The Antigravity CLI sends functionResponse parts (tool results) inside a
# content item with role="model" instead of the standard role="user". The
# Inspect bridge's messages_from_google_contents only recognizes
# functionResponse parts when role=="user"; under role=="model" they are
# silently dropped, producing a blank assistant turn immediately after every
# tool call and stalling the agent. Relabel (or split out) those parts to
# role="user" before forwarding to the bridge.
def _relabel_function_response_contents(
    contents: list[JsonValue],
) -> list[JsonValue]:
    normalized: list[JsonValue] = []
    changed = False

    for content in contents:
        if not isinstance(content, dict) or content.get("role") != "model":
            normalized.append(content)
            continue

        parts = content.get("parts")
        if not isinstance(parts, list):
            normalized.append(content)
            continue

        function_response_parts: list[JsonValue] = [
            part
            for part in parts
            if isinstance(part, dict)
            and ("functionResponse" in part or "function_response" in part)
        ]
        if not function_response_parts:
            normalized.append(content)
            continue

        changed = True
        remaining_parts: list[JsonValue] = [
            part for part in parts if part not in function_response_parts
        ]

        if remaining_parts:
            model_content = content.copy()
            model_content["parts"] = remaining_parts
            normalized.append(model_content)

        user_content = content.copy()
        user_content["role"] = "user"
        user_content["parts"] = function_response_parts
        normalized.append(user_content)

    return normalized if changed else contents


def _relabel_function_response_role(
    body: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    contents = body.get("contents")
    if not isinstance(contents, list):
        return body
    normalized_contents = _relabel_function_response_contents(contents)
    if normalized_contents is contents:
        return body
    normalized_body = body.copy()
    normalized_body["contents"] = normalized_contents
    return normalized_body

def _normalized_forward_body(content: bytes) -> bytes:
    try:
        body: JsonValue = json.loads(content)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return content
    if not isinstance(body, dict):
        return content
    normalized_body = _lowercase_tool_schema_types(body)
    normalized_body = _relabel_function_response_role(normalized_body)
    return content if normalized_body == body else json.dumps(normalized_body).encode()


async def request(flow: http.HTTPFlow) -> None:
    host, path = flow.request.pretty_host, flow.request.path
    if "aiplatform" not in host or "generatecontent" not in path.lower():
        return
    match = _MODEL_RE.search(path)
    model = match.group(1) if match else "inspect"
    url = (
        f"http://127.0.0.1:{BRIDGE_PORT}/v1beta/models/{model}:"
        "streamGenerateContent?alt=sse"
    )
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        functools.partial(
            requests.post,
            url,
            data=_normalized_forward_body(flow.request.content),
            headers={"content-type": "application/json"},
            timeout=600,
        ),
    )
    flow.response = http.Response.make(
        response.status_code,
        response.content,
        {
            "content-type": response.headers.get(
                "content-type", "text/event-stream; charset=utf-8"
            )
        },
    )
