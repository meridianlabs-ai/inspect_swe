"""Bridge configuration for Oh My Pi 18.6.1.

These settings control OMP routing, not sandbox network access. The caller must
also isolate HOME and the process environment, use an explicit --config overlay,
and disable extensions, skills, rules, and titles on the command line.
"""

from typing import Any

# OMP's modelRoles values are selectors, not provider/model objects. Pin every
# chat-capable role, including the tiny-model helpers, to the same bridge.
_CHAT_ROLES = (
    "default",
    "smol",
    "slow",
    "vision",
    "plan",
    "commit",
    "tiny",
    "memory",
    "task",
    "advisor",
    "judge",
)
_NON_CHAT_ROLES = ("image", "web", "speech", "dictation")

# enabledProviders only opts foreign configuration sources into discovery; it
# is NOT an inference-provider allowlist. Disable the actual discovery sources.
_DISCOVERY_PROVIDERS = (
    "native",
    "agents",
    "agents-md",
    "agent-plugins",
    "builtin-defaults",
    "claude",
    "claude-md",
    "claude-plugins",
    "cline",
    "codex",
    "cursor",
    "gemini",
    "github",
    "mcp-json",
    "omp-managed",
    "omp-plugins",
    "opencode",
    "skillshare",
    "ssh-json",
    "vscode",
    "windsurf",
)

# Snapshot of the bundled provider catalog in 18.6.1, plus the implicit
# llama.cpp provider. Disabling only credentialed providers is insufficient:
# OMP also refreshes public provider catalogs and probes local model servers.
# Revisit this list when updating the pinned OMP version. This is not an egress
# boundary: tools and project dotenv files still run inside the sandbox.
_MODEL_PROVIDERS = (
    "abliteration",
    "aiand",
    "aimlapi",
    "alibaba-coding-plan",
    "alibaba-token-plan",
    "amazon-bedrock",
    "anthropic",
    "apple",
    "azure",
    "baseten",
    "bedrock-mantle",
    "cerebras",
    "charm-hyper",
    "cline-pass",
    "cloudflare-ai-gateway",
    "commandcode",
    "coreweave",
    "cursor",
    "deepinfra",
    "deepseek",
    "devin",
    "factory-droid",
    "firepass",
    "fireworks",
    "github-copilot",
    "gitlab-duo",
    "gitlab-duo-agent",
    "gmi-cloud",
    "google",
    "google-antigravity",
    "google-gemini-cli",
    "google-vertex",
    "groq",
    "helmcode",
    "huggingface",
    "kilo",
    "kimi-code",
    "litellm",
    "llama.cpp",
    "lm-studio",
    "local",
    "meta",
    "minimax",
    "minimax-code",
    "minimax-code-cn",
    "mistral",
    "moonshot",
    "muse-code",
    "nanogpt",
    "novita",
    "nvidia",
    "ollama",
    "ollama-cloud",
    "openai",
    "openai-codex",
    "opencode-go",
    "opencode-zen",
    "openrouter",
    "qianfan",
    "qwen-portal",
    "sakana",
    "siliconflow",
    "siliconflow-cn",
    "singularityapi-dev",
    "singularityapi-tech",
    "stepfun",
    "synthetic",
    "together",
    "typesafe",
    "umans",
    "venice",
    "vercel-ai-gateway",
    "vllm",
    "wafer-serverless",
    "web",
    "xai",
    "xai-oauth",
    "xiaomi",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-sgp",
    "yolo-auto",
    "zai",
    "zenmux",
    "zhipu-coding-plan",
)


def omp_models_config(
    port: int, context_window: int, max_tokens: int
) -> dict[str, Any]:
    """Return models.yml content for the sole configured bridge model.

    Capacity comes from the caller, not OMP's defaults for unknown model IDs.
    The bridge owns provider credentials and reasoning policy; OMP receives only
    a dummy key and uses ordinary text/tool OpenAI Chat Completions requests.
    """
    return {
        "providers": {
            "inspect": {
                "baseUrl": f"http://localhost:{port}/v1",
                "api": "openai-completions",
                "apiKey": "sk-none",
                "auth": "apiKey",
                "models": [
                    {
                        "id": "inspect",
                        "name": "Inspect",
                        "reasoning": False,
                        "input": ["text"],
                        "contextWindow": context_window,
                        "maxTokens": max_tokens,
                        "compactionModel": "inspect/inspect",
                        "cost": {
                            "input": 0,
                            "output": 0,
                            "cacheRead": 0,
                            "cacheWrite": 0,
                        },
                        "compat": {
                            "supportsStore": False,
                            "supportsDeveloperRole": False,
                            "supportsReasoningEffort": False,
                            "maxTokensField": "max_tokens",
                        },
                    }
                ],
            }
        }
    }


def omp_settings() -> dict[str, Any]:
    """Return config.yml content, also required as a CLI --config overlay.

    An isolated global config alone does not override project settings. Native
    project model roles are loaded even when native discovery is disabled.
    Environment-backed settings can still override the overlay; this is not a
    security boundary against code or configuration in the evaluated sandbox.
    """
    return {
        "enabledModels": ["inspect/inspect"],
        "enabledProviders": [],
        "disabledProviders": list(
            dict.fromkeys((*_DISCOVERY_PROVIDERS, *_MODEL_PROVIDERS))
        ),
        "modelRoles": {
            **{role: "inspect/inspect" for role in _CHAT_ROLES},
            # Tombstones mask project assignments. Empty fallback chains below
            # suppress built-in priorities for these unsupported model kinds.
            **{role: None for role in _NON_CHAT_ROLES},
        },
        "retry": {
            "enabled": False,
            "modelFallback": False,
            "usageAwareFallback": False,
            "fallbackChains": {role: [] for role in (*_CHAT_ROLES, *_NON_CHAT_ROLES)},
        },
        "compaction": {
            "enabled": True,
            "asyncEnabled": False,
            "methodOrder": ["soft"],
        },
        "contextPromotion": {"enabled": False},
        "defaultThinkingLevel": "off",
        "externalThinking": False,
        "advisor": {"enabled": False},
        "autolearn": {"enabled": False, "autoContinue": False},
        "memory": {"backend": "off"},
        "prewalk": {"enabled": False},
        "images": {"describeForTextModels": False},
        "read": {"summarize": {"enabled": False}},
        "features": {"unexpectedStopDetection": "none"},
        "telemetry": {"otlpExportEnabled": False},
        "startup": {"checkUpdate": False},
        "marketplace": {"autoUpdate": "off"},
        "autoResume": False,
    }
