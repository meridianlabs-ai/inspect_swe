# Security Policy

## Reporting a vulnerability

[Report a vulnerability](https://github.com/meridianlabs-ai/inspect_swe/security/advisories/new) through GitHub's private vulnerability reporting on this repository. Only you, the maintainers at Meridian Research Labs, and anyone they add to the report can see it. Please do not open a public issue, discussion, or pull request for a suspected vulnerability.

Include as much of the following as you can:

- The version of `inspect_swe` and `inspect_ai` you tested, your Python version and operating system
- A description of the problem and what an attacker could do with it
- Steps to reproduce it, ideally a small script or example
- Any logs, stack traces, or output that help show the behavior

## What to expect

- We will acknowledge your report within 3 business days.
- We will tell you within 10 business days whether we consider it a vulnerability and what we plan to do.
- We aim to release a fix within 30 days of the report. If a fix will take longer, we will tell you why.

We reply in the advisory and keep you updated there as we work on a fix.

We do not run a bug bounty program and cannot pay for reports.

## Supported versions

We only fix security issues in the most recent release of `inspect_swe` on PyPI. We do not backport fixes to older versions. If you are running an older version, upgrade before reporting.

## Scope

Inspect SWE runs coding agents such as Claude Code and Codex CLI inside an Inspect sandbox. It downloads each agent's binary or package, installs it into the sandbox, and relays the agent's model calls back to Inspect, so the agent sees no model API keys. The agent and everything it runs are untrusted. The sections below describe what we treat as a vulnerability and what we do not.

### In scope

- A way for code in the sandbox to run code or read files on the host through Inspect SWE, or to get the host's model API keys or other credentials
- Inspect SWE installing an agent binary, package or tool that fails its checksum check, or using a cached copy that someone replaced
- API keys, tokens, or other secrets written to transcripts, logs, or error messages
- Flaws in our release process, packaging, or CI that would let someone publish or modify a release

### Out of scope

- What an agent does inside its sandbox. The sandbox exists to contain it.
- Escapes from the sandbox runtime itself, such as Docker or Kubernetes. Report those to that project.
- Vulnerabilities in the agents themselves (Claude Code, Codex CLI, Gemini CLI and the others) should be reported to their maintainers.
- Vulnerabilities in `inspect_ai` should be reported under [its security policy](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/SECURITY.md).
- Vulnerabilities in other third-party dependencies should be reported to that project. Tell us if Inspect SWE needs a change in response, such as a version pin.
- Reports produced by a security scanner, or written mainly by an automated tool or a language model, with no working proof of concept. We will close these without a detailed response.

## Disclosure

When the fix is released, we publish the advisory from your report and request a CVE through GitHub where one applies. We will tell you before we publish it. The advisory credits you unless you ask us not to.

Please keep the details private until the advisory is published.
