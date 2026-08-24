# AGENTS.md

## Pull Requests

- Title PRs as Conventional Commits (`<type>: <description>`)—we squash-merge, so the PR title becomes the commit message that drives releases; `pr-title-lint` enforces it
- `feat:`/`fix:` are for user-facing changes only: they headline the release notes and bump the version. `perf:`/`revert:` also appear in the notes (no bump); `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` are hidden
- Body lines starting with `<type>:` are parsed as extra changelog entries—don't begin description lines with a conventional-commit prefix unless that's intended
- Never edit `CHANGELOG.md`, version numbers, or `.release-please-manifest.json`—Release Please owns them
- See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines

## Agent Guardrails

- **Credential boundary**: treat the evaluated sandbox as untrusted. Real model-provider credentials never enter it—only dummy keys reach model inference. Other real credentials (e.g. MCP auth headers) may enter the sandbox for transport, but must stay outside every model- and tool-readable path.
- **Public API evolution**: add new parameters at the end of the signature, using `Literal` types with runtime validation for constrained options. Renaming or removing a released parameter needs a `**deprecated_args: Unpack[...]` shim; new public surface needs a concrete use case first.
