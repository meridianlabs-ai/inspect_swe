# Contributing to Inspect SWE

Thanks for your interest in contributing to Inspect SWE, a suite of software engineering agents for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

Inspect SWE requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
0.13.4 or later. Clone the repository and install it in editable mode with the
`[dev]` optional dependencies:

```bash
git clone https://github.com/meridianlabs-ai/inspect_swe
cd inspect_swe
uv sync --locked --extra dev
```

`uv sync --locked` creates `.venv` with the exact versions in `uv.lock`, the `[dev]` extra included, and installs Inspect SWE editable. It fails if `uv.lock` is out of date with `pyproject.toml`. Run Python commands from the environment: `uv run <command>`, `.venv/bin/<command>`, or `source .venv/bin/activate`. `uv sync --locked` without `--extra dev` removes the dev tools again.

To enable the pre-commit hooks:

```bash
make hooks
```

### Dependencies

[DEPENDENCIES.md](https://github.com/meridianlabs-ai/actions/blob/main/DEPENDENCIES.md) in meridianlabs-ai/actions tells how the Inspect repositories install and change dependencies. It covers the lock file, the cooldown, the canaries, inspect_ai `main`, the release build tools, and the procedures to add or update a dependency. Follow it. It is the same for every Inspect repository, so this file does not repeat it.

For this repository:

- inspect_swe needs unreleased inspect_ai changes. `[tool.uv.sources]` in `pyproject.toml` points `inspect-ai` at inspect_ai `main`.
- The 6-hour packages are in `[tool.uv.exclude-newer-package]` in `pyproject.toml`.

## Checks and tests

Run linting, formatting, and type checking:

```bash
make check
```

This runs `ruff format`, `ruff check --fix`, and `mypy src tests`.

Run the test suite:

```bash
make test
```

Most tests depend on a valid sandbox being available (either `docker` or `k8s`),
which is inferred from your shell environment. You can check which tests are
collected via:

```bash
pytest --co
```

## Documentation

The docs packages (including Quarto) are in the `[doc]` extra, which `[dev]` does
not include. Install it alongside `[dev]`:

```bash
uv sync --locked --extra dev --extra doc
```

Then preview the docs:

```bash
cd docs
quarto preview
```

## Commit messages and releases

We use [Conventional Commits](https://www.conventionalcommits.org/). Because we
squash-merge, **the PR title becomes the commit message** — so the title is what
matters. Format it as `<type>: <description>`.

Releases are automated with [Release Please](https://github.com/googleapis/release-please):
**don't edit `CHANGELOG.md` or bump the version by hand.** Release Please reads the
merged commit types, opens a release PR that updates the changelog and version, and
merging that PR tags the release; the publish then runs once a maintainer approves
the deployment.

Choose the type deliberately — `feat:` and `fix:` drive the version bump and
headline the release notes:

| Type | Effect |
| --- | --- |
| `feat:` | a user-facing feature — bumps the patch version (pre-1.0 policy) |
| `fix:` | a user-facing bug fix — bumps the patch version |
| `perf:`, `revert:` | appear in the release notes (no bump on their own) |
| `docs:`, `refactor:`, `chore:`, `build:`, `ci:`, `test:`, `style:` | hidden from the release notes |

Anything that isn't a user-facing feature or fix should avoid `feat:`/`fix:` so it
stays out of the headline sections.

## Reporting issues

Found a bug or have a feature request? Please open an issue on the
[GitHub issue tracker](https://github.com/meridianlabs-ai/inspect_swe/issues).
