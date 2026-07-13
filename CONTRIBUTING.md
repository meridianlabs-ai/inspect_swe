# Contributing to Inspect SWE

Thanks for your interest in contributing to Inspect SWE, a suite of software engineering agents for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

Inspect SWE requires Python 3.10 or later. Clone the repository and install it in
editable mode with the `[dev]` optional dependencies:

```bash
git clone https://github.com/meridianlabs-ai/inspect_swe
cd inspect_swe
pip install -e ".[dev]"
```

If you use [uv](https://docs.astral.sh/uv/), you can instead sync all groups and
extras with:

```bash
make sync
```

To enable the pre-commit hooks:

```bash
make hooks
```

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

## Commit messages and releases

We use [Conventional Commits](https://www.conventionalcommits.org/). Because we
squash-merge, **the PR title becomes the commit message** — so the title is what
matters. Format it as `<type>: <description>`.

Releases are automated with [Release Please](https://github.com/googleapis/release-please):
**don't edit `CHANGELOG.md` or bump the version by hand.** Release Please reads the
merged commit types, opens a release PR that updates the changelog and version, and
merging that PR tags and publishes the release.

Choose the type deliberately — only `feat:` and `fix:` appear in the release notes
and drive the version bump:

| Type | Use for |
| --- | --- |
| `feat:` | a user-facing feature |
| `fix:` | a user-facing bug fix |
| `docs:`, `refactor:`, `perf:`, `test:`, `build:`, `chore:`, `ci:` | everything else — excluded from the release notes |

Anything that isn't a user-facing feature or fix should avoid `feat:`/`fix:` so it
stays out of the release notes.

## Reporting issues

Found a bug or have a feature request? Please open an issue on the
[GitHub issue tracker](https://github.com/meridianlabs-ai/inspect_swe/issues).
