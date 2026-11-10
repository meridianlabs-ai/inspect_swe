# Contributing to Inspect SWE

Thanks for your interest in contributing to Inspect SWE, a suite of software engineering agents for [Inspect AI](https://inspect.aisi.org.uk/).

## Development setup

Inspect SWE requires Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
0.13.4 or later. Clone the repository and install it in editable mode with the
`[dev]` optional dependencies:

```bash
git clone https://github.com/meridianlabs-ai/inspect_swe
cd inspect_swe
uv sync --extra dev
```

`uv sync` creates `.venv` with the exact versions in `uv.lock`, the `[dev]` extra included, and installs Inspect SWE editable. Run Python commands from it: `uv run <command>`, `.venv/bin/<command>`, or `source .venv/bin/activate`. `uv sync` without `--extra dev` removes the dev tools again.

To enable the pre-commit hooks:

```bash
make hooks
```

### Dependencies and the cooldown

`uv.lock` pins every Python dependency except the build toolchain. CI's lint, type-check, test and build jobs install their dependencies from it, and fail when `uv.lock` is out of date with `pyproject.toml`. CI's build job and the release build use the build toolchain (build, hatchling, hatch-vcs) pinned with hashes in `scripts/build-dist/requirements.txt`. Dependabot updates that file weekly, and `.github/workflows/publish.yml` gives the command to update it by hand. When you change dependencies in `pyproject.toml`, run `uv lock` and commit both files.

`pyproject.toml` sets `exclude-newer = "7 days"` under `[tool.uv]`, so `uv lock`, `uv sync` and `uv pip` only pick releases that have been on PyPI for at least seven days. A compromised release is usually found and removed within that time, before it reaches `uv.lock`, CI or your machine. A few packages wait 6 hours instead, so Inspect SWE can adopt a new release the day it ships: `inspect-ai`, and the `anthropic` and `openai` SDKs. They are listed with `"6 hours"` in `[tool.uv.exclude-newer-package]`. Only a direct dependency (a package in `dependencies` or the `dev` extra in `pyproject.toml`) that is a model provider SDK or an Inspect package gets this exception. Every other package keeps the seven-day cooldown. Dependabot's `uv` entry in `.github/dependabot.yml` repeats the cooldown and the list, except `inspect-ai`, which it ignores. When you add a `"6 hours"` entry, also add the package to `cooldown.exclude` and the `six-hour-cooldown` group there. `tests/test_dependency_cooldown.py` checks that each entry is a direct dependency, that dated entries (below) are less than seven days old, and that the two files match. Whether a package is a provider SDK or an Inspect package is up to review.

Dependabot proposes `uv.lock` updates daily, once a release has cleared its cooldown. To adopt a release sooner, run `uv lock --upgrade-package <package>` and commit `uv.lock`. A release still inside its cooldown can't be locked; wait for it rather than overriding `exclude-newer`, unless it's an emergency (below).

In an emergency, for example a security fix, you can adopt a release that is less than seven days old with a one-time dated entry:

1. Under `[tool.uv.exclude-newer-package]` in `pyproject.toml`, add `<package> = "<timestamp>"`. The package must be a direct dependency. Use an RFC 3339 timestamp just after the upload time of the fixed release. Don't add a dated entry to `.github/dependabot.yml`.
2. Run `uv lock --upgrade-package <package>` and commit both files.
3. Within seven days, once the normal cooldown includes the release, delete the entry, run `uv lock --upgrade-package <package>`, and commit both files. A plain `uv lock` doesn't reliably remove the entry from `uv.lock`, and CI then fails. A test fails when a dated entry is seven days old or older.

Testing against the newest release of every dependency happens in the nightly canary in [meridianlabs-ai/canaries](https://github.com/meridianlabs-ai/canaries), so Inspect SWE's own CI doesn't need to.

### inspect_ai main

Development and CI track inspect_ai's `main` branch. `[tool.uv.sources]` in `pyproject.toml` points `inspect-ai` at its git repository, and `uv.lock` records a commit. Only uv reads `[tool.uv.sources]`: the published package requires the `inspect-ai>=` floor in `[project]`.

- CI moves inspect-ai to the latest `main` commit before every install (`.github/actions/uv-sync`), so it always tests against inspect_ai main. Other packages keep their locked versions unless inspect_ai main needs newer ones.
- `uv sync --extra dev` installs the commit in `uv.lock`. To move to the latest `main` commit, run `uv sync --upgrade-package inspect-ai --extra dev`. Committing the updated `uv.lock` is optional.
- On the release PR, CI ignores the git source and installs the newest inspect-ai release that has cleared its cooldown, so the release is tested against the inspect_ai users will get. `.github/workflows/release-pin-deps.yml` raises the floor to that release. Main keeps the git source.

To work against a local inspect_ai checkout, install it over the locked one:

```bash
uv sync --extra dev
uv pip install -e ../inspect_ai
```

Then run commands with `.venv/bin/<command>` or `uv run --no-sync <command>`. A plain `uv run` or `uv sync` reinstalls the locked inspect_ai.

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
uv sync --extra dev --extra doc
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
