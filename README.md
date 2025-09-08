Welcome to Inspect SWE, a suite of software engineering agents for [Inspect AI](https://inspect.aisi.org.uk/).

For details on using Inspect SWE, please visit <https://meridianlabs-ai.github.io/inspect_swe/>.

## Installation

Latest development version:

```bash
pip install git+https://github.com/meridianlabs-ai/inspect_swe
```

## Development

To work on development of Inspect SWE, clone the repository and install with the `-e` flag and `[dev]` optional dependencies:

```bash
git clone https://github.com/meridianlabs-ai/inspect_swe
cd inspect_swe
pip install -e ".[dev]"
```

Run linting, formatting, and tests via

```bash
make check
make test
```

### Sandbox-specific tests

Note that most tests depend on a valid sandbox being available (either `docker` or `k8s`), this is inferred from your shell environment.

You can check which are collected via `pytest --co`
