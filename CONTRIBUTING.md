# Contributing to cislunar-sim

Thank you for your interest in contributing. This document explains the process for submitting changes.

## Getting started

1. Fork the repository and clone your fork.
2. Create a virtual environment and install the development dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```
3. Run the test suite to confirm your environment is working:
   ```bash
   make test
   ```

## Branch naming

| Type | Pattern | Example |
|---|---|---|
| Feature | `feat/<short-description>` | `feat/j4-zonal-harmonic` |
| Bug fix | `fix/<short-description>` | `fix/eclipse-terminator-edge` |
| Documentation | `docs/<short-description>` | `docs/mascon-derivation` |
| Refactor | `refactor/<short-description>` | `refactor/integrator-error-control` |
| Validation | `validation/<short-description>` | `validation/hayabusa2-flyby` |

Branch off `main`. Keep branches short-lived and focused.

## Pull requests

- **One logical change per PR.** Physics fixes, new force models, and guidance laws should be separate PRs.
- **All PRs must pass CI** (lint + tests on Python 3.11, 3.12, 3.13, and 3.14).
- **New force models require a validation benchmark.** Add at least one test in the most relevant `tests/test_*.py` module (or a new focused module) that checks an analytically known limit, such as a zero-thrust circular orbit staying circular.
- **New guidance laws require a reproducible orbit scenario** demonstrating convergence.
- Write a clear PR description: what changed, why, and how you verified it.

## Code style

The project uses `ruff` for linting and formatting. Run `make lint` before committing, or install the pre-commit hooks (`pre-commit install`) to have this happen automatically.

Key conventions:
- Physics notation variables (`I`, `l`, `mu`) are not flagged as ambiguous — this is intentional.
- Line length is 100 characters.
- All quantities must be in SI units unless explicitly documented otherwise in the variable name or docstring.

## Adding validation data

External telemetry files (TLE archives, beacon frames) are not committed to the repository due to size and licensing. If your contribution requires new external data:

1. Document the data source, license, and download instructions in `validation/data/README.md`.
2. Write a fixture-based test (see `tests/fixtures/`) that runs in CI without the external file.
3. Write the full validation test with a `@pytest.mark.network` marker so it is only run explicitly via `pytest -m network`.

## Reporting issues

Please use the [GitHub issue tracker](https://github.com/arboreng/cislunar-sim/issues).

Include:
- Python version and OS
- Minimal reproducible example (a few lines, not a full mission script)
- Expected vs. actual output
- Whether the issue is a physics accuracy problem or a software bug

For **security vulnerabilities**, do not open a public issue — see [SECURITY.md](SECURITY.md).

## Scope

cislunar-sim is a **physics simulation library**. The intended scope is:

- Spacecraft dynamics: force models, integrators, state representations
- Guidance laws: steering algorithms for orbit raising and transfer
- Validation: benchmarks against published references and mission telemetry
- Utility: ephemeris, eclipse geometry, link budget tools

Out of scope: mission planning GUIs, ground station scheduling, and launch vehicle simulation.

## License

By contributing, you agree that your contributions will be licensed under the same license as this project. Please ensure you have the right to contribute the code you submit.
