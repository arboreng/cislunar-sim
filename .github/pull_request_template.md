## What changed and why

<!-- One paragraph: what is this PR doing and what motivated it? -->

## How it was verified

<!-- Describe the test or benchmark that proves the change is correct.
     New force models: include the analytically known limit you checked.
     New guidance laws: describe the orbit scenario and convergence metric.
     Bug fixes: paste the before/after output. -->

## Checklist

- [ ] `make lint` passes (or `pre-commit run --all-files`)
- [ ] `make test` passes on Python 3.11, 3.12, 3.13, and 3.14
- [ ] New force model: at least one focused test in the relevant `tests/test_*.py` module with an analytically known limit
- [ ] New guidance law: reproducible scenario demonstrating convergence is documented or tested
- [ ] CHANGELOG.md updated under `[Unreleased]`
- [ ] All quantities in SI units (or explicitly documented otherwise)
