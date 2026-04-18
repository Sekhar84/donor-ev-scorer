# Contributing to donor-ev-scorer

## Branch naming

feature/{name}/{description}
fix/{name}/{description}
experiment/{name}/{description}

Examples:
  feature/sekhar/add-calibration-curves
  fix/sekhar/temporal-leakage-in-lag-features
  experiment/sekhar/catboost-vs-xgboost

## Workflow

1. Branch off main — never commit directly to main
2. Make your changes
3. Run tests locally before pushing: `poetry run pytest --cov=src`
4. Open a PR using the template
5. 1 approval required before merging

## Code standards

- Linter: ruff — run `poetry run ruff check src/` before pushing
- Tests: pytest — coverage must stay above 85%
- Notebooks: strip outputs before committing (nbstripout handles this via pre-commit)

## Modelling decisions

Any PR that changes model architecture, feature engineering, or evaluation 
methodology must include in the PR description:
- What changed and why
- PR-AUC before and after (or expected direction if not yet run)
- Any leakage risk assessment

## Pre-commit hooks

poetry run pre-commit install

Hooks run automatically on git commit:
- ruff (linting)
- nbstripout (strips notebook outputs)
