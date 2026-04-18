## What
Brief description of what this PR does.

## Modelling decisions
<!-- Required for any change touching features, models, or evaluation -->
- What changed:
- Why:
- PR-AUC impact (before → after or expected direction):
- Leakage risk: none / low / assessed (explain)

## How to test
Steps to verify this works locally.

## Checklist
- [ ] `poetry run ruff check src/` passes
- [ ] `poetry run pytest --cov=src` passes with 85%+ coverage
- [ ] No raw data committed
- [ ] Notebook outputs stripped
- [ ] PR-AUC impact documented above (if model change)
