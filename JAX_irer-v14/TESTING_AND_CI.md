# Testing & Quality Gates

## Unit and Integration Tests
- All backend switching, analytics, and feedback logic must be covered by tests.
- Add/expand tests in:
  - `worker/tests/` (physics, backend abstraction)
  - `validation/tests/` (analytics, verdicts)

## Example: Backend Switching Test
```python
import os
os.environ['IRER_BACKEND'] = 'numpy'
from ir_physics.kernels import ... # test numpy ops
os.environ['IRER_BACKEND'] = 'jax'
from ir_physics.kernels import ... # test jax ops
```

## CI/CD Hooks
- Use `pytest`, `pytest-cov`, and `pytest-xdist` for parallel test runs and coverage.
- Add linting and type checks (e.g., `flake8`, `mypy`).
- Example GitHub Actions workflow:
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install flake8 mypy
      - name: Lint
        run: flake8 .
      - name: Type Check
        run: mypy .
      - name: Test
        run: pytest --cov=.
```

## Coverage
- Ensure all new features and bugfixes include tests.
- Monitor coverage reports and address gaps.
