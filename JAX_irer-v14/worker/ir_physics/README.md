# IR Physics Submodule

## Backend Abstraction
This module supports both JAX and NumPy backends for all numerical operations. The backend is selected at runtime via the `IRER_BACKEND` environment variable (`jax` or `numpy`).

- All core physics routines (kernels, solver, stability envelope) are backend-agnostic.
- To switch backend, set `IRER_BACKEND=numpy` or `IRER_BACKEND=jax` before running the worker.

## File Structure
- `kernels.py`: Backend-agnostic physics kernels.
- `solver.py`: Simulation scan engine (uses backend abstraction).
- `stability_envelope.py`: Stability mapping tools.
- `models.py`: Immutable state and parameter definitions.
- `backend.py`: Lazy backend loader for JAX/NumPy.

## Extending
To add new physics routines, import `lazy_load_backend` and use `xp`, `fft`, and `random` from the backend dictionary for all array and math operations.

## Example
```python
from .backend import lazy_load_backend
backend = lazy_load_backend()
xp = backend["xp"]
# Use xp for all array ops
```

## Reproducibility
All runs log library versions and environment details. See the main worker README for workflow and logging details.
