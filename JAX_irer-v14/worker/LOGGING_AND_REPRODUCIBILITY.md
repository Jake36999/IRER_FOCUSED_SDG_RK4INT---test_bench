# Logging Environment and Library Versions

To ensure reproducibility, every run logs the following:
- Python version
- All installed package versions (via `pip freeze` or `importlib.metadata`)
- Key environment variables (e.g., `IRER_BACKEND`, `REDIS_HOST`, etc.)

This is automatically appended to each artifact and manifest. For custom scripts, use:

```python
import sys
import os
import importlib.metadata
print('Python', sys.version)
print('Backend:', os.getenv('IRER_BACKEND', 'jax'))
for dist in importlib.metadata.distributions():
    print(dist.metadata['Name'], dist.version)
```

For full reproducibility, always archive the output of these commands with your results.
