# Step-by-Step Workflow: Running, Validating, and Extending Experiments

## 1. Setup
- Install Python 3.8+ and all dependencies for each module:
  ```
  pip install -r requirements.txt
  ```
- Ensure Redis and PostgreSQL are running and accessible.
- Set environment variables as needed (see each module's README).

## 2. Launch Orchestrator
- Start the orchestrator to queue jobs:
  ```
  python orchestrator/service.py
  ```

## 3. Launch Worker(s)
- Start one or more workers:
  ```
  python worker/worker_v14_sdg.py
  ```
- To switch backend (NumPy/JAX):
  ```
  set IRER_BACKEND=numpy  # or jax
  python worker/worker_v14_sdg.py
  ```

## 4. Run Validation
- After jobs complete, run the validation pipeline:
  ```
  python validation/validation_pipeline.py
  ```

## 5. Inspect Results
- Use the API or direct database queries to inspect job status and results.
- Artifacts are stored in MinIO (S3-compatible) and indexed in PostgreSQL.

## 6. Extending Experiments
- Add new kernels or analytics in `worker/ir_physics/`.
- Update orchestrator or validation logic as needed.
- Add/expand tests in `worker/tests/` and `validation/tests/`.

## 7. Reproducibility
- Every run logs library versions and environment details (see LOGGING_AND_REPRODUCIBILITY.md).
- Archive all logs and manifests for future reference.
