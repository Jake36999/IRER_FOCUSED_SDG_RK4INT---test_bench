# ASTE V11.0 Architecture

## Quickstart Guide

1. **Clone the repository:**
   ```sh
   git clone <your-repo-url>
   cd golden_hunter_low_SSE
   ```
2. **Set up a Python environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Run the Orchestrator:**
   ```sh
   python adaptive_hunt_orchestrator.py --config Alethiea/configs/config_true_golden.json
   ```
4. **Launch the UI:**
   ```sh
   uvicorn app:app --reload
   # Then open http://localhost:8000/static/ in your browser
   ```
5. **Validate Results:**
   ```sh
   python validation_pipeline.py --input simulation_data/rho_history_high_sse.h5 --params Alethiea/configs/config_true_golden.json --output_dir provenance_reports/
   ```

---

## Architecture Overview

- **adaptive_hunt_orchestrator.py**: Main orchestration logic for adaptive evolutionary search.
- **aste_hunter.py**: Evolutionary brain, manages population and selection.
- **worker_unified.py**: Physics engine, JAX hot-plane simulation.
- **validation_pipeline.py**: Post-simulation validation, provenance, and metrics.
- **app.py**: FastAPI backend, WebSocket, and static file serving.
- **UI/**: Frontend HTML/JS for real-time monitoring and control.
- **metrics/**: Analysis, validation, and collapse dynamics modules.
- **Alethiea/**: Diagnostics, configs, and gravity modules.
- **provenance_reports/**: Output CSVs and provenance logs.
- **tests/**: Unit and integration tests.

---

## Environment Setup

- Python 3.10+ recommended
- All dependencies in `requirements.txt`
- For full reproducibility, run:
  ```sh
  pip freeze > requirements.txt
  ```
- Hardware: Modern CPU, 32GB+ RAM, NVIDIA GPU recommended for large runs

---

## Launching the UI

- Start the FastAPI server:
  ```sh
  uvicorn app:app --reload
  ```
- Open your browser to [http://localhost:8000/static/](http://localhost:8000/static/)
- The UI will auto-reconnect to the backend and display live GIF/metrics updates.

---

## CLI Help

- All main scripts support `--help` for usage and options.
- Example:
  ```sh
  python adaptive_hunt_orchestrator.py --help
  ```

---

## Example Notebook

See `notebooks/ASTE_Quickstart.ipynb` for a hands-on workflow: loading simulation data, extracting fields, and visualizing topological structures.
