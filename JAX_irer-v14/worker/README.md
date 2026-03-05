# Worker Module Documentation

## Overview
The Worker module is responsible for processing tasks related to Spacetime-Density Gravity (SDG). It implements the new worker logic that enforces strict governance and ensures the integrity of job payloads through SHA-256 hash validation.

## File Structure
- `worker_v14_sdg.py`: Contains the main logic for the SDG worker.
- `ir_physics/`: A package that may include additional physics-related modules or classes.
- `simstate.py`: Defines the `SimState` Pytree for managing immutable simulation states.

## Setup Instructions
1. Ensure that you have Python 3.8 or higher installed.
2. Install the required dependencies by running:
   ```
   pip install -r requirements.txt
   ```

## Usage
To start the worker, execute the following command:
```
python worker_v14_sdg.py
```
Make sure that the Redis server is running and accessible.

## Governance
The worker enforces strict governance by refusing to start unless a valid SHA-256 hash is provided in the job payload. This ensures that only verified tasks are processed.

## Contribution
For contributions, please follow the project's contribution guidelines and ensure that all code adheres to the project's coding standards.