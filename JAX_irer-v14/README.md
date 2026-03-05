# IRER V14 Architecture

## Microservices Architecture: Redis Job Queue

This version uses Redis as the central job queue for orchestrator/worker communication. Ensure Redis is running (see `redis/README.md`).

**Quickstart:**
- Install Redis (see below)
- Start Redis server
- Set `REDIS_URL` in orchestrator and worker configs if needed

**Why Redis?**
- Removes disk I/O bottlenecks
- Enables multiple workers

See each service's README for details.

## Overview
The IRER V14 project is designed to pivot to a Microservices Architecture to address the "Stability-Fidelity Paradox" and enable massive parallel hunting. This document provides an overview of the project's structure, goals, and key components.

## Project Structure
The project is organized into several modules, each responsible for different aspects of the architecture:

- **Orchestrator**: Manages tasks and job manifests, interfacing with Redis for job queue management.
- **Worker**: Implements the new worker for processing Spacetime-Density Gravity (SDG) tasks, enforcing strict governance.
- **API**: Sets up a stateless FastAPI application for viewing job statuses.
- **Validation**: Contains logic for validating physics results before database storage.
- **Data**: Manages database migration and related tasks.
- **Redis**: Handles the setup and usage of Redis within the project.

## Goals
1. **Decouple Components**: Separate the "Brain" (Hunter), "Muscle" (Worker), and "Face" (API) to enhance stability and scalability.
2. **Improve Physics Core**: Transition to a clean, stable JAX core for physics calculations.
3. **Enhance Data Governance**: Ensure all results are traceable and valid, eliminating data loss.

## Setup Instructions
1. Clone the repository.
2. Navigate to each module directory and install the required dependencies listed in the `requirements.txt` files.
3. Follow the specific README files in each module for detailed setup and usage instructions.

## Key Changes from V13 to V14
- **Communication**: Transitioned from JSON files on disk to a Redis Task Queue for improved performance.
- **Orchestration**: Moved from a unified engine to a dedicated orchestrator service for better decoupling.
- **Physics Engine**: Updated to a pure SDG gravity implementation, removing outdated logic.
- **Data Management**: Migrated from SQLite to PostgreSQL for scalable data storage.

## Conclusion
The IRER V14 architecture aims to provide a robust framework for conducting simulations with improved stability, scalability, and data integrity. For more detailed information, refer to the individual module README files.