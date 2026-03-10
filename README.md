# IRER — Intelligent Relativistic Evolution Reservoir

**IRER** is an automated physics-discovery framework that simulates and searches the parameter space of a relativistic quantum-fluid system. It combines a JAX-accelerated RK4 PDE solver with a genetic-algorithm orchestrator, microservice infrastructure, and a live monitoring frontend to find physical configurations that satisfy a strict Hamiltonian constraint.

---

## Table of Contents

- [Overview](#overview)
- [Physics Background](#physics-background)
- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [Quick Start (Docker)](#quick-start-docker)
- [Manual Setup](#manual-setup)
- [Configuration & Parameters](#configuration--parameters)
- [Tech Stack](#tech-stack)
- [Known Issues & Development Goals](#known-issues--development-goals)

---

## Overview

IRER solves a complex-scalar-field PDE on a 32³ grid using 4th-order Runge-Kutta integration, then validates whether the resulting field satisfies a geometric Hamiltonian constraint. An evolutionary hunter continuously generates candidate parameter sets, runs batches of simulations in parallel, scores each run, and promotes the best-performing configurations to the next generation — all governed by an append-only PostgreSQL ledger.

**Key capabilities**

| Capability | Implementation |
|---|---|
| PDE integration | RK4 (4th-order), JAX JIT-compiled |
| Parameter search | Genetic algorithm (80 % elite + 20 % random diversity) |
| Validation | Hamiltonian H-norm + persistent homology (Betti curves) |
| Provenance | SHA-256 config hash → `kel_runs` PostgreSQL table |
| Artifacts | 4-D HDF5 snapshots stored in MinIO |
| Monitoring | FastAPI REST/WebSocket API + Next.js "God View" UI |

---

## Physics Background

The simulation evolves a complex scalar field **A** on a 3-D lattice:

```
∂A/∂t = ε·A + (1 + i·c₁)·∇²_cov·A − (1 + i·c₃)·ρ·A + splash

where:
  A         — complex scalar field  (32 × 32 × 32)
  ρ = |A|²  — local density
  ∇²_cov    — covariant Laplacian with geometric feedback ω
  splash    — FFT non-local coupling (Green's function weight)
```

### Geometric feedback (SDG — Spacetime-Density Gravity)

The geometry proxy **ω** encodes how local density curves the effective space (hat notation denotes the Fourier transform):

```
ω    = (ρ_vac / (ρ + ε))^(α/2)       [proxy geometry]
ω̂(k) = α · ρ̂(k) / k²                [spectral SDG, Fourier space]
```

### Hamiltonian constraint

A run is valid when the residual falls below the tolerance threshold:

```
H = ∇²ω + α·ρ·ω ≈ 0    (max_h_norm < 1e-5)
```

### Critical parameters

| Symbol | Default range | Role |
|---|---|---|
| `alpha` | [0.0, 10.0] | Geometric coupling strength (**most sensitive**) |
| `epsilon` | [−10, 10] | Linear amplitude growth / decay |
| `c1` | [−10, 10] | Complex diffusion coefficient |
| `c3` | [−10, 10] | Non-linear phase coupling |
| `splash_fraction` | [0, 10] | Non-local (Green's function) weight |
| `dt` | [1e-8, 1.0] | RK4 timestep (CFL-bounded) |
| `sigma_k` | [1e-8, 1e8] | Density sensitivity in ω feedback |
| `dx` | [1e-8, 1e8] | Grid spacing (typically 1.0) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Orchestrator  (Brain)                                  │
│  Genetic hunter → job manifests (SHA-256 config hash)  │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │    Redis  7+        │  ← job queue / result queue
              └──────────┬──────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Worker Pool  (Muscle)                                  │
│  JAX RK4 solver  ·  32³ grid  ·  1000 steps            │
│  Uploads 4-D HDF5 artifacts → MinIO                    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Validation  (Gatekeeper)                               │
│  H-norm check  ·  Betti curves (Ripser)                │
│  Writes VALID / INVALID → PostgreSQL (kel_runs)        │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  PostgreSQL  15+    │  ← governance ledger
              └─────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  API  (Face)                                            │
│  FastAPI  ·  HTTP + WebSocket telemetry                │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Testing Suite  (Oversight)                             │
│  AST-walk code scanner  ·  Next.js "God View" UI       │
└─────────────────────────────────────────────────────────┘
```

All services communicate over the `irer-mesh` Docker bridge network.

---

## Repository Layout

```
.
├── JAX_irer-v14/                    ← ✅  Active production version
│   ├── orchestrator/                   Genetic hunter + PostgreSQL KEL
│   ├── worker/
│   │   └── ir_physics/                JAX kernels, RK4 solver, data models
│   ├── validation/                     H-norm + Betti curve gatekeeper
│   ├── api/                            FastAPI read-only endpoint
│   ├── redis/                          Redis config & helper docs
│   ├── data/                           DB migration scripts
│   ├── docker-compose.yml              7-service stack definition
│   ├── WORKFLOW.md                     Step-by-step execution guide
│   ├── TESTING_AND_CI.md              Test infrastructure docs
│   └── README.md                       Module-level architecture notes
│
├── fully decoupled, IPC-safe,
│   asynchronous multi-GPU pipeline/  ← ✅  ASTE v11 (mature, 22 viz scripts)
│
├── Testing_suite/                    ← 🚧  AST scanner + Next.js God View UI
│
├── [legacy] cupy_portable_version/  ← ❌  Broken — do not use (see below)
│
└── agent_server.py                      FastAPI dev file server
```

---

## Quick Start (Docker)

The fastest way to run the full stack is via Docker Compose.

```bash
# 1. Navigate to the active version
cd JAX_irer-v14

# 2. Start all 7 services
docker compose up --build

# 3. Monitor the API
curl http://localhost:8000/status

# 4. (Optional) Open MinIO console
open http://localhost:9001   # admin: irer_minio_admin / irer_minio_password
```

Services that start:

| Service | Port | Description |
|---|---|---|
| Redis | 6379 | Job queue |
| PostgreSQL | 5432 | Governance ledger |
| MinIO | 9000 / 9001 | HDF5 artifact store |
| Orchestrator | — | Genetic hunter |
| Worker | — | JAX RK4 physics |
| Validation | — | Hamiltonian gatekeeper |
| API | 8000 | HTTP + WebSocket |

---

## Manual Setup

```bash
# Install Python dependencies per module
pip install -r JAX_irer-v14/orchestrator/requirements.txt
pip install -r JAX_irer-v14/worker/requirements.txt
pip install -r JAX_irer-v14/validation/requirements.txt
pip install -r JAX_irer-v14/api/requirements.txt

# Choose compute backend (default: jax)
export IRER_BACKEND=jax      # or: numpy

# Start services individually (ensure Redis + PostgreSQL are reachable)
python JAX_irer-v14/orchestrator/service.py
python JAX_irer-v14/worker/worker_v14_sdg.py
python JAX_irer-v14/validation/service.py
```

See [`JAX_irer-v14/WORKFLOW.md`](JAX_irer-v14/WORKFLOW.md) for the full step-by-step guide.

---

## Configuration & Parameters

Search bounds are read from `burn_in_config.json`. A fixed golden-run configuration is available in `config_true_golden.json`.

Key environment variables:

| Variable | Default | Description |
|---|---|---|
| `IRER_BACKEND` | `jax` | Physics backend (`jax` or `numpy`) |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `DB_HOST` / `DB_USER` / `DB_PASS` | — | PostgreSQL connection |
| `MINIO_ENDPOINT` | — | MinIO address |
| `BATCH_SIZE` | `4` | Jobs per orchestrator cycle |
| `WORKER_ID` | — | Unique worker identifier |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Physics core | [JAX](https://github.com/google/jax) (JIT/GPU) + NumPy fallback |
| Job queue | Redis 7+ |
| Governance DB | PostgreSQL 15+ |
| Artifact store | MinIO (S3-compatible) · HDF5 |
| API | FastAPI + Uvicorn |
| Topology (TDA) | Ripser (persistent homology) |
| Visualisation | Matplotlib · Seaborn · PyVista |
| Frontend UI | Next.js 14 · Tailwind CSS · React |
| Testing | pytest · pytest-asyncio |
| Deployment | Docker · Docker Compose |
| Language | Python 3.8+ · TypeScript |

---

## Known Issues & Development Goals

> **⚠️ CuPy legacy version is mathematically incorrect.**  
> The conformal-carry tuple in the RK4 solver and the omega gravity function are both broken.  
> **Do not use `[legacy] cupy_portable_version/`.**  
> `JAX_irer-v14` is the only production-ready implementation.

### Active development goals

1. **Robust ω calculation** — implement rigorous validation and falsification metrics for the omega field to prevent false positives.  
2. **GPU acceleration without JAX** — find a portable GPU backend (e.g. CuPy fixed, CUDA direct) that avoids JAX's Windows compatibility issues.  
3. **Mathematical fidelity via AST analysis** — complete the AST-walk scanner in `Testing_suite/` to automate verification of kernel correctness at the source-code level.

---

*For detailed module documentation see the README files inside each subdirectory.*
