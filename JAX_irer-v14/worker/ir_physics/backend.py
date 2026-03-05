"""
worker/ir_physics/backend.py
Lazy backend loader for JAX/NumPy abstraction.
"""
import os
import importlib

def lazy_load_backend():
    backend = os.getenv("IRER_BACKEND", "jax").lower()
    if backend == "numpy":
        np = importlib.import_module("numpy")
        fft = importlib.import_module("numpy.fft")
        random = importlib.import_module("numpy.random")
        return {
            "xp": np,
            "fft": fft,
            "random": random,
            "backend": "numpy"
        }
    else:
        jnp = importlib.import_module("jax.numpy")
        fft = importlib.import_module("jax.numpy.fft")
        random = importlib.import_module("jax.random")
        return {
            "xp": jnp,
            "fft": fft,
            "random": random,
            "backend": "jax"
        }
