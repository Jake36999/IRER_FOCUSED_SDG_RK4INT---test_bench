from jax import tree_util
import jax.numpy as jnp

class SimState:
    def __init__(self, **kwargs):
        self._state = kwargs

    @property
    def state(self):
        return self._state

    def update(self, **kwargs):
        self._state = {**self._state, **kwargs}

    def get(self, key):
        return self._state.get(key)

    def __repr__(self):
        return f"SimState({self._state})"

    def to_pytree(self):
        return tree_util.tree_flatten(self._state)

    @classmethod
    def from_pytree(cls, pytree):
        return cls(**tree_util.tree_unflatten(pytree))