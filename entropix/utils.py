import jax
import jax.numpy as jnp

@jax.jit
def stable_softmax(x):  # Removed the axis parameter
    """Numerically stable softmax."""
    shifted_x = x - jax.lax.stop_gradient(jnp.max(x, axis=-1, keepdims=True)) # Hardcoded axis=-1
    return jax.nn.softmax(shifted_x, axis=-1)  # Hardcoded axis=-1
