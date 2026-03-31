import jax
import jax.numpy as jnp
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Parameters container
# A plain NamedTuple acts as a JAX pytree natively,
# so jax.grad / jax.value_and_grad can differentiate through it directly.
# ---------------------------------------------------------------------------

class Params(NamedTuple):
    A: jnp.ndarray      # (K, L)
    b: jnp.ndarray      # (K, 1)
    delta: jnp.ndarray  # scalar — d = exp(delta) enforces d > 0


def init_params(key: jax.Array, K: int, L: int) -> Params:
    """
    Initialize parameters with small random values, matching the
    original PyTorch model initialisation (randn * 0.01).
    Requires an explicit JAX PRNG key.
    """
    key_A, key_b, key_delta = jax.random.split(key, 3)
    A     = jax.random.normal(key_A,     shape=(K, L))     * 0.01
    b     = jax.random.normal(key_b,     shape=(K, 1))     * 0.01
    delta = jax.random.normal(key_delta, shape=())         * 0.01
    return Params(A=A, b=b, delta=delta)


# ---------------------------------------------------------------------------
# Utility function
# U(q, p; params) = b'Aq  -  ||Aq||^2  -  exp(delta) * (p . q)
#
# Supports two calling modes:
#   • Single bundle : q shape (L,),   p shape (L,)   -> scalar
#   • Batched       : q shape (B, L), p shape (B, L) -> (B,)
# ---------------------------------------------------------------------------

def utility(params: Params, q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """
    Compute utility for a single bundle or a batch of bundles.

    Args:
        params : Params NamedTuple (A, b, delta)
        q      : consumption bundle(s),  shape (L,) or (B, L)
        p      : price vector(s),        shape (L,) or (B, L)

    Returns:
        Utility scalar (single) or shape (B,) array (batched).
    """
    d = jnp.exp(params.delta)                       # enforce d > 0

    batched = q.ndim == 2

    if batched:
        Aq            = q @ params.A.T              # (B, K)
        term_linear   = (Aq @ params.b).squeeze(-1) # (B,)
        term_quad     = -jnp.sum(Aq ** 2, axis=-1)  # (B,)
        term_price    = -d * jnp.sum(q * p, axis=-1)# (B,)
    else:
        Aq            = params.A @ q                # (K,)
        term_linear   = params.b.squeeze() @ Aq     # scalar
        term_quad     = -jnp.dot(Aq, Aq)            # scalar
        term_price    = -d * jnp.dot(q, p)          # scalar

    return term_linear + term_quad + term_price
