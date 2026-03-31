import numpy as np
import jax.numpy as jnp
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Neighborhood set generator
# Identical logic to the original function.py; kept in NumPy so it can be
# called freely outside jit-compiled code without PRNG key management.
# ---------------------------------------------------------------------------

def random_set(
    q_tilde: Iterable,
    m: int,
    I: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate I consumption bundles close to q_tilde (NTR set).

    Args:
        q_tilde : observed consumption bundle, 1-D non-negative integer array
        m       : maximum units to shift
        I       : number of bundles to generate (first row is q_tilde itself)
        rng     : optional numpy random Generator for reproducibility

    Returns:
        B_np : (I, L) float ndarray; first row is q_tilde.
               Returns (1, L) if the neighbourhood is too small (m*|Z|*|P| < I).
    """
    if rng is None:
        rng = np.random.default_rng()

    q_tilde = np.asarray(list(q_tilde), dtype=float)
    if q_tilde.ndim != 1:
        raise ValueError("q_tilde must be a 1-D vector.")
    if np.any(q_tilde < 0):
        raise ValueError("q_tilde must be in N_0^L (all entries >= 0).")
    if m <= 0 or I <= 0:
        raise ValueError("m and I must be positive integers.")

    Z = np.flatnonzero(q_tilde < 1)            # zero-consumption indices
    P = np.flatnonzero((q_tilde >= 1) & (q_tilde <= m))  # positive indices

    B_np = np.zeros((I, len(q_tilde)), dtype=float)
    B_np[0] = q_tilde

    #if m * len(Z) * len(P) < I:
        #return B_np[0:1, :]                    # neighbourhood too small

    for i in range(1, I):
        if len(P)==0:
            tilde_ell = rng.choice(Z)
            k        = int(rng.integers(1, m + 1))

            q          = q_tilde.copy()
            q[tilde_ell] = q[tilde_ell] + k
        else:
            hat_ell=rng.choice(P)
            tilde_pool = Z if len(Z) > 0 else np.arange(len(q_tilde))
            tilde_ell = rng.choice(tilde_pool)
            k        = int(rng.integers(1, m + 1))

            q          = q_tilde.copy()
            q[hat_ell] = 0
            q[tilde_ell] = q[tilde_ell] + k

        B_np[i] = q

    return B_np
