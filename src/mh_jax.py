import jax
import jax.numpy as jnp
from functools import partial
from model_jax import Params, utility


# ---------------------------------------------------------------------------
# Metropolis-Hastings sampler — fully JAX-jittable
#
# Key changes vs. the numpy version:
#   1. PRNG keys are threaded explicitly (JAX functional-random convention).
#   2. np.flatnonzero / np.append (data-dependent shapes) are replaced with
#      fixed-size masked arithmetic so shapes are static and jit-friendly.
#   3. Python if-branches on array values are replaced with jnp.where.
#   4. The Python for-loop is replaced with jax.lax.scan, which unrolls into
#      a single XLA computation.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# proposal
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("L",))
def proposal(key: jax.Array, q: jax.Array, L: int) -> jax.Array:
    """
    Propose a neighbouring bundle by moving one unit between two slots.

    Instead of building a variable-length set P and sampling from it, we:
      - Draw l_src uniformly from {0, 1, …, L}.
        If l_src == 0, the source is "nothing" (no unit is removed).
        If l_src > 0,  but q[l_src-1] == 0, we treat it the same as 0
        (cannot take from an empty slot) — this is equivalent to the
        original P ∪ {0} scheme because the original code first builds P
        (positive slots) and then appends 0; sampling from that set is
        proportional to |P|+1 outcomes.  The slight difference is that
        here we sample from all L+1 indices uniformly, which changes the
        proposal distribution slightly when some slots are empty.
        To preserve the *exact* original kernel:

        Original kernel:
          - l_src ~ Uniform(P ∪ {0}),  where P = {i : q[i] > 0}  (1-indexed)
          - l_dst ~ Uniform({0,1,…,L})

        We replicate this exactly using weight-based sampling:
          Build a weight vector of length L+1 for the source:
            w[0]   = 1          (the "nothing" option)
            w[i+1] = (q[i] > 0) for i in 0..L-1
          Then sample categorically from these weights.

    Args:
        key : JAX PRNG key
        q   : current bundle, shape (L,)
        L   : number of products (static — must be known at compile time)

    Returns:
        q_new : proposed bundle, shape (L,)
    """
    key_src, key_dst = jax.random.split(key)

    # --- source slot (1-indexed; 0 = "take nothing") ---
    # weight[0] = 1 always; weight[i+1] = indicator(q[i] > 0)
    w_src = jnp.concatenate([jnp.ones(1), (q > 0).astype(jnp.float32)])  # (L+1,)
    l_src = jax.random.choice(key_src, L + 1, p=w_src / w_src.sum())     # scalar

    # --- destination slot (1-indexed; 0 = "add nothing") ---
    l_dst = jax.random.randint(key_dst, shape=(), minval=0, maxval=L + 1) # scalar

    # Build one-hot indicator vectors (length L).
    # Index 0 maps to the zero vector (no-op).
    idx   = jnp.arange(1, L + 1)                # [1, 2, …, L]
    e_src = jnp.where(idx == l_src, 1.0, 0.0)   # (L,)
    e_dst = jnp.where(idx == l_dst, 1.0, 0.0)   # (L,)

    return q - e_src + e_dst


# ---------------------------------------------------------------------------
# transition_prob
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("L",))
def transition_prob(q_from: jax.Array, q_to: jax.Array, L: int) -> jax.Array:
    """
    Probability of proposing q_to from q_from under the proposal kernel.

    g(q_to | q_from) = 1/(L+1) * 1/(|P(q_from)|+1)   if q_to != q_from
                     = 1/(L+1)                          if q_to == q_from
    where |P(q_from)| is the number of positive entries in q_from.

    Returns a scalar JAX array.
    """
    n_positive = jnp.sum(q_from > 0)                     # |P|
    is_same    = jnp.all(q_from == q_to)

    # Probability for the "stay" outcome (l_src == l_dst) collapsed into one:
    prob_diff  = 1.0 / ((L + 1) * (n_positive + 1))
    prob_same  = 1.0 / (L + 1)

    return jnp.where(is_same, prob_same, prob_diff)


# ---------------------------------------------------------------------------
# one MH step  (jit-compiled)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("L",))
def mh_step(
    carry: tuple,            # (key, q_current)
    _,                       # dummy scan input
    params: Params,
    p_jnp: jax.Array,
    L: int,
) -> tuple:
    """
    One Metropolis-Hastings step, suitable for use inside jax.lax.scan.

    carry : (key, q)  — PRNG key and current bundle
    _     : unused scan input (we pass jnp.arange(n_samples) as xs)

    Returns:
        new_carry : (new_key, accepted_q)
        output    : accepted_q  (stored by scan)
    """
    key, q = carry

    key, key_prop, key_unif = jax.random.split(key, 3)

    q_new = proposal(key_prop, q, L)

    u_curr = utility(params, q,     p_jnp)
    u_prop = utility(params, q_new, p_jnp)

    g_fwd = transition_prob(q,     q_new, L)
    g_rev = transition_prob(q_new, q,     L)

    log_alpha = (u_prop - u_curr) + (
        jnp.log(g_rev + 1e-300) - jnp.log(g_fwd + 1e-300)
    )
    alpha = jnp.minimum(1.0, jnp.exp(log_alpha))

    accept  = jax.random.uniform(key_unif) < alpha
    q_next  = jnp.where(accept, q_new, q)

    return (key, q_next), q_next


# ---------------------------------------------------------------------------
# mh_sample  (public API)
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_samples",))
def mh_sample(
    key: jax.Array,
    params: Params,
    q_init: jax.Array,
    p: jax.Array,
    n_samples: int,
) -> jax.Array:
    """
    Run a Metropolis-Hastings chain and return all sampled bundles.

    Args:
        key      : JAX PRNG key
        params   : model parameters (Params NamedTuple)
        q_init   : initial consumption bundle, shape (L,)
        p        : price vector, shape (L,)
        n_samples: total number of MH steps (static — must be a Python int)

    Returns:
        samples : (n_samples, L) float32 array — one row per MH step.
                  Burn-in is NOT discarded here; caller handles it.
    """
    q     = q_init.flatten().astype(jnp.float32)
    p_jnp = p.flatten().astype(jnp.float32)
    L     = q.shape[0]                  # static after tracing

    step_fn = partial(mh_step, params=params, p_jnp=p_jnp, L=L)

    _, samples = jax.lax.scan(step_fn, (key, q), xs=None, length=n_samples)
    # samples shape: (n_samples, L)
    return samples
