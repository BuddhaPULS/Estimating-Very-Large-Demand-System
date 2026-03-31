import numpy as np
import jax
import jax.numpy as jnp
import optax
import pandas as pd
from functools import partial
from tqdm import tqdm

from model_jax    import Params, init_params, utility
from function_jax import random_set
from mh_jax       import mh_sample


# ---------------------------------------------------------------------------
# Loss functions  (single-sample, used as building blocks for vmap)
# ---------------------------------------------------------------------------

def ntr_loss(
    params: Params,
    q_tilde: jnp.ndarray,   # (L,)
    B_ntr: jnp.ndarray,     # (I, L)
    p: jnp.ndarray,         # (L,)
) -> jnp.ndarray:
    """NTR negative log-likelihood for one observation."""
    p_rep = jnp.tile(p, (B_ntr.shape[0], 1))   # (I, L)
    utils = utility(params, B_ntr, p_rep)        # (I,)
    return -jax.nn.log_softmax(utils)[0]         # scalar


def contrastive_loss(
    params: Params,
    q_obs: jnp.ndarray,      # (L,)
    q_samples: jnp.ndarray,  # (S, L)
    p: jnp.ndarray,          # (L,)
) -> jnp.ndarray:
    """Contrastive loss -(U_obs - mean U_samples) for one observation."""
    u_obs     = utility(params, q_obs, p)
    p_rep     = jnp.tile(p, (q_samples.shape[0], 1))   # (S, L)
    u_samples = utility(params, q_samples, p_rep)        # (S,)
    return -(u_obs - jnp.mean(u_samples))               # scalar


# ---------------------------------------------------------------------------
# Batch loss functions  (vmap over the batch dimension, then mean-reduce)
#
# Shapes after vmap:
#   Q        : (B, L)
#   B_ntr    : (B, I, L)
#   Q_samples: (B, S, L)
#   P        : (B, L)
# ---------------------------------------------------------------------------


def batch_ntr_loss(
    params: Params,
    Q: jnp.ndarray,      # (B, L)
    B: jnp.ndarray,      # (B, I, L)
    P: jnp.ndarray,      # (B, L)
) -> jnp.ndarray:
    """Mean NTR loss over a batch."""
    per_sample = jax.vmap(ntr_loss, in_axes=(None, 0, 0, 0))(params, Q, B, P)  # (B,)
    return jnp.mean(per_sample)


def batch_contrastive_loss(
    params: Params,
    Q: jnp.ndarray,         # (B, L)
    Q_samples: jnp.ndarray, # (B, S, L)
    P: jnp.ndarray,         # (B, L)
) -> jnp.ndarray:
    """Mean contrastive loss over a batch."""
    per_sample = jax.vmap(contrastive_loss, in_axes=(None, 0, 0, 0))(params, Q, Q_samples, P)  # (B,)
    return jnp.mean(per_sample)


def batch_combined_loss(
    params: Params,
    Q: jnp.ndarray,         # (B, L)
    B: jnp.ndarray,         # (B, I, L)
    Q_samples: jnp.ndarray, # (B, S, L)
    P: jnp.ndarray,         # (B, L)
    w: float,
) -> jnp.ndarray:
    """Mean combined loss over a batch."""
    l_ntr  = batch_ntr_loss(params, Q, B, P)
    l_cont = batch_contrastive_loss(params, Q, Q_samples, P)
    return w * l_ntr + (1.0 - w) * l_cont


# ---------------------------------------------------------------------------
# JIT-compiled update steps
#
# optimizer is captured at construction time (not passed as a jit argument)
# to avoid non-hashable object errors.
# w is static so each distinct value gets its own compiled kernel.
# ---------------------------------------------------------------------------

def make_update_step_ntr(optimizer: optax.GradientTransformation):
    """Return a jit-compiled batch NTR update step."""
    @jax.jit
    def _step(
        params: Params,
        opt_state,
        Q: jnp.ndarray,   # (B, L)
        B: jnp.ndarray,   # (B, I, L)
        P: jnp.ndarray,   # (B, L)
    ):
        loss, grads = jax.value_and_grad(batch_ntr_loss)(params, Q, B, P)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    return _step


def make_update_step_combined(optimizer: optax.GradientTransformation):
    """Return a jit-compiled batch combined-loss update step."""
    @partial(jax.jit, static_argnames=("w",))
    def _step(
        params: Params,
        opt_state,
        Q: jnp.ndarray,          # (B, L)
        B: jnp.ndarray,          # (B, I, L)
        Q_samples: jnp.ndarray,  # (B, S, L)
        P: jnp.ndarray,          # (B, L)
        w: float,
    ):
        loss_fn = lambda p_: batch_combined_loss(p_, Q, B, Q_samples, P, w)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, loss

    return _step


# ---------------------------------------------------------------------------
# Batch MH sampling
#
# vmap mh_sample over a batch of (key, q_init) pairs.
# Each sample in the batch gets an independent PRNG key, so chains are
# statistically independent.
# ---------------------------------------------------------------------------

def batch_mh_sample(
    keys: jax.Array,      # (B, 2)  — one key per sample
    params: Params,
    Q: jnp.ndarray,       # (B, L)
    P: jnp.ndarray,       # (B, L)
    n_samples: int,
) -> jnp.ndarray:
    """
    Run one independent MH chain per sample in the batch.

    Returns:
        samples : (B, n_samples, L)
    """
    # mh_sample signature: (key, params, q_init, p, n_samples) -> (n_samples, L)
    # vmap over axis 0 of keys and Q and P; params and n_samples are shared.
    return jax.vmap(
        lambda key, q_init, p: mh_sample(key, params, q_init, p, n_samples),
        in_axes=(0, 0, 0),
    )(keys, Q, P)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    q_np: np.ndarray,
    p_np: np.ndarray,
    indics: np.ndarray,
    K: int,
    w: float,
    epochs: int     = 5,
    batch_size: int = 32,
    lr: float       = 1e-3,
    m: int          = 5,
    I: int          = 20,
    s_n: int        = 10,
    seed: int       = 0,
):
    """
    Full training loop — batch-vectorised for efficient GPU utilisation.

    Key changes vs. the per-sample version
    ---------------------------------------
    1. Inner `for idx in idx_batch` loop is eliminated.
       All data for a mini-batch is assembled on CPU (numpy) and transferred
       to the GPU in a single call, giving the GPU a large matrix to work on.

    2. `update_ntr` / `update_combined` now receive entire batches:
         Q        (B, L)
         B        (B, I, L)
         Q_samples(B, S, L)   [combined only]
         P        (B, L)
       `jax.vmap` inside the loss functions parallelises across B on the GPU.

    3. MH sampling is also batched: `batch_mh_sample` vmaps `mh_sample`
       over the batch dimension, running B independent chains in parallel.

    Args:
        q_np      : consumption bundles,  (N, L)
        p_np      : prices by week,       (T, L)
        indics    : week index per row,   (N,)
        K         : number of latent factors
        w         : loss weight (1 = NTR only, 0 = contrastive only)
        epochs    : number of training epochs
        batch_size: mini-batch size B
        lr        : Adam learning rate
        m         : max shift for random_set
        I         : NTR set size
        s_n       : MH steps per observation (burn-in included)
        seed      : JAX PRNG seed

    Returns:
        params  : trained Params
        history : dict with 'loss' and 'log_ll' lists
    """
    N, L   = q_np.shape
    rng_np = np.random.default_rng(seed)

    # ── initialise parameters and optimizer ───────────────────────────────
    key          = jax.random.PRNGKey(seed)
    key, initkey = jax.random.split(key)
    params       = init_params(initkey, K, L)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    update_ntr      = make_update_step_ntr(optimizer)
    update_combined = make_update_step_combined(optimizer)

    history = {"loss": [], "log_ll": []}

    for epoch in tqdm(range(epochs), desc=f"K={K}, w={w}"):
        total_loss = 0.0
        total_ll   = 0.0
        perm       = rng_np.permutation(N)

        for i in range(0, N, batch_size):
            idx_batch = perm[i : i + batch_size]   # may be smaller at the last batch
            B_size    = len(idx_batch)

            # ── 1. Assemble the entire mini-batch on CPU (numpy) ──────────
            Q_np = q_np[idx_batch]                          # (B, L)
            P_np = p_np[indics[idx_batch].astype(int)]      # (B, L)

            # Build NTR sets for all samples in the batch at once.
            # random_set is still a numpy operation; we stack results.
            B_np = np.stack([
                random_set(Q_np[j], m=m, I=I, rng=rng_np)
                for j in range(B_size)
            ])                                               # (B, I, L)

            # ── 2. Single CPU→GPU transfer ─────────────────────────────────
            Q_jnp = jnp.array(Q_np, dtype=jnp.float32)     # (B, L)
            P_jnp = jnp.array(P_np, dtype=jnp.float32)     # (B, L)
            B_jnp = jnp.array(B_np, dtype=jnp.float32)     # (B, I, L)

            # ── 3. GPU update step ─────────────────────────────────────────
            if w == 1.0:
                params, opt_state, loss = update_ntr(
                    params, opt_state, Q_jnp, B_jnp, P_jnp
                )
                # NTR loss == -log_ll by definition
                ll = float(-loss)

            else:
                # Split one key per sample so MH chains are independent.
                key, subkey = jax.random.split(key)
                batch_keys  = jax.random.split(subkey, B_size)  # (B, 2)

                # Run B MH chains in parallel on the GPU.
                # Shape: (B, s_n, L)
                Q_samples_jnp = batch_mh_sample(
                    batch_keys, params, Q_jnp, P_jnp, n_samples=s_n
                )

                # NTR log-likelihood for monitoring (no gradient needed).
                ll = float(-batch_ntr_loss(params, Q_jnp, B_jnp, P_jnp))

                params, opt_state, loss = update_combined(
                    params, opt_state,
                    Q_jnp, B_jnp, Q_samples_jnp, P_jnp,
                    w,
                )

            total_loss += float(loss) * B_size   # weight by actual batch size
            total_ll   += ll          * B_size

        avg_loss = total_loss / N
        avg_ll   = total_ll   / N
        history["loss"].append(avg_loss)
        history["log_ll"].append(avg_ll)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Avg Log-LL: {avg_ll:.4f}")

    return params, history


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p_df = pd.read_parquet("C:\\BASIC_LEARNING\\mls\\code\\data\\p_d\\p_d_d_f.parquet")
    q_df = pd.read_parquet("C:\\BASIC_LEARNING\\mls\\code\\data\\q_d\\q_d_d_f.parquet")

    L      = q_df.shape[1] - 2
    indics = (q_df["WEEK_NO"] - 1).reset_index(drop=True).values

    p_np = p_df.iloc[:, -L:].values.astype(np.float32)
    q_np = q_df.iloc[:, -L:].values.astype(np.float32)

    Ks = [2, 5, 10]
    ws = [0, 0.25, 0.5, 0.75, 1]

    for K in Ks:
        for w in ws:
            print(f"\n{'='*50}")
            print(f"Training  K={K}  w={w}")
            print(f"{'='*50}")

            trained_params, history = train(
                q_np=q_np, p_np=p_np, indics=indics,
                K=K, w=w,
                epochs=5, batch_size=32, lr=1e-3,
                m=5, I=20, s_n=10,
            )

            np.savez(
                f"model_K{K}_w{w}.npz",
                A=np.array(trained_params.A),
                b=np.array(trained_params.b),
                delta=np.array(trained_params.delta),
            )
            print(f"Saved model_K{K}_w{w}.npz")
