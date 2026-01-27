import os
import pickle
import jax
import jax.numpy as jnp


def save_checkpoint(params, save_dir, step=None):
    """
    params: dict with 'actor' and 'critic' params (pytrees)
    """
    os.makedirs(save_dir, exist_ok=True)

    if step is None:
        fname = "checkpoint.pkl"
    else:
        fname = f"checkpoint_step_{step}.pkl"

    path = os.path.join(save_dir, fname)

    # convert to numpy for portability
    params_np = jax.tree_map(lambda x: jnp.array(x), params)

    with open(path, "wb") as f:
        pickle.dump(params_np, f)

    print(f"[Checkpoint] Saved to {path}")


def load_checkpoint(path):
    with open(path, "rb") as f:
        params = pickle.load(f)

    return jax.tree_map(lambda x: jnp.array(x), params)
