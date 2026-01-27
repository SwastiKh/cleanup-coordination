import jax
import jax.numpy as jnp


def compute_gae(traj, last_values, config):
    """
    traj: Transition with fields stacked over time
      - traj.rewards: (T, N)
      - traj.values:  (T, N)
      - traj.dones:   (T, N)
    last_values: (N,)
    """
    rewards = traj.rewards
    values = traj.values
    dones = traj.dones

    gamma = config["GAMMA"]
    lam = config["GAE_LAMBDA"]

    T, N = rewards.shape

    def gae_step(carry, t):
        gae, next_value = carry
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        return (gae, values[t]), gae

    (_, _), advantages = jax.lax.scan(
        gae_step,
        (jnp.zeros(N), last_values),
        jnp.arange(T),
        reverse=True,
    )

    targets = advantages + values
    return advantages, targets



def ppo_loss(
    pi,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    clip_eps: float,
):
    """
    Compute PPO clipped surrogate loss.

    pi:           distribution object
    actions:      (batch_size, action_dim)
    old_log_probs:(batch_size,)
    advantages:   (batch_size,)
    clip_eps:     float
    """

    log_probs = pi.log_prob(actions)  # (batch_size,)
    ratio = jnp.exp(log_probs - old_log_probs)  # (batch_size,)

    # Normalize advantages
    adv_mean = jnp.mean(advantages)
    adv_std = jnp.std(advantages) + 1e-8
    norm_advantages = (advantages - adv_mean) / adv_std

    unclipped_obj = ratio * norm_advantages
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped_obj = clipped_ratio * norm_advantages

    loss = -jnp.mean(jnp.minimum(unclipped_obj, clipped_obj))
    return loss