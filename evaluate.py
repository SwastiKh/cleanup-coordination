import os
import jax
import jax.numpy as jnp
import numpy as np
import imageio.v3 as iio
import wandb

from algos.mappo_ippo_basic import IPPOActor, MAPPOActor
from utils import *
from plot_dirt_fraction import *



def make_policy_fn(actor):
    def policy_fn(actor_params, obs, rnn_state, rng, deterministic):
        if USE_LSTM:
            pi, new_rnn_state = actor.apply(actor_params, obs, rnn_state)
        else:
            pi, _ = actor.apply(actor_params, obs)
            new_rnn_state = None
        if deterministic:
            return jnp.argmax(pi.logits, axis=-1), new_rnn_state
        else:
            return pi.sample(seed=rng), new_rnn_state
    return policy_fn



def evaluate_policy(
    env,
    params,
    num_steps,
    save_dir,
    seed=0,
    deterministic=False,
    log_wandb=True,
    current_step=None,
):
    """
    env        : socialjax env (cleanup / harvest)
    params     : dict with 'actor' params
    num_steps  : rollout length per episode
    save_dir   : directory to save GIF
    """
    print("[Evaluation] Starting evaluation at {} step...".format(current_step))

    num_agents = env.num_agents
    action_dim = env.action_space(0).n

    if ALGO_NAME == "MAPPO":
        actor = MAPPOActor(action_dim=action_dim, encoder_type=ENCODER.lower())
    elif ALGO_NAME == "IPPO":
        actor = IPPOActor(action_dim=action_dim, encoder_type=ENCODER.lower())

    policy_fn_jit = jax.jit(make_policy_fn(actor), static_argnames=("deterministic",))

    def scan_step(carry, _):
        obs, env_state, rng, rnn_state = carry
        rng, act_rng, step_rng = jax.random.split(rng, 3)
        obs_batch = jnp.stack(obs).reshape(num_agents, *obs[0].shape)
        actions, new_rnn_state = policy_fn_jit(
            params["actor"], obs_batch, rnn_state, act_rng, deterministic
        )
        new_obs, new_env_state, reward, done, info = env.step_env(
            step_rng, env_state, actions
        )
        if USE_LSTM:
            new_rnn_state = jax.tree_map(
                lambda x: jnp.where(done["__all__"], jnp.zeros_like(x), x),
                new_rnn_state,
            )
        return (new_obs, new_env_state, rng, new_rnn_state), (reward, info["dirtFraction"])

    # ── Metrics pass: one scan per episode, each at a different dirt fraction ──
    num_episodes = BATCH_SIZE // num_steps
    eval_dirt_fractions = np.linspace(0.0, 0.8, num_episodes)

    rng = jax.random.PRNGKey(seed)
    ep_rewards_list = []
    ep_dirt_list = []

    for ep, dirt_frac in enumerate(eval_dirt_fractions):
        rng, reset_rng, scan_rng = jax.random.split(rng, 3)
        obs, env_state = env.reset(reset_rng, float(dirt_frac), RESET_APPLE_FRACTION)
        init_rnn_state = (
            (jnp.zeros((num_agents, 128)), jnp.zeros((num_agents, 128)))
            if USE_LSTM else None
        )
        (_, _, _, _), (ep_rewards, ep_dirt) = jax.lax.scan(
            scan_step, (obs, env_state, scan_rng, init_rnn_state), None, length=num_steps
        )
        ep_rewards_list.append(ep_rewards)
        ep_dirt_list.append(ep_dirt)
        print(f"[Eval ep={ep}] initial dirtFraction: {float(ep_dirt[0]):.4f}")

    all_rewards = jnp.concatenate(ep_rewards_list, axis=0)   # (BATCH_SIZE, num_agents)
    dirt_fractions_all = np.concatenate(ep_dirt_list, axis=0) # (BATCH_SIZE,)

    episode_return = jnp.sum(all_rewards, axis=0)  # (num_agents,)

    # ── GIF pass (Python loop, one episode at full dirt, only when needed) ────
    save_gif_now = SAVE_GIF and (
        current_step == "final" or current_step % SAVE_GIF_INTERVAL == 0
    )
    if save_gif_now:
        rng = jax.random.PRNGKey(seed)
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng, RESET_DIRT_FRACTION, RESET_APPLE_FRACTION)
        if USE_LSTM:
            actor_rnn_state = (jnp.zeros((num_agents, 128)), jnp.zeros((num_agents, 128)))

        render_every = 2  # render 1 in every N steps → 500 frames at 1000 steps
        frames = [np.asarray(env.render(env_state))]
        for t in range(num_steps):
            rng, act_rng, step_rng = jax.random.split(rng, 3)
            obs_batch = jnp.stack(obs).reshape(num_agents, *obs[0].shape)
            if USE_LSTM:
                actions, actor_rnn_state = policy_fn_jit(
                    params["actor"], obs_batch, actor_rnn_state, act_rng, deterministic
                )
            else:
                actions, _ = policy_fn_jit(
                    params["actor"], obs_batch, None, act_rng, deterministic
                )
            obs, env_state, _, done, _ = env.step_env(step_rng, env_state, actions)
            if t % render_every == 0:
                frames.append(np.asarray(env.render(env_state)))
            if done["__all__"]:
                break

        out_dir = os.path.join(save_dir, str(current_step))
        os.makedirs(out_dir, exist_ok=True)

        # gif_path = os.path.join(out_dir, "evaluation.gif")
        # frames_pil = [Image.fromarray(f) for f in frames]
        # frames_pil[0].save(
        #     gif_path, save_all=True, append_images=frames_pil[1:], duration=150, loop=0
        # )
        mp4_path = os.path.join(out_dir, "evaluation.mp4")
        iio.imwrite(mp4_path, np.stack(frames), fps=14, codec="libx264")
        print(f"[Evaluation] Video saved to {mp4_path}")

        episode_boundaries = list(range(0, len(dirt_fractions_all), NUM_INNER_STEPS))
        dirt_img_path = os.path.join(out_dir, "dirt_fraction.png")
        plot_dirt_fraction(dirt_fractions_all, episode_boundaries, save_path=dirt_img_path)
        print(f"[Evaluation] Dirt fraction plot saved to {dirt_img_path}")

    print(f"[Evaluation] Episode return per agent: {episode_return}")


    if log_wandb:
        print("[Evaluation] Logging evaluation metrics to Weights & Biases...")
        per_episode_returns = jnp.stack([jnp.sum(r, axis=0) for r in ep_rewards_list])  # (num_episodes, num_agents)
        df_arr = dirt_fractions_all
        total_steps = len(df_arr)
        wandb.log({
            "eval/episode_return_mean": float(jnp.nanmean(episode_return)),
            "eval/episode_return_total": float(jnp.nansum(episode_return)),
            **{f"eval/episode_return_agent{i}": episode_return[i].item() for i in range(num_agents)},
            "eval/dirt_fraction_mean": float(np.mean(df_arr)),
            "eval/dirt_fraction_pct_high": float(np.sum(df_arr >= 0.32) / total_steps * 100),
            "eval/dirt_fraction_pct_mid": float(np.sum((df_arr >= 0.20) & (df_arr < 0.32)) / total_steps * 100),
            "eval/dirt_fraction_pct_low": float(np.sum(df_arr < 0.20) / total_steps * 100),
            "eval/dirt_fraction_hist": wandb.Histogram(df_arr),
            "eval/dirt_fraction_over_time": wandb.plot.line_series(
                xs=list(range(len(df_arr))),
                ys=[df_arr.tolist()],
                keys=["dirt_fraction"],
                title="Dirt Fraction Over Time",
                xname="timestep",
            ),
            "eval/per_episode_return_mean": float(jnp.nanmean(per_episode_returns, axis=1).mean()),
            "eval/per_episode_return_total": float(jnp.nansum(per_episode_returns, axis=1).sum()),
            #  **{f"eval/dirt_{eval_dirt_fractions[i]:.2f}/agent_{a}_return": float(per_episode_returns[i, a]) for i in range(num_episodes) for a in range(num_agents)},
        })

    return {
        "episode_return": episode_return,
    }
