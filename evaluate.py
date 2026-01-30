import os
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import wandb

from algos.mappo_ippo_basic import MAPPOActor
from utils import *



def make_policy_fn(actor):
    def policy_fn(actor_params, obs, rng, deterministic):
        pi, _ = actor.apply(actor_params, obs)
        if deterministic:
            return jnp.argmax(pi.logits, axis=-1)
        else:
            return pi.sample(seed=rng)
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
    num_steps  : rollout length
    save_dir   : directory to save GIF
    """
    print("[Evaluation] Starting evaluation at {} step...".format(current_step))

    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)

    obs, env_state = env.reset(reset_rng)

    num_agents = env.num_agents
    action_dim = env.action_space(0).n
    actor = MAPPOActor(
            action_dim=action_dim,
            encoder_type=ENCODER.lower()
        )
    # print("Actor initialized for evaluation at step {}.".format(current_step))


    policy_fn_jit = jax.jit(make_policy_fn(actor), static_argnames=("deterministic",),)

    
    frames = []

    img = env.render(env_state)
    frames.append(np.asarray(img))

    episode_return = jnp.zeros((num_agents,))

    # print("just before eval loop at step {}.".format(current_step))
    for t in range(num_steps):
        rng, act_rng, step_rng = jax.random.split(rng, 3)

        # obs: (num_agents, H, W, C)
        obs_batch = jnp.stack(obs).reshape(
            num_agents, *obs[0].shape
        )
        # CHANGE TO JAX
        # # print("Eval step {}, obs_batch shape: {}".format(t, obs_batch.shape))
        # pi, _ = actor.apply(params["actor"], obs_batch)
        # # print("Eval step {}, pi logits shape: {}".format(t, pi.logits.shape))

        # if deterministic:
        #     actions = jnp.argmax(pi.logits, axis=-1)
        # else:
        #     actions = pi.sample(seed=act_rng)

        actions = policy_fn_jit(
            params["actor"],
            obs_batch,
            act_rng,
            deterministic,
        )

        # print("Eval step {}, actions: {}".format(t, actions))
        obs, env_state, reward, done, info = env.step_env(
            step_rng, env_state, list(actions)
        )
        # print("Eval step {}, reward: {}, done: {}".format(t, reward, done))

        episode_return += jnp.array(reward)

        img = env.render(env_state)
        frames.append(np.asarray(img))

        if done["__all__"]:
            break

    gif_path = os.path.join(save_dir, str(current_step), "evaluation.gif")
    if SAVE_GIF and current_step % SAVE_GIF_INTERVAL == 0:
        os.makedirs(os.path.join(save_dir, str(current_step)), exist_ok=True)
        # Save GIF
        frames_pil = [Image.fromarray(f) for f in frames]
        frames_pil[0].save(
            gif_path,
            save_all=True,
            append_images=frames_pil[1:],
            duration=150,
            loop=0,
        )
        print(f"[Evaluation] GIF saved to {gif_path}")


    print(f"[Evaluation] Episode return per agent: {episode_return}")

    if log_wandb:
        wandb.log({
            "eval/episode_return_mean": episode_return.mean().item(),
            # "eval/episode_return_agent0": episode_return[0].item(),
            **{f"agent_{i}_episode_return": episode_return[i].item() for i in range(num_agents)}
            # "eval/gif": wandb.Video(gif_path, format="gif"),
        })

    return {
        "episode_return": episode_return,
        "gif_path": gif_path,
    }
