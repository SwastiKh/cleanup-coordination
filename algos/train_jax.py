import jax
import jax.numpy as jnp
# from flax.training.train_state import TrainState
# import optax
from algos.mappo_ippo_basic import *
from algos.ppo_utils import compute_adv_no_gae, compute_gae, ppo_loss
from utils import *
from typing import NamedTuple
import numpy as np

class Transition(NamedTuple):
    obs: jnp.ndarray        # (N, H, W, C)
    world_state: jnp.ndarray   # (H, W, C*N)
    rewards: jnp.ndarray     # (N,)
    dones: jnp.ndarray       # (N,)
    values: jnp.ndarray      # (N,)
    actions: jnp.ndarray     # (N,)
    log_probs: jnp.ndarray   # (N,)
    additional_info: dict    # Additional info from the environment


def train_jax(
    rng,
    env,
    config,
    actor,
    critic,
    actor_state,
    critic_state,
    obs,
    env_state,
    start_step,
    num_steps, 
):

    num_agents = env.num_agents

    def env_step(carry, _):
        actor_state, critic_state, env_state, obs, rng = carry
        rng, act_rng, step_rng = jax.random.split(rng, 3)

        # Actor
        # obs_batch = obs  # (num_agents, H, W, C)
        pi, _ = actor.apply(actor_state.params, obs)
        actions = pi.sample(seed=act_rng).astype(jnp.int32)
        logp = pi.log_prob(actions)

        # jax.debug.print("actions shape = {s}", s=actions.shape)

        world_state = jnp.concatenate(obs, axis=-1)[None, ...]
        if ALGO_NAME == "MAPPO":
            # Central critic
            values = critic.apply(critic_state.params, world_state)
            values = jnp.repeat(values, obs.shape[0])
        elif ALGO_NAME == "IPPO":
            values = critic.apply(critic_state.params, obs)



        
        obs, env_state, reward, done, info = env.step(
            step_rng, env_state, actions)

        done_flag = jnp.full((obs.shape[0],), done["__all__"])

        # transition = (obs, actions, logp, values, reward, done_flag)
        # transition = (obs, actions, values, reward, done_flag)
        
        transition = Transition(
            obs=obs,
            world_state=world_state,
            rewards=reward,
            dones=done_flag,
            values=values,
            actions=actions,
            log_probs=logp,
            additional_info=info,
        )
        # transition = Transition(
        #     obs=jnp.stack(obs),  # (T, N, H, W, C)
        #     world_state=jnp.stack(world_state),  # (T, H, W, C*N)
        #     rewards=jnp.stack(reward),  # (T, N)
        #     dones=jnp.stack(done_flag),  # (T, N)
        #     values=jnp.stack(values),  # (T, N)
        #     actions=jnp.stack(actions),  # (T, N)
        #     log_probs=jnp.stack(logp),  # (T, N)
        #     additional_info=info  # list of dicts
        # )
        #     transition = Transition(
        #         obs=[],
        #         world_state=[],
        #         rewards=[],
        #         dones=[],
        #         values=[],
        #         actions=[],
        #         log_probs=[],
        #         additional_info=[],
        #     )
        #     transition.obs.append(obs)
        #     transition.world_state.append(world_state)
        #     transition.rewards.append(reward)
        #     transition.dones.append(done_flag)
        #     transition.values.append(values)
        #     transition.actions.append(actions)
        #     transition.log_probs.append(logp)
        #     transition.additional_info.append(info)

        
        # jax.debug.print(
        #     "obs shape: {s}, actions shape: {a}, rewards shape: {r}, dones shape: {d}, values shape: {v}, log_probs shape: {l}, transition obs shape: {ts}",
        #     s=obs.shape,
        #     a=actions.shape,
        #     r=reward.shape,
        #     d=done_flag.shape,
        #     v=values.shape,
        #     l=logp.shape,
        #     ts=transition.obs.shape,
        # )

        # breakpoint()
        return (actor_state, critic_state, env_state, obs, rng), transition


    def collect_rollout(carry):
        # return jax.lax.scan(env_step, carry, None, config["NUM_INNER_STEPS"])
        return jax.lax.scan(env_step, carry, None, config["EVAL_INTERVAL"])

    def ppo_update(actor_state, critic_state, traj, adv, targets, config):
        obs = traj.obs                     # (T, N, H, W, C)
        world_state = traj.world_state     # (T, H, W, C*N)
        actions = traj.actions             # (T, N)
        logp_old = traj.log_probs          # (T, N)

        # jax.debug.print(
        #     "Starting PPO update. obs shape: {s}, world_state shape: {ws}, actions shape: {a}, logp_old shape: {l}",
        #     s=obs.shape,
        #     ws=world_state.shape,
        #     a=actions.shape,
        #     l=logp_old.shape,
        # )

        print("Starting PPO update. obs shape: {}, world_state shape: {}, actions shape: {}, logp_old shape: {}".format(
            obs.shape, world_state.shape, actions.shape, logp_old.shape))

        T, N = actions.shape

        # ---------- flatten per-agent tensors ----------
        obs_flat = obs.reshape((T * N,) + obs.shape[2:])
        actions_flat = actions.reshape((T * N,))
        logp_old_flat = logp_old.reshape((T * N,))
        adv_flat = adv.reshape((T * N,))
        targets_flat = targets.reshape((T * N,))

        def loss_fn(actor_params, critic_params):
            # ---------- Actor ----------
            pi, _ = actor.apply(actor_params, obs_flat)
            logp = pi.log_prob(actions_flat)

            ratio = jnp.exp(logp - logp_old_flat)
            # adv_n = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
            # adv_std = jnp.maximum(adv_flat.std(), 1e-4) #1e-8, 0.05
            # adv_n = (adv_flat - adv_flat.mean()) / (adv_std)

            policy_loss = -jnp.mean(
                jnp.minimum(
                    ratio * adv_flat,
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    ) * adv_flat
                )
            )

            # ---------- Critic ----------
            if ALGO_NAME == "MAPPO":
                # Central critic
                # world_state: (T, H, W, C*N)
                values_t = critic.apply(
                    critic_params,
                    world_state.reshape((T,) + world_state.shape[1:])
                )  # (T,)

                # Repeat for each agent
                # values_flat = jnp.repeat(values_t[:, None], N, axis=1)  # (T, N)
                targets_t = targets.reshape(T, N).mean(axis=1)  # (T,)
                value_loss = jnp.mean((values_t - targets_t) ** 2)

            elif ALGO_NAME == "IPPO":
                values_t = critic.apply(
                    critic_params,
                    obs_flat
                )  # (T*N,)

                value_loss = jnp.mean((values_t - targets_flat) ** 2)
            # value_loss = jnp.mean((values_flat - targets_flat) ** 2)
            # make no. of agents invariant in value loss



            entropy = pi.entropy().mean()
            # calculated later during logging 
            # total_loss = (
            #     policy_loss
            #     + config["VF_COEF"] * value_loss
            #     - ent_coef_log * entropy
            # )

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                # "total_loss": total_loss,
                
            }
            return _, metrics

        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(actor_state.params, critic_state.params)

        actor_state = actor_state.apply_gradients(grads=grads[0])
        critic_state = critic_state.apply_gradients(grads=grads[1])

        return actor_state, critic_state, metrics


    def update_step(carry, update_idx):
        actor_state, critic_state, env_state, obs, rng, step = carry

        if ALGO_NAME == "MAPPO":
            world_state = jnp.concatenate(obs, axis=-1)[None, ...]  # (1, H, W, C*N)
            # Central critic
            last_values = critic.apply(critic_state.params, world_state)
            last_values = jnp.repeat(last_values, obs.shape[0])     # (N,)
        elif ALGO_NAME == "IPPO":
            last_values = critic.apply(critic_state.params, obs)


        (actor_state, critic_state, env_state, obs, rng), traj = collect_rollout(
            (actor_state, critic_state, env_state, obs, rng)
        )

        # advantages, targets = compute_gae(traj, config)
        # advantages, targets = compute_gae(traj, last_values, config)
        advantages, targets = compute_adv_no_gae(traj, last_values, config)
        # jax.debug.print(
        #     "targets stats: min={mi}, max={ma}, mean={m}",
        #     mi=targets.min(),
        #     ma=targets.max(),
        #     m=targets.mean(),
        # )

        ent_coef_log = jnp.interp(
            step,
            np.array([0, config["NUM_OUTER_STEPS"]]),
            np.array([config["ENT_COEF_START"], config["ENT_COEF_END"]])
        )

        actor_state, critic_state, metrics = ppo_update(
            actor_state, critic_state, traj, advantages, targets, config
        )


        step = step + 1

        # traj.rewards shape: (T, N)
        episode_returns = jnp.sum(traj.rewards, axis=0)   # (N,)
        traj_additional_info = traj.additional_info
        # print("additional_info keys: ", additional_info.keys())
        # additional_info keys:  dict_keys(['clean_action_info', 'cleaned_water', 'original_rewards', 'shaped_rewards', 'shared_cleaning_reward', 'total_apples_collected', 'total_successful_cleans'])
        # if SHARED_CLEANING_REWARDS==True:
        additional_info_wandb = {
            "total_successful_cleans": jnp.sum(traj_additional_info["total_successful_cleans"]),
            "total_apples_collected": jnp.sum(traj_additional_info["total_apples_collected"]),
        }
        # print("additional_info sample: ", {k: v[0] for k, v in additional_info.items()})
        mean_episode_return = jnp.nanmean(episode_returns)

        # to add to payload per episode:
        # - number of successful cleans
        # - number of apples collected
        # - per agent returns?

        log_payload = {
            "policy_loss": metrics["policy_loss"],
            "value_loss": metrics["value_loss"],
            "entropy": metrics["entropy"],
            "total_loss": (
                metrics["policy_loss"]
                + config["VF_COEF"] * metrics["value_loss"]
                - ent_coef_log * metrics["entropy"]
            ),
            "mean_episode_return": mean_episode_return,
            **{f"agent_{i}_episode_return": episode_returns[i] for i in range(num_agents)},
            "total_episode_return": jnp.sum(episode_returns),
            **{k: v for k, v in additional_info_wandb.items()}
        }

        # if log_wandb:
        # jax.debug.callback(wandb_log_callback, metrics)
        def wandb_log_callback_safe(payload):
            payload = jax.tree_map(jax.device_get, payload)
            payload = dict(payload)

            wandb.log(payload)
        
        if LOG_WANDB:
            jax.debug.callback(wandb_log_callback_safe, log_payload)


        return (actor_state, critic_state, env_state, obs, rng, step), metrics
    




    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)

    # carry = (actor_state, critic_state, env_state, obs, rng, step)
    # carry0 = (actor_state, critic_state, env_state, obs, rng, 0)


    # (final_actor_state, final_critic_state, env_state, obs, rng, step), metrics = jax.lax.scan(
    #     update_step,
    #     (actor_state, critic_state, env_state, obs, rng, 0),
    #     None,
    #     config["NUM_OUTER_STEPS"]
    # )
    (final_actor_state, final_critic_state, env_state, obs, rng, step), metrics = jax.lax.scan(
        update_step,
        (actor_state, critic_state, env_state, obs, rng, 0),
        None,
        num_steps
    )

    # save_checkpoint(
    #     {"actor": final_actor_state.params, "critic": final_critic_state.params},
    #     SAVE_DIR,
    #     step="final",
    # )



    return {
    "params": {
        "actor": final_actor_state.params,
        "critic": final_critic_state.params,
    },
    "metrics": metrics,
    "actor_state": final_actor_state,
    "critic_state": final_critic_state,
    "env_state": env_state,
    "obs": obs,
    "rng": rng,
    "step": step,

}


