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
    next_obs: jnp.ndarray   # (N, H, W, C)
    world_state: jnp.ndarray   # (H, W, C*N)
    rewards: jnp.ndarray     # (N,)
    dones: jnp.ndarray       # (N,)
    values: jnp.ndarray      # (N,)
    actions: jnp.ndarray     # (N,)
    log_probs: jnp.ndarray   # (N,)
    additional_info: dict    # Additional info from the environment


def wandb_log_callback_safe(payload, current_step):
    payload = jax.tree_map(jax.device_get, payload)
    payload = dict(payload)

    wandb.log(payload, step=int(current_step))




def train_jax(
    rng,
    env,
    config,
    actor,
    critic,
    actor_state,
    critic_state,
    # obs,
    # env_state,
    step,
    num_steps, 
):

    """
    Main training loop 
    """
    print("Starting training...")

    # def collect_rollout(carry):
    #     # return jax.lax.scan(env_step, carry, None, config["NUM_INNER_STEPS"])
    #     return jax.lax.scan(env_step, carry, None, config["EVAL_INTERVAL"])

    # ENV STEP FUNCTION

    def env_step(carry, _):
        actor_state, critic_state, env_state, obs, rng, step = carry
        rng, act_rng, step_rng, reset_rng = jax.random.split(rng, 4)

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



        
        next_obs, env_state, reward, done, info = env.step(
            step_rng, env_state, actions)

        done_flag = jnp.full((obs.shape[0],), done["__all__"])

        # jax.debug.print(
        #     "done flag shape: {s}, done flag sample: {d}",
        #     s=done_flag.shape,
        #     d=done_flag,
        # )
        # jax.debug.breakpoint()

        # transition = (obs, actions, logp, values, reward, done_flag)
        # transition = (obs, actions, values, reward, done_flag)
        
        # transition = Transition(
        #     obs=obs,
        #     world_state=world_state,
        #     rewards=reward,
        #     dones=done_flag,
        #     values=values,
        #     actions=actions,
        #     log_probs=logp,
        #     additional_info=info,
        # )
        # jax.debug.breakpoint()
        # jax.debug.print(
        #     "if condition on step{s} = {c}",
        #     s=step,
        #     c=(step % config["NUM_INNER_STEPS"] == 0)
        # )

        operand = (env_state, reset_rng, obs, next_obs, world_state, reward, done_flag, values, actions, logp, info)
        def if_done(operand):
            current_env_state, rng, obs, next_obs, world_state, reward, done_flag, values, actions, logp, info = operand
            done_flag = jnp.ones((obs.shape[0],), dtype=bool)
            transition = Transition(
                obs=jnp.stack(obs),  # (T, N, H, W, C)
                next_obs=jnp.stack(next_obs),  # (T, N, H, W, C)
                world_state=jnp.stack(world_state),  # (T, H, W, C*N)
                rewards=jnp.stack(reward),  # (T, N)
                dones=jnp.stack(done_flag),  # (T, N)
                values=jnp.stack(values),  # (T, N)
                actions=jnp.stack(actions),  # (T, N)
                log_probs=jnp.stack(logp),  # (T, N)
                additional_info=info  # list of dicts
            )
            new_obs, new_env_state = env.reset(rng)
            return transition, new_obs, new_env_state

        def not_done(operand):
            current_env_state, rng, obs, next_obs, world_state, reward, done_flag, values, actions, logp, info = operand
            done_flag = jnp.full((obs.shape[0],), done["__all__"])

            transition = Transition(
                obs=jnp.stack(obs),  # (T, N, H, W, C)
                next_obs=jnp.stack(next_obs),  # (T, N, H, W, C)
                world_state=jnp.stack(world_state),  # (T, H, W, C*N)
                rewards=jnp.stack(reward),  # (T, N)
                dones=jnp.stack(done_flag),  # (T, N)
                values=jnp.stack(values),  # (T, N)
                actions=jnp.stack(actions),  # (T, N)
                log_probs=jnp.stack(logp),  # (T, N)
                additional_info=info  # list of dicts
            )

            obs = next_obs.copy()
            return transition, obs, current_env_state
            # return transition, obs

        condition = (step % config["NUM_INNER_STEPS"] == 0)
        transition, obs, env_state = jax.lax.cond(condition, if_done, not_done, operand)


        # if step % config["NUM_INNER_STEPS"] == 0:
        #     done_flag = jnp.ones((obs.shape[0],), dtype=bool)
        #     transition = Transition(
        #         obs=jnp.stack(obs),  # (T, N, H, W, C)
        #         next_obs=jnp.stack(next_obs),  # (T, N, H, W, C)
        #         world_state=jnp.stack(world_state),  # (T, H, W, C*N)
        #         rewards=jnp.stack(reward),  # (T, N)
        #         dones=jnp.stack(done_flag),  # (T, N)
        #         values=jnp.stack(values),  # (T, N)
        #         actions=jnp.stack(actions),  # (T, N)
        #         log_probs=jnp.stack(logp),  # (T, N)
        #         additional_info=info  # list of dicts
        #     )
        #     obs, env_state = env.reset(reset_rng)
        # else:
        #     transition = Transition(
        #         obs=jnp.stack(obs),  # (T, N, H, W, C)
        #         next_obs=jnp.stack(next_obs),  # (T, N, H, W, C)
        #         world_state=jnp.stack(world_state),  # (T, H, W, C*N)
        #         rewards=jnp.stack(reward),  # (T, N)
        #         dones=jnp.stack(done_flag),  # (T, N)
        #         values=jnp.stack(values),  # (T, N)
        #         actions=jnp.stack(actions),  # (T, N)
        #         log_probs=jnp.stack(logp),  # (T, N)
        #         additional_info=info  # list of dicts
        #     )

        #     obs = next_obs.copy()

        step+=1


        # if done_flag[0]:  # Assuming all agents are done at the same time
        #     rng, reset_rng = jax.random.split(rng)
        #     print("random reset rng in env_step: ", reset_rng)
        #     obs, env_state = env.reset(reset_rng)


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
        return (actor_state, critic_state, env_state, obs, rng, step), transition

    
    rng, reset_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng)

    # (final_actor_state, final_critic_state, env_state, terminal_obs, rng, step), trajectory_rollout = jax.lax.scan(
    #     update_step,
    #     (actor_state, critic_state, env_state, obs, rng, 0),
    #     None,
    #     num_steps//config["NUM_INNER_STEPS"],
    # )
    (final_actor_state, final_critic_state, env_state, terminal_obs, rng, next_step), trajectory_rollout = jax.lax.scan(
        env_step, (actor_state, critic_state, env_state, obs, rng, step), None, config["EVAL_INTERVAL"]
    )

    # jax.debug.breakpoint()



    if ALGO_NAME == "MAPPO":
        world_state = jnp.concatenate(terminal_obs, axis=-1)[None, ...]  # (1, H, W, C*N)
        # Central critic
        last_values = critic.apply(critic_state.params, world_state)
        last_values = jnp.repeat(last_values, terminal_obs.shape[0])     # (N,)
    elif ALGO_NAME == "IPPO":
        last_values = critic.apply(critic_state.params, terminal_obs)


    # advantages, targets = compute_gae(traj, config)
    advantages, targets = compute_gae(trajectory_rollout, last_values, config)
    # advantages, targets = compute_adv_no_gae(trajectory_rollout, last_values, config)
    # adv_std = jnp.maximum(advantages.std(), 1e-4) #1e-8, 0.05
    normalized_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    randomized_indices = jax.random.permutation(rng, trajectory_rollout.rewards.shape[0])
    trajectory_rollout = trajectory_rollout._replace(
        obs=trajectory_rollout.obs[randomized_indices],
        world_state=trajectory_rollout.world_state[randomized_indices],
        rewards=trajectory_rollout.rewards[randomized_indices],
        dones=trajectory_rollout.dones[randomized_indices],
        values=trajectory_rollout.values[randomized_indices],
        actions=trajectory_rollout.actions[randomized_indices],
        log_probs=trajectory_rollout.log_probs[randomized_indices],
    )
    normalized_advantages = normalized_advantages[randomized_indices]
    targets = targets[randomized_indices]
    # print("traj obs shape after shuffling:", trajectory_rollout.obs.shape)
    # print("traj world_state shape after shuffling:", trajectory_rollout.world_state.shape)
    # print("traj rewards shape after shuffling:", trajectory_rollout.rewards.shape)
    # print("traj dones shape after shuffling:", trajectory_rollout.dones.shape)
    # print("traj values shape after shuffling:", trajectory_rollout.values.shape)
    # print("traj actions shape after shuffling:", trajectory_rollout.actions.shape)
    # print("traj log_probs shape after shuffling:", trajectory_rollout.log_probs.shape)
    # print("advantages shape after shuffling:", normalized_advantages.shape)
    # print("targets shape after shuffling:", targets.shape)
    # jax.debug.breakpoint()
    # TODO: DONE
    # now I can pick 256 each and update ppo for NUM_EPOCHS epochs
    # 2 loops here for NUM_EPOCHS and MINIBATCH_SIZE (5120/256 = 20 minibatches per epoch)
    # put randomized indices in the epoch loop so a different sample is updated every epoch

    num_minibatches = config["EVAL_INTERVAL"] // config["MINIBATCH_SIZE"]
    batch_size = config["MINIBATCH_SIZE"]

    def reshape_batch(x):
        if hasattr(x, "shape"): # Only reshape JAX arrays
            return x.reshape((num_minibatches, batch_size) + x.shape[1:])
        return x
    
    batched_rollout = jax.tree_map(reshape_batch, trajectory_rollout)
    batched_advantages = reshape_batch(normalized_advantages)
    batched_targets = reshape_batch(targets)

    def ppo_update(actor_state, critic_state, batch_rollout, batch_adv, batch_targets, config):
        obs = batch_rollout.obs
        world_state = batch_rollout.world_state
        actions = batch_rollout.actions
        logp_old = batch_rollout.log_probs

        T, N = actions.shape

        obs_flat = obs.reshape((T * N,) + obs.shape[2:])
        actions_flat = actions.reshape((T * N,))
        logp_old_flat = logp_old.reshape((T * N,))
        adv_flat = batch_adv.reshape((T * N,))
        targets_flat = batch_targets.reshape((T * N,))
        targets_global = batch_targets.mean(axis=1)
        world_state_flat = world_state.squeeze(axis=1)

        def loss_fn(actor_params, critic_params):
            pi, _ = actor.apply(actor_params, obs_flat)
            logp = pi.log_prob(actions_flat)
            ratio = jnp.exp(logp - logp_old_flat)

            # PPO loss 
            loss_actor_1 = ratio * adv_flat
            loss_actor_2 = jnp.clip(
                ratio,
                1.0 - config["CLIP_EPS"],
                1.0 + config["CLIP_EPS"]
            ) * adv_flat
            
            policy_loss = -jnp.mean(jnp.minimum(loss_actor_1, loss_actor_2))

            if ALGO_NAME == "MAPPO":
                values = critic.apply(critic_params, world_state_flat)
                # values = jnp.repeat(values, obs.shape[1])
                # targets_t = targets_flat.reshape(T, N).mean(axis=1)
                values_t = values.squeeze() 
                value_loss = jnp.mean((values_t - targets_global) ** 2)
            elif ALGO_NAME == "IPPO":
                values = critic.apply(critic_params, obs_flat)
                values_t = values.squeeze() 
                value_loss = jnp.mean((values_t - targets_flat) ** 2)

            # values_t = values.squeeze() 
            # value_loss = jnp.mean((values_t - targets_flat) ** 2)

            # Total loss
            entropy = jnp.mean(pi.entropy())

            total_loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "total_loss": total_loss,
            }

            return total_loss, metrics
        
        (loss, metrics), grads = jax.value_and_grad(
            loss_fn, has_aux=True, argnums=(0, 1)
        )(actor_state.params, critic_state.params)

        # Apply updates
        actor_state = actor_state.apply_gradients(grads=grads[0])
        critic_state = critic_state.apply_gradients(grads=grads[1])

        return actor_state, critic_state, metrics

            # policy_loss = ppo_loss(pi, actions_flat, logp_old_flat, adv_flat, config["CLIP_EPS"])
            # value_loss = jnp.mean((values - targets_flat) ** 2)
            # entropy = jnp.mean(pi.entropy())

            # total_loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF_START"] * entropy

            # return total_loss, (policy_loss, value_loss, entropy)


    
    def update_epoch(carry, _):
        # Unpack the state
        actor_state, critic_state = carry

        def update_minibatch(inner_carry, batch_data):
            curr_actor, curr_critic = inner_carry
            b_rollout, b_adv, b_targets = batch_data
            
            # Call your existing PPO update function
            new_actor, new_critic, metrics = ppo_update(
                curr_actor, 
                curr_critic, 
                b_rollout, 
                b_adv, 
                b_targets, 
                config
            )

            # jax.debug.print(
            #     "Updated minibatch with policy_loss={pl}, value_loss={vl}, entropy={ent}, total_loss={tl}",
            #     pl=metrics["policy_loss"],
            #     vl=metrics["value_loss"],
            #     ent=metrics["entropy"],
            #     tl=metrics["total_loss"],
            # )
            return (new_actor, new_critic), metrics
        
        (next_actor, next_critic), epoch_metrics = jax.lax.scan(
            update_minibatch,
            (actor_state, critic_state),          # Init Inner Carry
            (batched_rollout, batched_advantages, batched_targets) # Scanned Data
        )

        jax.debug.print(
            "Completed epoch with policy_loss={pl}, value_loss={vl}, entropy={ent}, total_loss={tl}",
            pl=epoch_metrics["policy_loss"][-1],
            vl=epoch_metrics["value_loss"][-1],
            ent=epoch_metrics["entropy"][-1],
            tl=epoch_metrics["total_loss"][-1],
        )



        
        return (next_actor, next_critic), epoch_metrics


    init_state = (actor_state, critic_state)

    (final_actor_state, final_critic_state), all_metrics = jax.lax.scan(
        update_epoch,
        init_state,
        None, # We don't need varying inputs for epochs, we reuse the data inside
        length=config["NUM_EPOCHS"]
    )


    # for i in range(config["NUM_EPOCHS"]):
    #     for j in range(config["EVAL_INTERVAL"]//config["MINIBATCH_SIZE"]):

    #         actor_state, critic_state, metrics = ppo_update(
    #             actor_state, critic_state, trajectory_rollout[j*config["MINIBATCH_SIZE"]:(j+1)*config["MINIBATCH_SIZE"]], normalized_advantages[j*config["MINIBATCH_SIZE"]:(j+1)*config["MINIBATCH_SIZE"]], targets[j*config["MINIBATCH_SIZE"]:(j+1)*config["MINIBATCH_SIZE"]], config
    #         )



    # traj.rewards shape: (T, N)
    episode_returns = jnp.sum(trajectory_rollout.rewards, axis=0)   # (N,)
    mean_episode_return = jnp.nanmean(episode_returns)
    total_episode_return = jnp.nansum(episode_returns)
    traj_additional_info = trajectory_rollout.additional_info
    # print("additional_info keys: ", additional_info.keys())
    # additional_info keys:  dict_keys(['clean_action_info', 'cleaned_water', 'original_rewards', 'shaped_rewards', 'shared_cleaning_reward', 'total_apples_collected', 'total_successful_cleans'])
    # if SHARED_CLEANING_REWARDS==True:
    additional_info_wandb = {
        "total_successful_cleans": jnp.sum(traj_additional_info["total_successful_cleans"]),
        "total_apples_collected": jnp.sum(traj_additional_info["total_apples_collected"]),
    }
    # print("additional_info sample: ", {k: v[0] for k, v in additional_info.items()})

    # to add to payload per episode:
    # - number of successful cleans
    # - number of apples collected
    # - per agent returns?

    # ent_coef_log = jnp.interp(
    #     step,
    #     np.array([0, config["NUM_OUTER_STEPS"]]),
    #     np.array([config["ENT_COEF_START"], config["ENT_COEF_END"]])
    # )

    log_payload = {
        "train/mean_policy_loss": all_metrics["policy_loss"].mean(),
        "train/mean_value_loss": all_metrics["value_loss"].mean(),
        "train/mean_entropy": all_metrics["entropy"].mean(),
        "train/mean_total_loss": all_metrics["total_loss"].mean(),
        "train/mean_episode_return": mean_episode_return,
        "train/total_episode_return": total_episode_return,
        **{f"train/agent_{i}_episode_return": episode_returns[i] for i in range(NUM_AGENTS)},
        "train/total_episode_return": jnp.sum(episode_returns),
        **{f"train/{k}": v for k, v in additional_info_wandb.items()}
    }

    # if log_wandb:
    # jax.debug.callback(wandb_log_callback, metrics)
    # def wandb_log_callback_safe(payload):
    #     payload = jax.tree_map(jax.device_get, payload)
    #     payload = dict(payload)

    #     wandb.log(payload)
    
    if LOG_WANDB:
        jax.debug.callback(wandb_log_callback_safe, log_payload, current_step=next_step)

    

    return {
    "params": {
        "actor": final_actor_state.params,
        "critic": final_critic_state.params,
    },
    "metrics": all_metrics,
    "actor_state": final_actor_state,
    "critic_state": final_critic_state,
    "env_state": env_state,
    "obs": obs,
    "rng": rng,
    "step": next_step,

}
