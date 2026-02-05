import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import optax
from algos.mappo_ippo_basic import *
# from algos.ippo_cnn_cleanup import *
# from algos.mappo_cnn_cleanup import *
from algos.ppo_utils import compute_gae, ppo_loss
from utils import *
# from checkpoint import save_checkpoint, load_checkpoint
from typing import NamedTuple
import numpy as np
from PIL import Image
import os
from evaluate import evaluate_policy
from socialjax import make

class Transition(NamedTuple):
    obs: jnp.ndarray        # (N, H, W, C)
    world_state: jnp.ndarray   # (H, W, C*N)
    rewards: jnp.ndarray     # (N,)
    dones: jnp.ndarray       # (N,)
    values: jnp.ndarray      # (N,)
    actions: jnp.ndarray     # (N,)
    log_probs: jnp.ndarray   # (N,)
    additional_info: dict    # Additional info from the environment


# def eval_callback(actor_params, critic_params, step):
#     step = int(step)

#     eval_dir = os.path.join(SAVE_DIR, f"eval_step_{step}")

#     # IMPORTANT: create a fresh eval env
#     eval_env = make(
#         'clean_up',
#         num_inner_steps=NUM_INNER_STEPS,
#         num_outer_steps=NUM_OUTER_STEPS,
#         num_agents=NUM_AGENTS,
#         maxAppleGrowthRate=MAX_APPLE_GROWTH_RATE,
#         thresholdDepletion=THRESHOLD_DEPLETION,
#         thresholdRestoration=THRESHOLD_RESTORATION,
#         dirtSpawnProbability=DIRT_SPAWN_PROBABILITY,
#         delayStartOfDirtSpawning=DELAY_START_OF_DIRT_SPAWNING,
#         shared_rewards=SHARED_REWARDS,
#         shared_cleaning_rewards=SHARED_CLEANING_REWARDS,
#         inequity_aversion=INEQUITY_AVERSION,
#         inequity_aversion_target_agents=INEQUITY_AVERSION_TARGET_AGENTS
#     )

#     evaluate_policy(
#         env=eval_env,
#         params={
#             "actor": actor_params,
#             "critic": critic_params,
#         },
#         num_steps=NUM_INNER_STEPS,
#         save_dir=eval_dir,
#         deterministic=True,
#         log_wandb=True,
#     )



# def train_mappo_jax(rng, env, config, algo="MAPPO"):
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
    # action_dim = env.action_space(0).n
    # print(f"Action dim in train_jax: {action_dim}")  #9
    # obs_shape = env.observation_space()[0].shape
    # print(f"Obs shape in train_jax: {obs_shape}")  #(11, 11, 19)

    # if ALGO_NAME == "MAPPO":
    #     actor = MAPPOActor(action_dim=action_dim, encoder_type=ENCODER.lower())
    #     critic = MAPPOCritic(encoder_type=ENCODER.lower())
    # elif ALGO_NAME == "IPPO":
    #     actor = IPPOActor(action_dim=action_dim, encoder_type=ENCODER.lower())
    #     critic = IPPOCritic(encoder_type=ENCODER.lower())

    # rng, a_rng, c_rng = jax.random.split(rng, 3)

    # dummy_obs = jnp.zeros((1,) + obs_shape)

    # # initialize actor params - CNN weights, Dense(64) weights, Action logits weights
    # actor_params = actor.init(a_rng, dummy_obs) 

    # dummy_world = jnp.zeros((1,) + obs_shape[:-1] + (obs_shape[-1] * num_agents,))
    # if ALGO_NAME == "MAPPO":
    #     critic_params = critic.init(c_rng, dummy_world)
    # elif ALGO_NAME == "IPPO":
    #     critic_params = critic.init(c_rng, dummy_obs)

    # print("Initialized actor and critic parameters.")
    # print(f"Actor params keys: {list(actor_params.keys())}")
    # print(f"Critic params keys: {list(critic_params.keys())}")
    # print(f"Actor params: {jax.tree_map(lambda x: x.shape, actor_params)}")
    # print(f"Critic params: {jax.tree_map(lambda x: x.shape, critic_params)}")
    # print(f"Actor params values: {list(actor_params.values())}")

    # actor_state = TrainState.create(
    #     apply_fn=actor.apply,
    #     params=actor_params,
    #     # tx=optax.adam(LEARNING_RATE) #can change to adamw because relu unavailable on optax
    #     tx=optax.adam(ACTOR_LR)
    # )

    # critic_state = TrainState.create(
    #     apply_fn=critic.apply,
    #     params=critic_params,
    #     # tx=optax.adam(LEARNING_RATE) #can change to adamw because relu unavailable on optax
    #     tx=optax.adam(CRITIC_LR)
    # )

    # print("Created TrainState for actor and critic.")
    # print(f"Actor apply fn: {actor_state.apply_fn}")
    # print(f"Critic apply fn: {critic_state.apply_fn}")

    # def save_gif_callback(args):
    #     """
    #     Saves a GIF of the agent's observations.
    #     args: (obs_sequence, step_idx, save_dir)
    #     """
    #     obs_seq, step_idx, save_dir = args
        
    #     # obs_seq shape is likely: (Time, Num_Agents, Height, Width, Channels)
    #     # We want: (Time, Height, Width, Channels)
    #     frames_data = np.array(obs_seq[:, 0]) 

    #     # 2. CRITICAL FIX: Handle High Channel Counts
    #     # If the observation has more than 3 channels (e.g. 19), it's likely a "World State" 
    #     # or stacked frames. We must slice strictly the first 3 channels (RGB).
    #     if frames_data.shape[-1] > 3:
    #         # Take only the first 3 channels (R, G, B)
    #         frames_data = frames_data[..., :3]
        
    #     if frames_data.max() <= 1.0:
    #         frames_data = (frames_data * 255).astype(np.uint8)
    #     else:
    #         frames_data = frames_data.astype(np.uint8)

    #     frames = [Image.fromarray(frame) for frame in frames_data]
        
    #     # Ensure directory exists
    #     gif_dir = os.path.join(save_dir, "gifs")
    #     os.makedirs(gif_dir, exist_ok=True)
        
    #     filename = os.path.join(gif_dir, f"step_{step_idx}.gif")
        
    #     frames[0].save(
    #         filename,
    #         format="GIF",
    #         append_images=frames[1:],
    #         save_all=True,
    #         duration=100,  # 100ms per frame
    #         loop=0
    #     )
    #     print(f"Saved GIF to {filename}")


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

        return (actor_state, critic_state, env_state, obs, rng), transition


    def collect_rollout(carry):
        return jax.lax.scan(env_step, carry, None, config["NUM_INNER_STEPS"])

    # def ppo_update(actor_state, critic_state, traj, adv, targets, config):
    #     # obs, actions, logp_old, values_old, _, _ = traj
    #     obs = traj.obs
    #     T, N = obs.shape[:2]
    #     actions = traj.actions
    #     logp_old = traj.log_probs
    #     values_old = traj.values

    #     obs_flat = obs.reshape((T * N,) + obs.shape[2:])
    #     actions = traj.actions.reshape((T * N,))
    #     logp_old = traj.log_probs.reshape((T * N,))
    #     adv = adv.reshape((T * N,))
    #     targets = targets.reshape((T * N,))


    #     def loss_fn(actor_params, critic_params):
    #         pi, _ = actor.apply(actor_params, obs_flat)
    #         logp = pi.log_prob(actions)
    #         ratio = jnp.exp(logp - logp_old)

    #         adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    #         policy_loss = -jnp.mean(
    #             jnp.minimum(
    #                 ratio * adv_n,
    #                 jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * adv_n
    #             )
    #         )

    #         world_state = traj.world_state
    #         T = world_state.shape[0]

    #         world_state_flat = world_state.reshape(
    #             (T,) + world_state.shape[1:]
    #         )

    #         # values = critic.apply(critic_params, values_old)
    #         values = critic.apply(critic_params, world_state_flat)

    #         value_loss = jnp.mean((values - targets) ** 2)

    #         entropy = pi.entropy().mean()
    #         total_loss = policy_loss + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy

    #         metrics = {
    #             "policy_loss": policy_loss,
    #             "value_loss": value_loss,
    #             "entropy": entropy,
    #         }
    #         return total_loss, metrics

    #     (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True, argnums=(0,1))(
    #         actor_state.params, critic_state.params
    #     )

    #     actor_state = actor_state.apply_gradients(grads=grads[0])
    #     critic_state = critic_state.apply_gradients(grads=grads[1])

    #     return actor_state, critic_state, metrics

    def ppo_update(ent_coef_log, actor_state, critic_state, traj, adv, targets, config):
        obs = traj.obs                     # (T, N, H, W, C)
        world_state = traj.world_state     # (T, H, W, C*N)
        actions = traj.actions             # (T, N)
        logp_old = traj.log_probs          # (T, N)

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
            adv_std = jnp.maximum(adv_flat.std(), 1e-4) #1e-8, 0.05
            adv_n = (adv_flat - adv_flat.mean()) / (adv_std)

            policy_loss = -jnp.mean(
                jnp.minimum(
                    ratio * adv_n,
                    jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    ) * adv_n
                )
            )

            # jax.debug.print(
            #     "ratio stats: min={a}, max={b}, mean={c}",
            #     a=jnp.min(ratio),
            #     b=jnp.max(ratio),
            #     c=jnp.mean(ratio),
            # )
            # jax.debug.print(
            #     "adv stats: mean={m}, std={s}, min={mi}, max={ma}",
            #     m=adv_flat.mean(),
            #     s=adv_flat.std(),
            #     mi=adv_flat.min(),
            #     ma=adv_flat.max(),
            # )




            # if ratio.min() < 0.7 or ratio.min() > 0.9 or ratio.max() < 1.1 or ratio.max() > 1.3:
            #     jax.debug.print(
            #         "ratio stats: min={a}, max={b}, mean={c}",
            #         a=jnp.min(ratio),
            #         b=jnp.max(ratio),
            #         c=jnp.mean(ratio),
            #     )
            #     # a=0.7-0.9, b=1.1-1.3, c=~1.0

            # if adv_flat.std() < 1e-3 or adv_flat.abs().max() > 100:
            #     jax.debug.print(
            #         "adv stats: mean={m}, std={s}, min={mi}, max={ma}",
            #         m=adv_flat.mean(),
            #         s=adv_flat.std(),
            #         mi=adv_flat.min(),
            #         ma=adv_flat.max(),
            #     )
            #     # std ≈ 0.3 – 3.0 
            #     # !!!std < 1e-3  OR  |adv| > 100


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

            total_loss = (
                policy_loss
                + config["VF_COEF"] * value_loss
                - ent_coef_log * entropy
            )

            metrics = {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                # "total_loss": total_loss,
                
            }
            return total_loss, metrics

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
        advantages, targets = compute_gae(traj, last_values, config)
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
            ent_coef_log, actor_state, critic_state, traj, advantages, targets, config
        )


        # DOING IN MAIN FILE NOW
        # # Checkpoint saving every 1000 steps
        # do_save = ((step+1) % 1000) == 0


        # def save_checkpoint_callback(args):
        #     payload, do_save = args
        #     if not do_save:
        #         return

        #     payload = jax.tree_map(lambda x: jax.device_get(x), payload)

        #     save_checkpoint(
        #         {
        #             "actor": payload["actor"],
        #             "critic": payload["critic"],
        #         },
        #         SAVE_DIR,
        #         step=int(payload["step"]),
        #     )

        # ckpt_payload = {
        #     "actor": actor_state.params,
        #     "critic": critic_state.params,
        #     "step": step + 1,
        # }

        # DOING IN MAIN FILE NOW
        # jax.debug.callback(
        #     save_checkpoint_callback,
        #     (ckpt_payload, do_save),
        # )

        # jax.debug.callback(
        #     lambda args: (
        #         save_checkpoint(
        #             {
        #                 "actor": actor_state.params,
        #                 "critic": critic_state.params,
        #             },
        #             SAVE_DIR,
        #             step=int(args),
        #         )
        #         if args
        #         else None
        #     ),
        #     do_save,
        # )




        # if (step + 1) % 1000 == 0:
        #     jax.debug.callback(
        #         lambda p: save_checkpoint(...),
        #         None,
        #     )


        # # GIF saving
        # do_save_gif = ((step) % SAVE_GIF_INTERVAL) == 0 & SAVE_GIF
        # cond_args = (traj.obs, step)
        # def _trigger_save_callback(args):
        #     _obs, _step = args
        #     # define a tiny Python helper here that captures 'SAVE_DIR' from the outer scope
        #     def _python_callback_wrapper(callback_args):
        #         obs_val, step_val = callback_args
        #         save_gif_callback((obs_val, step_val, SAVE_DIR))
        #     jax.debug.callback(_python_callback_wrapper, (_obs, _step))

        # jax.lax.cond(
        #     do_save_gif,
        #     _trigger_save_callback,
        #     lambda x: None,
        #     cond_args
        # )



        # TODO:NOT WORKING PROPERLY
        # # Evaluation callback
        # do_eval = jnp.logical_and(step > 0, step % EVAL_INTERVAL == 0)
        # def maybe_eval(args):
        #     actor_p, critic_p, step_val, do_eval = args
        #     if do_eval:
        #         eval_callback(actor_p, critic_p, step_val)

        # jax.debug.callback(
        #     maybe_eval,
        #     (
        #         actor_state.params,
        #         critic_state.params,
        #         step,
        #         do_eval,
        #     ),
        #     # effect="io",
        # )


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


