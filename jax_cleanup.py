import jax
import jax.numpy as jnp
from PIL import Image
import socialjax
from socialjax import make
from pathlib import Path
import math
import numpy as np
import os
from utils import *
import wandb
from algos.mappo_ippo_basic import *
from algos.train_jax import train_jax
# from algos.ppo_utils import compute_gae, ppo_loss
from evaluate import evaluate_policy
from checkpoint import save_checkpoint, load_checkpoint




random_number = np.random.randint(0, 9999)
print("random_number:", random_number)
rng = jax.random.PRNGKey(random_number)
# rng = jax.random.key(random_number)
rng, reset_rng = jax.random.split(rng)
env = make('clean_up',
        num_inner_steps=NUM_INNER_STEPS,
        num_outer_steps=NUM_OUTER_STEPS,
        num_agents=NUM_AGENTS,
        # grid_size=grid_size,
        maxAppleGrowthRate=MAX_APPLE_GROWTH_RATE,
        thresholdDepletion=THRESHOLD_DEPLETION,
        thresholdRestoration=THRESHOLD_RESTORATION,
        dirtSpawnProbability=DIRT_SPAWN_PROBABILITY,
        delayStartOfDirtSpawning=DELAY_START_OF_DIRT_SPAWNING,
        shared_rewards=SHARED_REWARDS,
        shared_cleaning_rewards=SHARED_CLEANING_REWARDS,
        inequity_aversion=INEQUITY_AVERSION,
        inequity_aversion_target_agents=INEQUITY_AVERSION_TARGET_AGENTS
    )
print("Environment loaded.")


run = wandb.init(
    entity="swasti",
    project="mthesis",
    name=SAVE_DIR,
    config=config,
)


print(f"env observation space: {env.observation_space()}")
# print(f"env observation space: {env.observation_space[1]}")
agent_obs_space, obs_shape = env.observation_space() 
print(f"agent observation space: {agent_obs_space}")
print(f"obs shape: {obs_shape}")


root_dir = SAVE_DIR
path = Path(root_dir + "/state_pics")
path.mkdir(parents=True, exist_ok=True)
# # check if directory exists
# path_exists = os.path.exists(path)
# print(f"Directory exists: {path_exists}")

total_reward_per_agent = [0.0 for _ in range(NUM_AGENTS)]
cumulative_total_reward = 0.0

print("Starting JAX-native training...")

obs, env_state = env.reset(reset_rng)
# print(f"Observation after reset: {obs}")
actor_state, critic_state = [None, None]
step = 0


while step < NUM_OUTER_STEPS:
    print(f"\n=== Training steps {step} → {step + EVAL_INTERVAL} ===")

    train_out = train_jax(
        rng = rng,
        env=env,
        config=config,
        actor_state=actor_state,
        critic_state=critic_state,
        obs=obs,
        env_state=env_state,
        start_step=step,
        num_steps=EVAL_INTERVAL,   # <- NEW
    )

    # --- unpack ---
    actor_state = train_out["actor_state"]
    critic_state = train_out["critic_state"]
    env_state = train_out["env_state"]
    obs = train_out["obs"]
    rng = train_out["rng"]
    # step = train_out["step"]
    step = step + EVAL_INTERVAL
    print(f"Completed up to step {step}.")


    params = train_out["params"]
    metrics = train_out["metrics"]
    save_checkpoint(params, SAVE_DIR, step=str(step)) # save every 10-20k steps or so
    # Evaluate
    evaluate_policy(
        env,
        params,
        num_steps=NUM_EVAL_STEPS,
        save_dir=SAVE_DIR,
        current_step=step,
    )

    # print(f"metrics to be logged: {metrics}")
    # wandb_log_callback(metrics)

print("Training finished.")


#PUT INSIDE LOOP ALSO
# # Save final checkpoint explicitly
save_checkpoint(params, SAVE_DIR, step="final")

# Evaluate
evaluate_policy(
    env,
    params,
    num_steps=NUM_INNER_STEPS,
    save_dir=SAVE_DIR,
    current_step="final",
)



# for o_t in range(NUM_OUTER_STEPS):
#     print(f"\n Outer step {o_t+1}/{NUM_OUTER_STEPS} reset done.")
#     obs, old_state = env.reset(_rng)

#     trajectory_buffer = []

#     # render each timestep
#     pics = []

#     img = env.render(old_state)
#     img_np = np.asarray(img)
#     pics.append(img_np)
#     per_agent_rewards = [0.0 for _ in range(NUM_AGENTS)]
#     total_successful_cleans = 0
#     total_apples_collected = 0
#     # total_reward = 0.0

#     if o_t == 0:
#         Image.fromarray(img_np).save(f"{root_dir}/state_pics/init_state.png")

#     for t in range(NUM_INNER_STEPS):

#         rng, *rngs = jax.random.split(rng, NUM_AGENTS+1)
#         if ALGO_NAME == "RANDOM":
#             actions, log_probs, values = [], [], [] # Placeholders
#             for a in range(NUM_AGENTS):
#                 actions.append(jax.random.choice(
#                     rngs[a],
#                     a=env.action_space(0).n,
#                     p=jnp.array([0.1, 0.1, 0.09, 0.09, 0.09, 0.19, 0.05, 0.1, 0.5])))
#         elif ALGO_NAME == "PPO":
#             _rng_action = rngs[0]
#             actions, log_probs, values = ppo_agent.sample_action(ppo_state, obs, _rng_action)
#         elif ALGO_NAME == "IPPO":
#             _rng_action = rngs[0]
#             actions, log_probs, values = ippo_agent.sample_action(ippo_state, obs, _rng_action)
#         elif ALGO_NAME == "MAPPO":
#             _rng_action = rngs[0]
#             actions, log_probs, values = mappo_agent.sample_action(mappo_state, obs, _rng_action)
#         else:
#             raise NotImplementedError("Please check the algorithm name.")
        


#         next_obs, state, reward, done, info = env.step_env(
#             rng, old_state, [a for a in actions]
#         )


#         print('###################')
#         print(f'timestep: {t} to {t+1}')
#         print(f'actions: {[action.item() for action in actions]}')
#         print(f'reward: {[r.item() for r in reward]}')
#         print(f"info: {info}")
#         print("###################")

#         # wandb.log({
#         #     "reward_agent0": reward[0].item(),
#         #     "reward_agent1": reward[1].item(),
#         #     "total_reward": reward[0].item() + reward[1].item(),
#         #     "total_successful_cleans": info['total_successful_cleans'],
#         #     "total_apples_collected": info['total_apples_collected'],
#         # })
#         total_successful_cleans += info['total_successful_cleans']
#         total_apples_collected += info['total_apples_collected']

#         if ALGO_NAME in ["PPO", "IPPO", "MAPPO"]:
#             done_array = jnp.full((NUM_AGENTS,), done["__all__"])
#             trajectory_buffer.append({
#                 "obs": obs,
#                 "actions": actions,
#                 "log_probs": log_probs,
#                 "rewards": reward,
#                 "dones": done_array,
#                 "values": values,
#             })

#         img = env.render(state)
#         # Image.fromarray(img).save(
#         #     f"{root_dir}/state_pics/state_{t+1}.png"
#         # )
#         pics.append(img)

#         old_state = state
#         obs = next_obs
#         per_agent_rewards = [per_agent_rewards[i] + reward[i].item() for i in range(NUM_AGENTS)]
#         # total_reward += sum([reward.item() for reward in reward])

#     if ALGO_NAME == "PPO":
#         print("Updating PPO agent...")
#         rng, _rng = jax.random.split(rng)
        
#         # 'obs' is now the last 'next_obs' from the loop, required for GAE
#         ppo_state, loss_info = ppo_agent.update(ppo_state, trajectory_buffer, obs, _rng)
        
#         print(f"Update done. Total Loss: {loss_info['total_loss']:.4f}, "
#               f"Policy Loss: {loss_info['policy_loss']:.4f}, "
#               f"Value Loss: {loss_info['value_loss']:.4f}")
        
#         print("PPO agent updated.")
        
#     elif ALGO_NAME == "IPPO":
#         print("Updating IPPO agent...")
#         rng, _rng = jax.random.split(rng)
        
#         ippo_state, loss_info = ippo_agent.update(ippo_state, trajectory_buffer, obs, _rng)
        
#         print(f"Update done. Total Loss: {loss_info['total_loss']:.4f}, "
#               f"Policy Loss: {loss_info['policy_loss']:.4f}, "
#               f"Value Loss: {loss_info['value_loss']:.4f}")
        
#         print("IPPO agent updated.")
        
#     elif ALGO_NAME == "MAPPO":
#         print("Updating MAPPO agent...")
#         rng, _rng = jax.random.split(rng)
        
#         mappo_state, loss_info = mappo_agent.update(mappo_state, trajectory_buffer, obs, _rng)
        
#         print(f"Update done. Total Loss: {loss_info['total_loss']:.4f}, "
#               f"Policy Loss: {loss_info['policy_loss']:.4f}, "
#               f"Value Loss: {loss_info['value_loss']:.4f}")
        
#         print("MAPPO agent updated.")

#     if not SAVE_GIF:
#         continue
#     else:
#         if (o_t + 1) % 1000 == 0 or o_t == 0:
#             # create and save gif
#             print(f"Saving GIF for the outer step {o_t + 1} in location {root_dir}/state_outer_step_{o_t+1}.gif...")
#             pics = [Image.fromarray(img) for img in pics]
#             pics[0].save(
#             f"{root_dir}/state_outer_step_{o_t+1}.gif",
#             format="GIF",
#             save_all=True,
#             optimize=False,
#             append_images=pics[1:],
#             duration=150,
#             loop=0,
#             )

#     # total_reward = sum([reward.item() for reward in reward])
#     print("Per-agent rewards:", per_agent_rewards)
#     print(f"Total reward in outer step {o_t+1}: {sum(per_agent_rewards)}")
#     cumulative_total_reward += sum(per_agent_rewards)
#     total_reward_per_agent = [total_reward_per_agent[i] + per_agent_rewards[i] for i in range(NUM_AGENTS)]

#     log_dict = {
#         "combined_reward": sum(per_agent_rewards), 
#         "agent0_reward": per_agent_rewards[0],
#         "agent1_reward": per_agent_rewards[1],
#         "total_successful_cleans": total_successful_cleans,
#         "total_apples_collected": total_apples_collected,
#         "cumulative_total_reward": cumulative_total_reward,
#     }
    
#     # Add loss info only if we have a training algorithm
#     if ALGO_NAME in ["PPO", "IPPO", "MAPPO"]:
#         log_dict.update({
#             "total_loss": loss_info['total_loss'],
#             "policy_loss": loss_info['policy_loss'],
#             "value_loss": loss_info['value_loss'],
#         })
    
#     run.log(log_dict)




# params = load_checkpoint("MAPPOMLP_gif/.../checkpoint_step_1000.pkl")


# print(f"\nCumulative total reward after {NUM_OUTER_STEPS} outer steps: {cumulative_total_reward}")
# for i in range(NUM_AGENTS):
#     print(f"Total reward for agent {i}: {total_reward_per_agent[i]}")

# print("\nSimulation completed.")
