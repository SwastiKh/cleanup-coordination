import wandb
#  All hyperparameters
LOG_WANDB = True  

# ENV SETTINGS
NUM_AGENTS=3
# GRID_SIZE = (18,25) #make it smaller (maybe half the size) for faster training
NUM_INNER_STEPS=256 #256
NUM_OUTER_STEPS=10240000 #10e4
BATCH_SIZE = 51200
EVAL_INTERVAL = 51200 # keep multiple of 256
NUM_EVAL_STEPS = 256  # Number of steps during evaluation
SAVE_CHECKPOINT_INTERVAL = 512000  # Save model checkpoint every n outer steps
SAVE_GIF = True
SAVE_GIF_INTERVAL = 512000  # Save GIF every n outer steps
# EVAL_STEPS should be similar to steps in training episodes for consistency
MAX_APPLE_GROWTH_RATE=0.3  # #deepmind paper- 0.03, human=0.067
THRESHOLD_DEPLETION=0.32  # 0.4 #deepmind paper- 0.32, human=0.6
THRESHOLD_RESTORATION=0.2 # 0.0 #deepmind paper- 0.0, human=0.3
DIRT_SPAWN_PROBABILITY=0.5  # 0.5 #deepmind paper- 0.5, human=0.6
DELAY_START_OF_DIRT_SPAWNING=50  # 50
RUN_NAME = "share_clean_rwd_no_LSTM_bigEnv_obs11"  # "SHARED" or "INEQUITY_AVERSION" or "BASIC_ENV"
SHARED_CLEANING_REWARDS = True  # True or False
SHARED_REWARDS = False  # True or False
INEQUITY_AVERSION = False  # True or False
INEQUITY_AVERSION_TARGET_AGENTS = None  # None or list of agent indices

OBS_SIZE = 7 #default is 11 for bigger env 

# Algo settings
ALGO_NAME = "IPPO" # "RANDOM" or "PPO" or "MAPPO" or "IPPO"
ENCODER = "CNN"  # "MLP" or "CNN" or "RNN"
USE_LSTM = False

# PPO 
ACTOR_LR = 3e-4
CRITIC_LR = 1e-3
# LEARNING_RATE = 3e-4
GAMMA = 0.995           # Discount factor
GAE_LAMBDA = 0.95      # Lambda for Generalized Advantage Estimation
CLIP_EPS = 0.1         # PPO clipping epsilon
NUM_EPOCHS = 4         # Number of epochs to train on a batch
MINIBATCH_SIZE = 128    # Number of minibatches to split a batch into 
# ENT_COEF_START = 0.9        # Entropy coefficient
# ENT_COEF_END = 0.02        # Entropy coefficient
ENT_COEF = 0.01
VF_COEF = 0.25          # Value function coefficient

# OTHER
# SAVE_DIR = f"{ALGO_NAME+ENCODER}_gif/a{NUM_AGENTS}_i{NUM_INNER_STEPS}_o{NUM_OUTER_STEPS}_dep{THRESHOLD_DEPLETION}_res{THRESHOLD_RESTORATION}_dsp{DIRT_SPAWN_PROBABILITY}_dsds{DELAY_START_OF_DIRT_SPAWNING}_rs{REWARD_STRUCTURE}"
SAVE_DIR = f"{ALGO_NAME+ENCODER}_gif/a{NUM_AGENTS}_i{NUM_INNER_STEPS}_rs{RUN_NAME}"

USE_CHECKPOINTS = False
LOAD_DIR = f"{SAVE_DIR}/checkpoint_step_{NUM_OUTER_STEPS}.pkl"


# if ALGO_NAME == "PPO":
ppo_hyperparams = {
    # "LEARNING_RATE": LEARNING_RATE,
    "ACTOR_LR": ACTOR_LR,
    "CRITIC_LR": CRITIC_LR,
    "GAMMA": GAMMA,
    "GAE_LAMBDA": GAE_LAMBDA,
    "CLIP_EPS": CLIP_EPS,
    "NUM_EPOCHS": NUM_EPOCHS,
    "MINIBATCH_SIZE": MINIBATCH_SIZE,
    # "ENT_COEF_START": ENT_COEF_START,
    # "ENT_COEF_END": ENT_COEF_END,
    "ENT_COEF": ENT_COEF,
    "VF_COEF": VF_COEF,
    "NUM_INNER_STEPS": NUM_INNER_STEPS,
}

config = {
    "NUM_AGENTS": NUM_AGENTS,
    "NUM_INNER_STEPS": NUM_INNER_STEPS,
    "NUM_OUTER_STEPS": NUM_OUTER_STEPS,
    "EVAL_INTERVAL": EVAL_INTERVAL,
    "BATCH_SIZE": BATCH_SIZE,
    "NUM_EVAL_STEPS": NUM_EVAL_STEPS,
    # "LEARNING_RATE": LEARNING_RATE,
    "ACTOR_LR": ACTOR_LR,
    "CRITIC_LR": CRITIC_LR,
    "GAMMA": GAMMA,
    "GAE_LAMBDA": GAE_LAMBDA,
    "CLIP_EPS": CLIP_EPS,
    "NUM_EPOCHS": NUM_EPOCHS,
    "MINIBATCH_SIZE": MINIBATCH_SIZE,
    # "ENT_COEF_START": ENT_COEF_START,
    # "ENT_COEF_END": ENT_COEF_END,
    "ENT_COEF": ENT_COEF,
    "VF_COEF": VF_COEF,
    "ENCODER": ENCODER,
    "USE_LSTM": USE_LSTM,
}

def wandb_log_callback(metrics):
    wandb.log({k: float(v) for k, v in metrics.items()})

