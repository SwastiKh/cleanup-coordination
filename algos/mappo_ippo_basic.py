import jax
import jax.numpy as jnp
import flax.linen as nn
import distrax
from typing import Any, Tuple
from flax.linen.initializers import orthogonal, constant



class MLPEncoder(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) or (B, D)
        x = x.reshape((x.shape[0], -1))  # 🔴 REQUIRED
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x



# class CNNEncoder(nn.Module):
#     hidden_dim: int = 64

#     @nn.compact
#     def __call__(self, x):
#         x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = nn.relu(x)
#         x = nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2))(x)
#         x = nn.relu(x)
#         x = x.reshape((x.shape[0], -1))  # Flatten
#         x = nn.Dense(self.hidden_dim)(x)
#         x = nn.relu(x)
#         return x

class CNNEncoder(nn.Module):
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = nn.Conv(32, (3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return x


class GRUEncoder(nn.Module): #TODO: incomplete -> no done masking, no hidden reset
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x, h):
        gru = nn.GRUCell()
        h, y = gru(h, x)
        return y, h

    def init_hidden(self, batch_size):
        return jnp.zeros((batch_size, self.hidden_dim))

class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh

        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = activation(x)

        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = activation(x)

        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0))(x)
        x = activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = activation(x)
        return x


# create LSTM from scratch to modify it's elements
# class LSTMEncoder(nn.Module):
#     hidden_dim: int = 64

#     @nn.compact
#     def __call__(self, input, hidden):
#         lstm = nn.LSTMCell()
#         h, y = lstm(hidden, input)
#         return y, h

#     def init_hidden(self, batch_size):
#         return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), self.hidden_dim)

#     def init_carry(self, batch_size):
#         return nn.LSTMCell.initialize_carry(jax.random.PRNGKey(0), (batch_size,), self.hidden_dim)

# class LSTMScratch:
#     def __init__(self, input_size, hidden_size, output_size, num_epochs, learning_rate):
#         # Hyperparameters
#         self.learning_rate = learning_rate
#         self.hidden_size = hidden_size
#         self.num_epochs = num_epochs

#         # Forget Gate
#         self.wf = initWeights(input_size, hidden_size)
#         self.bf = np.zeros((hidden_size, 1))

#         # Input Gate
#         self.wi = initWeights(input_size, hidden_size)
#         self.bi = np.zeros((hidden_size, 1))

#         # Candidate Gate
#         self.wc = initWeights(input_size, hidden_size)
#         self.bc = np.zeros((hidden_size, 1))

#         # Output Gate
#         self.wo = initWeights(input_size, hidden_size)
#         self.bo = np.zeros((hidden_size, 1))

#         # Final Gate
#         self.wy = initWeights(hidden_size, output_size)
#         self.by = np.zeros((output_size, 1))

#     def reset(self):
#         self.concat_inputs = {}

#         self.hidden_states = {-1:np.zeros((self.hidden_size, 1))}
#         self.cell_states = {-1:np.zeros((self.hidden_size, 1))}

#         self.activation_outputs = {}
#         self.candidate_gates = {}
#         self.output_gates = {}
#         self.forget_gates = {}
#         self.input_gates = {}
#         self.outputs = {}

#     def forward(self, inputs):
#         self.reset()

#         outputs = []
#         for q in range(len(inputs)):
#             self.concat_inputs[q] = np.concatenate((self.hidden_states[q - 1], inputs[q]))

#             self.forget_gates[q] = sigmoid(np.dot(self.wf, self.concat_inputs[q]) + self.bf)
#             self.input_gates[q] = sigmoid(np.dot(self.wi, self.concat_inputs[q]) + self.bi)
#             self.candidate_gates[q] = tanh(np.dot(self.wc, self.concat_inputs[q]) + self.bc)
#             self.output_gates[q] = sigmoid(np.dot(self.wo, self.concat_inputs[q]) + self.bo)

#             self.cell_states[q] = self.forget_gates[q] * self.cell_states[q - 1] + self.input_gates[q] * self.candidate_gates[q]
#             self.hidden_states[q] = self.output_gates[q] * tanh(self.cell_states[q])

#             outputs += [np.dot(self.wy, self.hidden_states[q]) + self.by]

#         return outputs

# class MAPPOActor(nn.Module):
#     action_dim: int
#     encoder_type: str = "cnn"   # mlp | cnn | rnn
#     hidden_dim: int = 64

#     def setup(self):
#         if self.encoder_type == "mlp":
#             self.encoder = MLPEncoder(self.hidden_dim)
#         elif self.encoder_type == "cnn":
#             self.encoder = CNNEncoder(self.hidden_dim)
#         elif self.encoder_type == "rnn":
#             self.encoder = GRUEncoder(self.hidden_dim)

#         self.logits = nn.Dense(self.action_dim)

#     def __call__(self, obs, h=None):
#         if self.encoder_type == "rnn":
#             emb, h = self.encoder(obs, h)
#         else:
#             emb = self.encoder(obs)
#             h = None

#         logits = self.logits(emb)
#         pi = distrax.Categorical(logits=logits)
#         return pi, h

# # world_state shape: (B, H, W, C * num_agents)
# class MAPPOCritic(nn.Module):
#     encoder_type: str = "cnn"
#     hidden_dim: int = 64

#     def setup(self):
#         if self.encoder_type == "mlp":
#             self.encoder = MLPEncoder(self.hidden_dim)
#         elif self.encoder_type == "cnn":
#             self.encoder = CNNEncoder(self.hidden_dim)

#         self.value = nn.Dense(1)

#     def __call__(self, world_state):
#         x = self.encoder(world_state)
#         v = self.value(x)
#         return jnp.squeeze(v, -1)


# def collect_rollout(rng, env, actor_params, critic_params, actor, critic,
#                      env_state, obs, h, config):

#     def step_fn(carry, _):
#         rng, env_state, obs, h = carry
#         rng, rng_act, rng_env = jax.random.split(rng, 3)

#         obs_batch = jnp.reshape(obs, (-1,) + obs.shape[2:])

#         pi, h_new = actor.apply(actor_params, obs_batch, h)
#         actions = pi.sample(seed=rng_act)
#         logp = pi.log_prob(actions)

#         world_state = obs["world_state"]
#         world_state = jnp.reshape(world_state, (-1,) + world_state.shape[2:])
#         values = critic.apply(critic_params, world_state)

#         actions = actions.reshape(env.num_agents, -1)
#         obs, env_state, reward, done, info = env.step(rng_env, env_state, actions)

#         transition = (obs_batch, actions, logp, values, reward, done)
#         return (rng, env_state, obs, h_new), transition

#     carry = (rng, env_state, obs, h)
#     carry, traj = jax.lax.scan(step_fn, carry, None, config["NUM_STEPS"])
#     return carry, traj



# def ppo_actor_loss(pi, actions, old_logp, adv, clip_eps):
#     logp = pi.log_prob(actions)
#     ratio = jnp.exp(logp - old_logp)
#     adv = (adv - adv.mean()) / (adv.std() + 1e-8)

#     loss1 = ratio * adv
#     loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv
#     return -jnp.mean(jnp.minimum(loss1, loss2))


# def value_loss(values, targets, clip_eps, old_values):
#     v_clipped = old_values + jnp.clip(values - old_values, -clip_eps, clip_eps)
#     loss1 = (values - targets) ** 2
#     loss2 = (v_clipped - targets) ** 2
#     return 0.5 * jnp.mean(jnp.maximum(loss1, loss2))


class IPPOActor(nn.Module):
    action_dim: int
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, rnn_state):
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        # OLD: new_rnn_state, lstm_out = nn.LSTMCell()(rnn_state, embedding)
        new_rnn_state, lstm_out = nn.LSTMCell(features=64)(rnn_state, embedding) 

        x_d1 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(lstm_out)
        x_a1 = nn.relu(x_d1)

        x_d2 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x_a1)
        x_a2 = nn.relu(x_d2)

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x_a2)

        pi = distrax.Categorical(logits=logits)
        return pi, new_rnn_state

class IPPOCritic(nn.Module):
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, rnn_state):
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        # OLD: new_rnn_state, lstm_out = nn.LSTMCell()(rnn_state, embedding)
        new_rnn_state, lstm_out = nn.LSTMCell(features=64)(rnn_state, embedding) 

        x_d1 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(lstm_out)
        x_a1 = nn.relu(x_d1)

        x_d2 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x_a1)
        x_a2 = nn.relu(x_d2)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x_a2)

        #return jnp.squeeze(value, axis=-1)
        return jnp.squeeze(value, axis=-1), new_rnn_state


class MAPPOActor(nn.Module):
    action_dim: int
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs, rnn_state):
        # self.hidden_dim = 128
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        # OLD: new_rnn_state, lstm_out = nn.LSTMCell()(rnn_state, embedding)
        new_rnn_state, lstm_out = nn.LSTMCell(features=64)(rnn_state, embedding)    

        x_d1 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(lstm_out)
        x_a1 = nn.relu(x_d1)

        x_d2 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x_a1)
        x_a2 = nn.relu(x_d2)

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x_a2)

        pi = distrax.Categorical(logits=logits)
        return pi, new_rnn_state
    
class MAPPOCritic(nn.Module):
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state, rnn_state):
        # world_state: (N, H, W, C * num_agents) or (batch, H, W, C * num_agents)

        embedding = CNN(self.activation)(world_state)

        # OLD: new_rnn_state, lstm_out = nn.LSTMCell()(rnn_state, embedding)
        new_rnn_state, lstm_out = nn.LSTMCell(features=64)(rnn_state, embedding)

        x_d1 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(lstm_out)
        x_a1 = nn.relu(x_d1)

        x_d2 = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(x_a1)
        x_a2 = nn.relu(x_d2)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x_a2)

        # return jnp.squeeze(value, axis=-1)
        return jnp.squeeze(value, axis=-1), new_rnn_state

