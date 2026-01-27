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
    def __call__(self, obs):
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(embedding)
        x = nn.relu(x)

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)

        pi = distrax.Categorical(logits=logits)
        return pi, None

class IPPOCritic(nn.Module):
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(embedding)
        x = nn.relu(x)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)

        return jnp.squeeze(value, axis=-1)


class MAPPOActor(nn.Module):
    action_dim: int
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        # obs: (N, H, W, C) or (batch, H, W, C)

        embedding = CNN(self.activation)(obs)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(embedding)
        x = nn.relu(x)

        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)

        pi = distrax.Categorical(logits=logits)
        return pi, None
    
class MAPPOCritic(nn.Module):
    encoder_type: str = "cnn"
    activation: str = "relu"

    @nn.compact
    def __call__(self, world_state):
        # world_state: (N, H, W, C * num_agents) or (batch, H, W, C * num_agents)

        embedding = CNN(self.activation)(world_state)

        x = nn.Dense(64, kernel_init=orthogonal(jnp.sqrt(2)),
                     bias_init=constant(0.0))(embedding)
        x = nn.relu(x)

        value = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)

        return jnp.squeeze(value, axis=-1)

