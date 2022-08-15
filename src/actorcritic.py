from typing import Callable, List, Sequence, Tuple
import equinox as eqx
import jax.numpy as jnp
import jax
from distrax import Categorical

# use our own implementation of MLP, since it allows for different layer sizes
class MLP(eqx.Module):
    layers: Sequence[eqx.nn.Linear]
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray]

    def __init__(
        self,
        features: List[int],
        key: jax.random.KeyArray,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.leaky_relu
    ):
        keys = jax.random.split(key, len(features) - 1)
        self.layers = [
            eqx.nn.Linear(features[i], features[i + 1], key=keys[i])
            for i in range(len(features) - 1)
        ]
        self.activation_fn = activation_fn

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for (idx, layer) in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = self.activation_fn(x)
        return x


class Actor(MLP):
    def __init__(
        self,
        obs_space_size: int,
        hidden_features: List[int],
        action_space_size: int,
        key: jax.random.KeyArray,
    ):
        features = [obs_space_size] + hidden_features + [action_space_size]
        super(Actor, self).__init__(features, key)


class Critic(MLP):
    def __init__(
        self, obs_space_size: int, hidden_features: List[int], key: jax.random.KeyArray
    ):
        features = [obs_space_size] + hidden_features + [1]
        super(Critic, self).__init__(features, key)


class ActorCritic(eqx.Module):
    """
    Separate Networks architecture implementation of an Actor Critic
    (https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
    """

    actor: Actor
    critic: Critic

    def __init__(
        self,
        obs_space_size: int,
        action_space_size: int,
        actor_hidden_features: List[int],
        critic_hidden_features: List[int],
        key: jax.random.KeyArray,
    ):
        (actor_key, critic_key) = jax.random.split(key)

        self.actor = Actor(
            obs_space_size, actor_hidden_features, action_space_size, actor_key
        )
        self.critic = Critic(obs_space_size, critic_hidden_features, critic_key)

    @eqx.filter_jit
    def get_action_logits(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.actor(observation)

    @eqx.filter_jit
    def act_with_logits(
        self, observation: jnp.ndarray, key: jax.random.KeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        action_logits = self.get_action_logits(observation)
        return (Categorical(action_logits).sample(seed=key), action_logits)

    @eqx.filter_jit
    def act(self, observation: jnp.ndarray, key: jax.random.KeyArray) -> jnp.ndarray:
        action_logits = self.get_action_logits(observation)
        return Categorical(action_logits).sample(seed=key)

    @eqx.filter_jit
    def critique(self, observation: jnp.ndarray) -> jnp.ndarray:
        return self.critic(observation)

    def __call__(
        self, observation: jnp.ndarray, key: jax.random.KeyArray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.act_with_logits(observation, key)
