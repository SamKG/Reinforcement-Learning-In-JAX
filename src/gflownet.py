from typing import List, Tuple
from jax import numpy as jnp
import jax
import equinox as eqx

from src.actorcritic import MLP


class FlowNetwork(eqx.Module):
    net: MLP

    def __init__(
        self,
        obs_space_size: int,
        observation_hidden_features: List[int],
        key: jax.random.KeyArray,
    ):
        self.net = MLP([obs_space_size + 1] + observation_hidden_features + [1], key)

    @eqx.filter_jit
    def __call__(
        self, observation: jnp.ndarray, action: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the flow for a given observation and action
        """
        return self.net(jnp.hstack([observation, action]))


def flow_matching_loss(
    flow_net: FlowNetwork,
    states,
    rewards,
    episode_mask,
    transition_function,
    inverse_transition_function,
    state_encoding_function,
    epsilon=1e-05,
):
    """
    Compute the flow matching loss between the predicted flow and the ground truth flow.
    (https://papers.nips.cc/paper/2021/file/e614f646836aaed9f89ce58e837e2310-Paper.pdf)
    """

    def inflow_loss(state):
        tree = inverse_transition_function(state)
        return (
            jax.vmap(
                lambda parent, action, valid: valid
                * jnp.exp(flow_net(state_encoding_function(parent), action)),
                0,
            )(tree["states"], tree["actions"], tree["valid"]).sum()
            + epsilon
        )

    def outflow_loss(state, reward):
        tree = transition_function(state)
        return (
            jax.vmap(
                lambda action, valid: valid
                * jnp.exp(flow_net(state_encoding_function(state), action)),
                0,
            )(tree["actions"], tree["valid"]).sum()
            + epsilon
            + reward
        )

    def loss(state, reward, mask):
        parents = inverse_transition_function(state)
        n_valid = jnp.sum(parents["valid"])
        return mask * jax.lax.cond(
            n_valid > 0,
            lambda _: jax.lax.integer_pow(
                jnp.log(inflow_loss(state)) - jnp.log(outflow_loss(state, reward)),
                2,
            ),
            lambda _: 0.0,
            None,
        )

    return jax.vmap(
        loss,
    )(states, rewards, episode_mask).sum()
