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
        action_space_size: int,
        key: jax.random.KeyArray,
    ):
        self.net = MLP(
            [obs_space_size] + observation_hidden_features + [action_space_size],
            key,
        )

    def __call__(self, observation: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Computes the flow for a given observation and action
        """
        return self.net(observation)


# TODO: Cleanup excessive mask args
def flow_matching_loss(
    flow_net: FlowNetwork,
    states,
    rewards,
    dones,
    transition_function,
    inverse_transition_function,
    state_encoding_function,
    epsilon=1e-05,
):
    """
    Compute the flow matching loss between the predicted flow and the ground truth flow.
    (https://papers.nips.cc/paper/2021/file/e614f646836aaed9f89ce58e837e2310-Paper.pdf)
    """

    def sum_exp_inflow(state):
        parents = inverse_transition_function(state)
        parents_action_prob = jax.vmap(
            lambda s, a: flow_net(state_encoding_function(s))[a]
        )(parents["states"], parents["actions"])
        return jax.vmap(lambda v, p: v * jnp.exp(p))(
            parents["valid"], parents_action_prob
        ).sum()

    def sum_exp_outflow(state):
        children = transition_function(state)
        actions_probs = flow_net(state_encoding_function(state))

        return jax.vmap(lambda v, a: v * jnp.exp(a))(
            jnp.concatenate([children["valid"], jnp.array([1.0])]), actions_probs
        ).sum()

    def loss(idx, state, reward, is_terminal):
        is_source = (idx == 0) | (dones[idx - 1])
        inflow = jnp.log(sum_exp_inflow(state) + epsilon)
        outflow = jnp.log(
            is_terminal * sum_exp_outflow(state) + epsilon  +  (1 - is_terminal) * reward
        )

        # jax.debug.print("{x} {y} {z}", x=inflow, y=outflow, z=inflow - outflow)

        return is_source * jax.lax.integer_pow(inflow - outflow, 2)

    return jax.vmap(
        loss,
    )(jnp.arange(0, states.shape[0]), states, rewards, dones).sum()
