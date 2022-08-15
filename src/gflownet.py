from multiprocessing.resource_sharer import stop
from typing import List, Tuple
from jax import numpy as jnp
import jax
import equinox as eqx

from src.actorcritic import MLP


class ReplayBuffer(eqx.Module):
    buffer: jnp.ndarray
    size: int
    currsize: int

    def __init__(self, observation_shape, size=100):
        self.buffer = {
            "observations": jnp.zeros((size, *observation_shape)),
            "actions": jnp.zeros((size,)),
            "rewards": jnp.zeros((size,)),
        }
        self.size = size
        self.currsize = 0

    def add(self, observation, action, reward):
        observations = (
            jnp.roll(self.buffer["observations"], 1, axis=0).at[0].set(observation)
        )
        actions = jnp.roll(self.buffer["actions"], 1, axis=0).at[0].set(action)
        rewards = jnp.roll(self.buffer["rewards"], 1, axis=0).at[0].set(reward)

        newbuff = eqx.tree_at(
            lambda _: "buffer",
            newbuff,
            {"observations": observations, "actions": actions, "rewards": rewards},
        )
        newbuff = eqx.tree_at(
            lambda _: "currsize", newbuff, jnp.min([newbuff.currsize + 1, newbuff.size])
        )
        return newbuff


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


def flow_matching_loss(
    flow_net: FlowNetwork,
    states,
    rewards,
    terminal_states,
    stop_actions,
    transition_function,
    inverse_transition_function,
    state_encoding_function,
    epsilon=1e-05,
):
    """
    Compute the flow matching loss between the predicted flow and the ground truth flow.
    (https://papers.nips.cc/paper/2021/file/e614f646836aaed9f89ce58e837e2310-Paper.pdf).
    Assumes that last action is the stop action.
    """

    dones_indices = jnp.argwhere(
        terminal_states, size=terminal_states.shape[0]
    ).flatten()

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

    def loss(idx, state, reward, is_terminal, is_stop_action):
        is_source = jax.lax.cond(
            (idx == 0), lambda _: True, lambda _: (terminal_states[idx - 1]), None
        )
        inflow = jnp.log(sum_exp_inflow(state) + epsilon)

        outflow = jnp.log(
            (1 - (is_terminal * (1 - is_stop_action))) * sum_exp_outflow(state)
            + epsilon
            + (1 - is_stop_action) * reward
        )
        state_flowloss = jax.lax.integer_pow(inflow - outflow, 2)

        # if a stop action was used, we insert an artificial terminal state where the only parent is the state the stop action was used in
        stop_action_inflow = jax.lax.cond(
            is_stop_action,
            lambda _: jnp.log(
                epsilon + jnp.exp(flow_net(state_encoding_function(state))[-1])
            ),
            lambda _: 0.0,
            None,
        )
        stop_action_outflow = jax.lax.cond(
            is_stop_action, lambda _: jnp.log(reward + epsilon), lambda _: 0.0, None
        )

        stop_action_flowloss = jax.lax.integer_pow(
            stop_action_inflow - stop_action_outflow, 2
        )

        # we discard the loss for the remaining states (if any), as they don't have a reward signal
        should_keep = (idx <= jnp.max(dones_indices)) * (1 - is_source)

        return should_keep * (state_flowloss + stop_action_flowloss)

    return jax.vmap(loss,)(
        jnp.arange(0, states.shape[0]), states, rewards, terminal_states, stop_actions
    ).sum()
