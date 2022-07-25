from dataclasses import dataclass
import functools
import jax.numpy as jnp
import jax
import rlax
from distrax import Categorical
from jax import device_put
import equinox as eqx

from src.actorcritic import ActorCritic


# %%
from re import A
from typing import Dict
from gym import Env


@functools.partial(jax.jit, static_argnames=("gamma"))
def trajectory_pytree(
    observations, actions, logits, rewards, discounts_mask, values, gamma
):
    """
    Computes a PyTree of useful quantities from trajectory data.
    """
    observations = jnp.array(observations)
    actions = jnp.array(actions)
    logits = jnp.array(logits)
    rewards = jnp.array(rewards)
    discounts_mask = jnp.array(discounts_mask)
    values = jnp.concatenate(values)

    # what's the point of passing values here? they are never used..
    discounted_rewards = rlax.discounted_returns(
        rewards, discounts_mask * gamma, values, stop_target_gradients=True
    )

    advantages = discounted_rewards - values

    # We don't compute gradient of loss w.r.t. old probabilities (https://arxiv.org/pdf/1707.06347.pdf)
    old_probabilities = jax.vmap(
        lambda action, logits: Categorical(logits).log_prob(action), 0
    )(jax.lax.stop_gradient(actions), jax.lax.stop_gradient(logits))

    return {
        "observations": observations,
        "actions": actions,
        "logits": logits,
        "rewards": rewards,
        "discounts": discounts_mask * gamma,
        "values": values,  # values are *estimates* given by criticc
        "discounted_rewards": discounted_rewards,
        "advantages": advantages,  # advantages are *estimates*
        "probabilities": old_probabilities,
    }


split_keys_permanent = jax.jit(jax.random.split, static_argnames=("num",))


def policy_trajectory(
    env: Env,
    agent: ActorCritic,
    steps_per_episode: int,
    gamma: float,
    key: jax.random.KeyArray,
) -> Dict[str, jnp.ndarray]:
    """
    Collects steps_per_episode steps from the environment using the agent's policy. Note that there may be multiple terminal states in the trajectory.
    """
    observation, done = device_put(jnp.array(env.reset()).flatten()), False
    observations, actions, logits, rewards, discounts_mask, values = (
        [None] * steps_per_episode,
        [None] * steps_per_episode,
        [None] * steps_per_episode,
        [None] * steps_per_episode,
        [None] * steps_per_episode,
        [None] * steps_per_episode,
    )
    keys = split_keys_permanent(key, steps_per_episode)
    for step in range(steps_per_episode):
        if done:
            observation = device_put(jnp.array(env.reset()).flatten())

        observations[step] = observation
        # discounts of 0.0 are used to signal terminal states
        discounts_mask[step] = 0.0 if done else 1.0

        action, action_logits = agent.act_with_logits(observation, keys[step])
        observation, reward, done, info = env.step(action.item())
        observation = device_put(jnp.array(observation).flatten())

        actions[step] = action
        logits[step] = action_logits
        rewards[step] = reward
        values[step] = agent.critique(observation)

    trajectory = trajectory_pytree(
        observations, actions, logits, rewards, discounts_mask, values, gamma=gamma
    )

    return trajectory


@functools.partial(eqx.filter_jit)
def step_model_ppo(agent, trajectory, optimizer, optimizer_state):
    @eqx.filter_value_and_grad
    def get_loss(agent: ActorCritic, trajectory, c1: float, c2: float):
        observation_logits = jax.vmap(agent.get_action_logits, 0)(
            trajectory["observations"]
        )
        action_probs = jax.vmap(
            lambda logits, action, oldprob: jnp.exp(
                Categorical(logits).log_prob(action) - oldprob
            ),
            0,
        )(
            observation_logits,
            trajectory["actions"],
            trajectory["probabilities"],
        )

        entropies = jax.vmap(lambda logits, action: Categorical(logits).entropy(), 0)(
            trajectory["logits"], trajectory["actions"]
        )

        v_tm1 = trajectory["values"][:-1]
        v_t = trajectory["values"][1:]
        td_losses = jax.vmap(rlax.td_learning, 0)(
            v_tm1,
            trajectory["rewards"][1:],
            trajectory["discounts"][1:],
            v_t,
        )

        return (
            rlax.clipped_surrogate_pg_loss(
                action_probs, trajectory["advantages"], epsilon=0.2
            )
            + c1 * td_losses.mean()
            - c2 * entropies.mean()
        )

    loss, gradients = get_loss(agent, trajectory, c1=0.5, c2=0.01)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, agent)
    agent = eqx.apply_updates(agent, updates)
    return loss, agent, optimizer_state
