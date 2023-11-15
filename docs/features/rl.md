# Reinforcement Learning

We currently support PyTorch based Reinforcement Learning frameworks. EnCortex integrates with [Stable-Baselines 3](https://stable-baselines3.readthedocs.io/), [D3RLPY](https://d3rlpy.readthedocs.io/) and has a native implementation of some algorithms as well.

## Stable Baselines 3

1. DQN
2. A2C
3. PPO

## D3RLPY

1. Discrete SAC

## Native

1. DQN
2. Noisy DQN

In addition to this, we also support Offline Reinforcement Learning via D3RLPY. To collect a D3RLPY compatible dataset, `OfflineCollectTrajectoryCallback` can be used directly with any RL environment. Usage: `from encortex.callbacks.offline_rl_callback import OfflineCollectTrajectoryCallback`