# Training tutorial agent with RNN model `MyRNNModel`

This is the agent from the tutorial, with architecture `FC -> LSTM -> Policy, Value`.

Takes >1M steps (1.8 hours on 8 workers) to achieve 177.12 mean reward (200 max). Agent reached 200 max reward after
only 200K timesteps, but took a while to fine tune after that (mean reward 91). I expected the sample efficiency to be
low, but not this low!

Agent saved at `PPO_2021-03-15_13-09-58.zip`.

```
== Status ==
Memory usage on this node: 4.0/12.4 GiB
Using FIFO scheduling algorithm.
Resources requested: 9/12 CPUs, 0/0 GPUs, 0.0/6.93 GiB heap, 0.0/2.34 GiB objects
Result logdir: /mnt/c/Code/weile-lab/cachey/ray_results/PPO_2021-03-15_13-09-58
Number of trials: 1/1 (1 RUNNING)
+-----------------------------+----------+-------------------+--------+------------------+---------+----------+----------------------+----------------------+--------------------+
| Trial name                  | status   | loc               |   iter |   total time (s) |      ts |   reward |   episode_reward_max |   episode_reward_min |   episode_len_mean |
|-----------------------------+----------+-------------------+--------+------------------+---------+----------+----------------------+----------------------+--------------------|
| PPO_CartPole-v0_a7399_00000 | RUNNING  | 172.23.47.39:3276 |    250 |          6614.97 | 1.2e+06 |   177.12 |                  200 |                   47 |             177.12 |
+-----------------------------+----------+-------------------+--------+------------------+---------+----------+----------------------+----------------------+--------------------+
```