

## package input
import gym
import numpy as np
from DDQN_tensorflow import DQN_method
import matplotlib.pyplot as plt



env = gym.make('CartPole-v0')
env = env.unwrapped

action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]

RL_agent = DQN_method(action_dim, state_dim, reload_flag=True)


for ep in range(10):
    state = env.reset()
    state_now = np.reshape(state, [1, 4])

    for step in range(5000):
        env.render()

        action = RL_agent.chose_action(state_now, train=False)
        # action = env.action_space.sample()

        # print(action)
        state_next, reward, done, _  = env.step(action)
        x, x_dot, theta, theta_dot = state_next
        state_next = np.reshape(state_next, [1, 4])
        # the smaller theta and closer to center the better
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        # reward = -100 if done else reward

        # state update
        state_now = state_next


        if done:
            # 打印分数并且跳出游戏循环
            print("episode: {}/{}, score: {}，epsilon:{}"
                  .format(ep, 300, step, RL_agent.epsilon))
            break

