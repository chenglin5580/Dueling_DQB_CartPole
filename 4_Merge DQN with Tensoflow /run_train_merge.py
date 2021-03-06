"""
Dueling DQN & Natural DQN comparison

Lin Cheng 2018.01.15

"""

## package input
import gym
import numpy as np
from DDQN_tensorflow import DQN_method, Doubel_DQN_method, Dueling_DQN_method
import matplotlib.pyplot as plt

# 导入environment
env = gym.make('CartPole-v0')
env = env.unwrapped
env.seed(1)

# 确定动作和状态维度
action_dim = env.action_space.n
state_dim = env.observation_space.shape[0]
print(state_dim)

#
method_flag = 3

if method_flag == 1:
    RL_agent = DQN_method(action_dim, state_dim)
elif method_flag == 2:
    RL_agent = Doubel_DQN_method(action_dim, state_dim)
else:
    RL_agent = Dueling_DQN_method(action_dim, state_dim)


def train(RL_agent):
    max_ep = 120
    acc_step = np.zeros((max_ep))
    for ep in range(max_ep):
        state_now = env.reset()
        state_now = np.reshape(state_now, [1, 4])

        for step in range(5000):
            env.render()

            action = RL_agent.chose_action(state_now, train=True)
            # action = env.action_space.sample()

            # print(action)
            state_next, reward, done, _ = env.step(action)
            x, x_dot, theta, theta_dot = state_next
            state_next = np.reshape(state_next, [1, 4])
            # the smaller theta and closer to center the better
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            reward = r1 + r2
            # reward = -100 if done else reward

            # store memory
            RL_agent.memory_store(state_now, action, reward, state_next, done)

            # learn
            # if RL_agent.memory_counter > RL_agent.memory_size:
            if RL_agent.memory_counter > 1000:
                RL_agent.Learn()

            # state update
            state_now = state_next

            if done:
                # 打印分数并且跳出游戏循环
                # plt.scatter(ep, step, color='b')
                # plt.pause(0.1)
                acc_step[ep] = np.array((step))
                print("episode: {}/{}, score: {}，epsilon:{}"
                      .format(ep, 300, step, RL_agent.epsilon))
                break
    RL_agent.model_save()
    return acc_step

acc_step = train(RL_agent)

plt.figure(1)
plt.plot(acc_step)
plt.show()














