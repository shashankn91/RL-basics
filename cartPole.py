import gym
import numpy as np
env = gym.make('FrozenLake-v0')
env.reset()
num_states = 100
num_actions = 4
alpha = 0.2
gamma = 0.9
epsilon = 0.003
q = np.array([[0]*num_actions]*num_states , dtype=float)
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action

    #state 32*11*2