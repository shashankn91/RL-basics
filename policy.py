''' policy.py

Generic wrapper for policy generation
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Generator:
    ''' Generate a policy to solve an environment
    '''

    def _create_policy(self, env, gamma):
        ''' Create the policy based on state values

        Params:
            env: the env to generate a policy for
            gamma: the discount rate
        Returns:
            Optimal policy for env
        '''
        states = env.observation_space.n
        actions = env.action_space.n
        state_action = env.env.P
        policy = np.zeros(states)
        for state in range(states):
            state_action_value = np.zeros(actions)
            for action in range(actions):
                for prob, s_prime, reward, done in state_action[state][action]:
                    state_action_value[action] += (prob * (reward + gamma * self._values[s_prime]))
            policy[state] = np.argmax(state_action_value)
        return policy

    def get_policy(self):
        ''' Get the optimal policy for an environment

        Returns:
            np.array of action to take per state
        '''
        return self._policy

    def get_iteration_diffs(self):
        ''' Get the iteration differences

        Returns:
            np.array of differences
        '''
        return self._iteration_diffs


class ValueIteration(Generator):
    ''' Run value iteration against an environment
    '''

    def __init__(self, env, gamma):
        ''' Constructor

        Params:
            env: the env to generate a policy for
            gamma: the discount rate
        '''
        log.info('Running value-iteration')
        self._values = self.__generate_state_values(env, gamma)
        self._policy = self._create_policy(env, gamma)

    def __generate_state_values(self, env, gamma, max_iterations=10000, converge=1e-20):
        ''' Core value iteration algorithm

        Params:
            env: the env to generate a policy for
            gamma: the discount rate
            max_iterations: max amount to run for
            converge: convergence criteria
        Returns:
            Values for each state in env
        '''
        states = env.observation_space.n
        actions = env.action_space.n
        state_action = env.env.P
        values = np.zeros(states)
        deltas = []
        # self imposed max range
        start = time.perf_counter()
        for i in range(max_iterations):
            prev_v = np.copy(values)
            if i % 1000 == 0:
                log.info(f'Value iteration running at {i}')
            for state in range(states):
                state_action_value = np.zeros(actions)
                for action in range(actions):
                    action_value = 0
                    # action may result in stochastic s_prime
                    for prob, s_prime, reward, done in state_action[state][action]:
                        action_value += prob * (reward + gamma * prev_v[s_prime])
                    state_action_value[action] = action_value
                values[state] = np.max(state_action_value)
            diff = np.sum(np.abs(prev_v - values))
            deltas.append(diff)
            if (diff <= converge):
                log.info(f'Value-iteration converged at iteration {i + 1} total s {time.perf_counter() - start}')
                break
        self._iteration_diffs = deltas
        return values


class PolicyIteration(Generator):
    ''' Run policy iteration against an enviroment
    '''

    def __init__(self, env, gamma):
        ''' Constructor

        Params:
            env: the env to generate a policy for
            gamma: the discount rate
        '''
        self._policy = self.__policy_iteraiton(env, gamma)

    def __policy_iteraiton(self, env, gamma, max_iterations=10000):
        ''' Create the policy based on state values

        Params:
            env: the env to generate a policy for
            gamma: the discount rate
        Returns:
            Optimal policy for env
        '''
        # random initial policy
        policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
        start = time.perf_counter()
        col_diff = []
        for i in range(max_iterations):
            if i % 10 == 0:
                log.info(f'Policy-Iteration at {i}')
            self._values, diffs = self.__compute_values(env, policy, gamma)
            new_policy = self._create_policy(env, gamma)
            col_diff.append(diffs)
            if (np.all(policy == new_policy)):
                log.info(f'Policy-Iteration converged at at iteration {i + 1} and took {time.perf_counter() - start}')
                break
            policy = new_policy
        self._iteration_diffs = pd.DataFrame(col_diff).mean(axis=0)
        return policy

    def __compute_values(self, env, policy, gamma=1.0, converge=1e-10):
        ''' Core policy iteration calculation

        Params:
            env: current environment
            policy: current best policy
            gamma: discount rate
            converge: convergence criteria
        Return:
            np.array values of all states
        '''
        states = env.observation_space.n
        state_action = env.env.P
        values = np.zeros(states)
        diffs = []
        while True:
            previous_values = np.copy(values)
            for state in range(states):
                action = policy[state]
                action_value = 0
                for prob, s_prime, reward, done in state_action[state][action]:
                    action_value += prob * (reward + gamma * previous_values[s_prime])
                values[state] = action_value
            diff = np.sum((np.abs(previous_values - values)))
            diffs.append(diff)
            if (diff <= converge):
                # self._iteration_diffs.append(diffs)
                # log.info(f'Converged following existing policy')
                break
        return values, diffs


class QLearning(Generator):

    def __init__(self, env, gamma, alpha, epsilon, epsilon_decay, q_init):
        ''' Setup the q learner

        Params:
            env: open ai gym environment
            gamma: future state influence
            alpha: learning rate
            epsilon: starting random value
            epsilon_decay: decary of random value
            q_init: init q_table with
        '''
        self.__actions = env.action_space.n
        # be careful
        self.__q_table = np.full((env.observation_space.n, self.__actions), q_init, dtype=np.float64)
        self.__gamma = gamma
        self.__alpha = alpha
        self.__epsilon = epsilon
        self.__epsilon_decay = epsilon_decay

    def action(self, state):
        ''' Given a state, perform the best action

        Returns:
            return the action whatever that may be
        '''
        if np.random.uniform(0, 1) < self.__epsilon:
            # take one of the actions
            return np.random.randint(0, self.__actions)
        else:
            return np.argmax(self.__q_table[state])

    def learn(self, state, action, state_prime, reward, done):
        ''' Learn the action of taking a particular action in a state

        Params:
            state: starting state
            action: action taken
            sstate_prime: next state
            reward: reward for taking an action
            done: boolean done or not
        '''
        current_q_val = self.__q_table[state, action]
        # immedate reward + future discounted reward * learning rate
        # don't have to account for end
        next_q_val = self.__alpha * (reward + self.__gamma * self.__q_table[state_prime].max() * (not done))
        self.__q_table[state, action] = (1 - self.__alpha) * current_q_val + next_q_val
        self.__epsilon *= self.__epsilon_decay

    def get_policy(self):
        ''' Return optimal policy

        Returns:
            the q_table's optimal policy
        '''
        return np.argmax(self.__q_table, axis=1)