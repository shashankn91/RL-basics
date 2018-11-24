''' experiment.py

Run and collect all of the experiment metrics here
'''
import numpy as np
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class Experiment:
    ''' Run a policy and environment together
    '''

    def __init__(self, env, policy):
        ''' Constructor

        Params:
            problem: name of the problem
            env: the env to run against
            policy: the policy to use
        '''
        self.__env = env
        self.__policy = policy

    def evaluate_policy(self, episodes=1000, render=False):
        ''' Evaluate a policy

        Params:
            episdoes: number of episodes to run
            render: debug a mdp
        Returns:
            DataFrame of scores per episode run
        '''
        scores = []
        for _ in range(episodes):
            obs = self.__env.reset()
            current_score_trail = []
            total_reward = 0
            idx = 0
            while True:
                if render:
                    self.__env.render()
                obs, reward, done, _ = self.__env.step(int(self.__policy[obs]))
                current_score_trail.append(reward)
                if done:
                    break
            scores.append(current_score_trail)
        return pd.DataFrame(scores)