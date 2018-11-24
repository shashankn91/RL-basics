'''main.py

The driver to the entire assignment
'''
import argparse
from pathlib import Path
from policy import ValueIteration, PolicyIteration, QLearning
from experiment import Experiment
import gym
import logging
from matplotlib import rcParams, rc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
output_dir = './results/'

# setup the plotter
rcParams.update({'figure.autolayout': False, 'lines.linewidth': 2.0})
rc('axes', grid=True)
rc('grid', linestyle='dotted')
rc('figure', dpi=1000)

title = {
    'frozen-lake': 'Frozen Lake',
    'taxi': 'Taxi',
    'value-iteration': 'Value Iteration',
    'policy-iteration': 'Policy Iteration',
    'q-learning': 'Q-learning'
}


def value_policy_iteration_results(results, mdp='frozen-lake', algorithm='value-iteration'):
    ''' Generate and print out results

    Params:
        results: the reults from running the algorithm
        mdp: name of mdp
        algorithm: name of algorithm
    '''
    mdp_title = title[mdp]
    algorithm_title = title[algorithm]
    all_plots = [
        plt.subplots(3, 2, num=0, sharex=True, sharey=True),
        plt.subplots(3, 2, num=1, sharex=True, sharey=True),
        plt.subplots(3, 2, num=3, sharex=True, sharey=True)
    ]
    for i, (gamma, result, diffs) in enumerate(results):
        fig, axes = all_plots[0]
        action_average = result.mean(axis=0)
        current_axis = axes[int(i / 2), i % 2]
        current_axis.plot(action_average.index, action_average)
        current_axis.set_title(f'$\gamma$ = {gamma}')
        # if mdp == 'frozen-lake':
        #    current_axis.set_ylim([0, 0.15])

        fig, axes = all_plots[1]
        current_axis = axes[int(i / 2), i % 2]
        rolling_score = result.sum(axis=1).rolling(window=100, center=False).mean().dropna().as_matrix()
        current_axis.plot(range(100, 1001), rolling_score)
        current_axis.set_title(f'$\gamma$ = {gamma}')

        fig, axes = all_plots[2]
        current_axis = axes[int(i / 2), i % 2]
        current_axis.plot(range(0, len(diffs)), diffs)
        current_axis.set_title(f'$\gamma$ = {gamma}')

    log.info('Making plots dir')
    Path(f'{output_dir}/{mdp}-{algorithm}').mkdir(exist_ok=True)
    for current_plot in range(4):
        fig = plt.figure(current_plot)
        if current_plot == 0:
            # 6.4, 4.8
            fig.set_size_inches(7.2, 6.0)
            fig.suptitle(f'Average Reward Recieved at Action over all Episodes \n  Solving {mdp_title}')
            fig.text(0.5, 0.025, 'Actions Taken', ha='center')
            fig.text(0.025, 0.5, 'Average Reward after Actions', va='center', rotation='vertical')
            fig.savefig(f'{output_dir}/{mdp}-{algorithm}/average-reward-per-iteration.pdf')
        elif current_plot == 1:
            fig.set_size_inches(6.4, 6.0)
            fig.suptitle(f'Rolling Mean of Rewards Over 100 Episodes \n Solving {mdp_title}')
            fig.text(0.5, 0.025, 'Episode', ha='center')
            fig.text(0.025, 0.5, 'Rolling Mean of Rewards', va='center', rotation='vertical')
            fig.savefig(f'{output_dir}/{mdp}-{algorithm}/rolling-score.pdf')
        elif current_plot == 2:
            iterations_used = list(map(lambda item: item[1].notnull().sum(axis=1).mean(), results))
            gammas = list(map(lambda item: str(item[0]), results))
            x_ticks_range = range(len(gammas))
            plt.plot(2)
            plt.bar(x_ticks_range, iterations_used)
            plt.xticks(x_ticks_range, gammas)
            plt.title(f'Average Actions Used Per Episode Solving {mdp_title}')
            plt.ylabel('Actions Used')
            plt.xlabel('$\gamma$')
            plt.savefig(f'{output_dir}/{mdp}-{algorithm}/actions-gamma.pdf')
        elif current_plot == 3:
            fig.set_size_inches(7.2, 6.0)
            fig.suptitle(f'Absolute Value of State Changes at Iteration N \n of {algorithm_title} Solving {mdp_title}')
            fig.text(0.5, 0.025, 'Iteration', ha='center')
            fig.text(0.025, 0.5, 'Absolute Value of State Changes', va='center', rotation='vertical')
            fig.savefig(f'{output_dir}/{mdp}-{algorithm}/diff.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='main.py')
    parser.add_argument('-p', '--problem', help='Problem to solve',
                        choices=['taxi', 'frozen-lake','Blackjack'], default='frozen-lake')
    parser.add_argument('-a', '--algorithm', help='RL algorithm to use',
                        choices=['value-iteration', 'policy-iteration', 'q-learning'], default='policy-iteration')
    args = parser.parse_args()

    algorithm, problem = args.algorithm, args.problem
    log.info(f'{problem} is being solved by {algorithm}')

    if problem == 'frozen-lake':
        env = gym.make('FrozenLake-v0')
    elif problem == 'taxi':
        env = gym.make('Taxi-v2')
    elif problem == 'Blackjack':
        env = gym.make('Blackjack-v0')

    # always seed to the same thing
    np.random.seed(5)
    env.seed(5)

    log.info('Making results directory')
    Path(output_dir).mkdir(exist_ok=True)
    if algorithm == 'value-iteration' or algorithm == 'policy-iteration':
        results = []
        for gamma in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]:
            log.info(f'Running {algorithm} with gamma {gamma}')
            if algorithm == 'value-iteration':
                algo = ValueIteration(env, gamma)
            else:
                algo = PolicyIteration(env, gamma)
            experiment = Experiment(env, algo.get_policy())
            results.append((gamma, experiment.evaluate_policy(), algo.get_iteration_diffs()))
        value_policy_iteration_results(results, mdp=problem, algorithm=algorithm)
    else:
        if problem == 'frozen-lake':
            episodes = 5000
            init_with = [0, 1]
            params = [
                [0.99, 0.8, 0.9999, 0.999, 1],
                [0.99, 0.8, 0.9999, 0.999, 0],
                [0.99, 0.8, 0.5, 0.5, 1],
                # [0.99, 0.8, 0.9999, 0.9, 1],
                [0.99, 0.8, 0.9, 0.999, 1],
                [0.99, 0.5, 0.9999, 0.999, 1],
                [0.9, 0.8, 0.9999, 0.999, 1]
            ]
        else:
            rc('figure', dpi=100)
            episodes = 10000
            init_with = [0, 20]
            params = [
                # gamma, alpha, epsilon, epsilon decay, init with
                [0.99, 0.8, 0.9999, 0.999, 20],
                [0.99, 0.8, 0.9999, 0.999, 0],
                [0.99, 0.8, 0.5, 0.5, 20],
                # [0.99, 0.8, 0.9999, 0.9, 1],
                [0.99, 0.8, 0.9, 0.999, 20],
                [0.99, 0.5, 0.9999, 0.999, 20],
                [0.9, 0.8, 0.9999, 0.999, 20]
            ]
        mdp_title = title[problem]
        algorithm_title = title[algorithm]
        storage = []
        for gamma, alpha, epsilon, epsilon_decay, init_q in params:
            log.info(
                f'Running q-learning with gamma {gamma} alpha {alpha} epsilon {epsilon} epsilon_decay {epsilon_decay} init {init_q}')
            algo = QLearning(env, gamma, alpha, epsilon, epsilon_decay, init_q)
            for episode in range(episodes):
                state = env.reset()
                actions = 0
                rewards = 0
                start = time.perf_counter()
                while True:
                    action = algo.action(state)
                    state_prime, reward, done, _ = env.step(action)
                    algo.learn(state, action, state_prime, reward, done)
                    state = state_prime
                    actions += 1
                    rewards += reward
                    if done:
                        storage.append([str(gamma), str(alpha), str(epsilon), str(epsilon_decay), str(init_q),
                                        episode, actions, rewards, time.perf_counter() - start])
                        break
        episode_storage = pd.DataFrame(columns=['gamma', 'alpha', 'epsilon',
                                                'epsilon_decay', 'init', 'episode', 'actions', 'reward', 'time'],
                                       data=storage)
        # p = episode_storage.reward.rolling(window=100, center=False).mean().dropna()
        # print(p[p >= 0.78].count())

        Path(f'{output_dir}/{problem}-{algorithm}').mkdir(exist_ok=True)
        groups = episode_storage.groupby(['gamma', 'alpha', 'epsilon', 'epsilon_decay', 'init'])
        graphs = [
            plt.subplots(6, num=0, sharex=True, sharey=True),
            plt.subplots(6, num=1, sharex=True, sharey=True),
            plt.subplots(6, num=2, sharex=True, sharey=True)
        ]
        for t, (fig, axes) in enumerate(graphs):
            if problem == 'frozen-lake':
                fig.set_size_inches(6.4, 8)
            else:
                fig.set_size_inches(6.4, 10)
            if t == 0:
                fig.suptitle(f'Rolling Mean of Rewards Over 100 Episodes Solving \n {mdp_title} with {algorithm_title}')
                fig.text(0.5, 0.025, 'Episodes', ha='center')
                fig.text(0.025, 0.5, 'Rolling Mean of Rewards', va='center', rotation='vertical')
            elif t == 1:
                fig.suptitle(
                    f'Average Actions Taken Following the Optimal Policy in \n {mdp_title} with {algorithm_title}')
                fig.text(0.5, 0.025, 'Actions', ha='center')
                # fig.text(0.025, 0.5, 'Actions', va='center', rotation='vertical')
                # fig.set_size_inches(6.4, 7)
            elif t == 2:
                fig.suptitle(
                    f'Actions Taken per Episode Following the Optimal Policy in \n {mdp_title} with {algorithm_title}')
                fig.text(0.5, 0.025, 'Episode', ha='center')
                fig.text(0.025, 0.5, 'Actions', va='center', rotation='vertical')

            for i, (name, group) in enumerate(groups):
                log.info(f'Graphing {name}')
                localized_group = group[['episode', 'actions', 'reward', 'time']]
                current_axis = axes[i]
                current_axis.set_title(
                    f'$\gamma$ = {name[0]}, $\\alpha$ = {name[1]}, $\epsilon$ = {name[2]}, $\epsilon$ decay = {name[3]}, $Q(s,a)_0$ = {name[4]}')
                if t == 0:
                    rewards = localized_group[['reward']].rolling(window=100, center=False).mean()
                    current_axis.plot(range(100, episodes + 1), rewards.dropna())
                    if problem == 'frozen-lake':
                        current_axis.set_ylim([0, 1.0])
                        solve = 0.78
                    else:
                        current_axis.set_ylim([-10, 10])
                        solve = 8.0
                    current_axis.axhline(y=solve, color='r', linestyle='--', label=solve)
                    start_index = localized_group.index[0]
                    for reward_index in rewards.index:
                        item = rewards.loc[reward_index]
                        if item.reward > solve:
                            log.info(
                                f'episodes = {(reward_index - start_index)+ 1} time =  {episode_storage.loc[start_index:reward_index + 1].time.sum()}')
                            break
                    # for row in rewards.iterrows():
                    #    print(row)
                elif t == 1:
                    actions = localized_group.actions
                    current_axis.xaxis.set_ticks_position('none')
                    current_axis.barh(0, actions.mean())
                    current_axis.set_yticks([])
                elif t == 2:
                    actions = localized_group.actions
                    current_axis.plot(range(0 + 1, episodes + 1), actions)

            fig.subplots_adjust(hspace=0.5)
            if t == 0:
                fig.savefig(f'{output_dir}/{problem}-{algorithm}/rewards.png')
            elif t == 1:
                fig.savefig(f'{output_dir}/{problem}-{algorithm}/actions.png')
            elif t == 2:
                fig.savefig(f'{output_dir}/{problem}-{algorithm}/total-actions.png')

        '''
        for gamma in [0.99, 0.9, 0.8, 0.7, 0.6, 0.5]:
            for epsilon_decay in [0.9999, 0.999, 0.99]:
                for epsilon in [0.99, 0.98, 0.97, 0.96, 0.95]:
                    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5]:
                        for init_q in init_with:
                            # algo = QLearning(env, gamma, 0.8, 0.95, 0.999)
                            log.info(f'Running q-learning with gamma {gamma} alpha {alpha} epsilon {epsilon} epsilon_decay {epsilon_decay} init {init_q}')
                            algo = QLearning(env, gamma, alpha, epsilon, epsilon_decay, init_q)
                            for episode in range(episodes):
                                state = env.reset()
                                actions = 0
                                rewards = 0
                                start = time.perf_counter()
                                while True:
                                    action = algo.action(state)
                                    state_prime, reward, done, _ = env.step(action)
                                    algo.learn(state, action, state_prime, reward, done)
                                    state = state_prime
                                    actions += 1
                                    rewards += reward
                                    if done:
                                        # episode_storage = episode_storage.append([gamma, alpha, epsilon, epsilon_decay, init_q, 
                                          # actions, reward, time.perf_counter() - start], ignore_index=True)
                                        storage.append([gamma, alpha, epsilon, epsilon_decay, init_q, 
                                           actions, reward, time.perf_counter() - start])
                                        # log.info(f'Took {actions} for episode {episode} and got {rewards} rewards')
                                        break
                            # p = episode_storage.reward.rolling(window=100, center=False).mean().dropna()
                            # print(p[p >= 0.78].count())
        episode_storage = pd.DataFrame(columns=['gamma', 'alpha', 'epsilon', 
          'epsilon_decay', 'init', 'actions', 'reward', 'time'], data=storage)
        episode_storage.to_csv(f'./results/{problem}-q-learning.csv')
        '''
    # return pd.DataFrame(scores)[0].rolling(window=100, center=False).mean().dropna().as_matrix()
    # go ahead and make the dir
    # Path(f'./{problem}').mkdir(exist_ok=True)
    # gym.wrappers.Monitor(env, f'./{problem}', force=True)