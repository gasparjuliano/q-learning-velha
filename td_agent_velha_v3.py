#!/usr/bin/env python
import os
import sys
import random
import time
import logging
import json
from collections import defaultdict, namedtuple
from itertools import product
from multiprocessing import Pool
from tempfile import NamedTemporaryFile

import pickle as pkl
import numpy as np
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler

import pandas as pd
import click
from tqdm import tqdm as _tqdm
tqdm = _tqdm

from env_velha import TicTacToeEnv, set_log_level_by, agent_by_mark,\
    next_mark, check_game_status, after_action_state, O_REWARD, X_REWARD
from human_agent import HumanAgent
from base_agent import BaseAgent


DEFAULT_VALUE = 0
EPISODE_CNT = 10000
BENCH_EPISODE_CNT = 3000
MODEL_FILE = 'best_td_agent.dat'
EPSILON = 0.08
ALPHA = 0.4
CWD = os.path.dirname(os.path.abspath(__file__))

# Código para os plots do código principal
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])

st_values = {}
st_visits = defaultdict(lambda: 0)


def reset_state_values():
    global st_values, st_visits
    st_values = {}
    st_visits = defaultdict(lambda: 0)


def set_state_value(state, value):
    st_visits[state] += 1
    st_values[state] = value


def best_val_indices(values, fn):
    best = fn(values)
    return [i for i, v in enumerate(values) if v == best]


class Estimator:
    """
    O aproximador da função de valor
    """

    def __init__(self, env):
        # Pré-processamento: padronização (média zero e variância unitária)
        # Utiliza alguns exemplos de observações para isso
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        self.scaler = sklearn.preprocessing.StandardScaler()
        observation_examples = np.expand_dims(observation_examples, 1)
        self.scaler.fit(observation_examples)

        # O estado é convertido para uma representação de mais alta dimensionalidade via kernels RBF (a estilo da SVM)
        # Uma combinação de diferentes kernels é utilizada, com diferentes variâncias
        self.featurizer = sklearn.pipeline.FeatureUnion([
            ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
            ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
            ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
            ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(self.scaler.transform(observation_examples))

        # Um modelo separado é criado para cada ação no espaço de ações
        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            # O método partial_fit é utilizado para inicializar o modelo
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):
        """
        Retorna a representação de um estado no espaço de características
        """
        state = np.array(state[0]).reshape(-1, 1)
        scaled = self.scaler.transform(state)
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, s, a=None):
        """
        Faz predições da função de valor.

        Se uma ação a foi passada, retorna um único número de predição
        Se nenhuma ação foi passada, retorna um vetor para todas as ações naquele estado

        """
        features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([features])[0] for m in self.models])
        else:
            return self.models[a].predict([features])[0]

    def update(self, s, a, y):
        """
        Realiza o update dos parâmetros do estimador para um dado (estado, ação) com relação a y
        """
        features = self.featurize_state(s)
        #print(f'features {features} \t\t y {y}')
        self.models[a].partial_fit([features], [y])


def ajusta_q_value(q_values, available_actions):
    # carrega lista com todos os espaços do cenário
    cenario = [n for n in range(len(q_values))]

    casas_ocupadas = list(set(cenario) - set(available_actions))

    # só coloca -infinito quando ainda sobrar pelo menos 1 casas sem jogada
    if len(casas_ocupadas) <= (len(q_values) - 1):
        for c in casas_ocupadas:
            # coloca -infinito no q_values das casas já jogadas
            # para que não sejam escolhidas como best_action
            q_values[c] = -np.inf

    return q_values


class TDAgent(object):
    def __init__(self, mark, epsilon, alpha, estimator):
        self.mark = mark
        self.alpha = alpha
        self.epsilon = epsilon
        self.episode_rate = 1.0
        self.estimator = estimator

    #Observação: nA = env.action_space.n
    def make_epsilon_greedy_policy(self, epsilon, nA):
        """
        Cria uma política epsilon-greedy baseado em uma função de valor Q aproximada e um epsilon. Retorna as
        probabilidades de cada ação
        """

        def policy_fn(observation, ava_actions):
            A = np.ones(nA, dtype=float) * epsilon / nA
            q_values = self.estimator.predict(observation)

            q_values = ajusta_q_value(q_values, ava_actions)

            best_action = np.argmax(q_values)
            A[best_action] += (1.0 - epsilon)
            return A

        return policy_fn

    def act(self, state, ava_actions):
        #return self.egreedy_policy(state, ava_actions)
        action_probs = self.policy(state, ava_actions)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action

    def act_play(self, state, ava_actions, nA, estimator):
        #return self.egreedy_policy(state, ava_actions)

        #policy
        A = np.ones(nA, dtype=float) * self.epsilon / nA
        q_values = estimator.predict(state)
        q_values = ajusta_q_value(q_values, ava_actions)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - self.epsilon)
        action_probs = A

        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

        return action
    # def egreedy_policy(self, state, ava_actions):
    #     """Returns action by Epsilon greedy policy.
    #
    #     Return random action with epsilon probability or best action.
    #
    #     Args:
    #         state (tuple): Board status + mark
    #         ava_actions (list): Available actions
    #
    #     Returns:
    #         int: Selected action.
    #     """
    #     logging.debug("egreedy_policy for '{}'".format(self.mark))
    #     e = random.random()
    #     if e < self.epsilon * self.episode_rate:
    #         logging.debug("Explore with eps {}".format(self.epsilon))
    #         action = self.random_action(ava_actions)
    #     else:
    #         logging.debug("Exploit with eps {}".format(self.epsilon))
    #         action = self.greedy_action(state, ava_actions)
    #     return action
    #
    # def random_action(self, ava_actions):
    #     return random.choice(ava_actions)
    #
    # def greedy_action(self, state, ava_actions):
    #     """Return best action by current state value.
    #
    #     Evaluate each action, select best one. Tie-breaking is random.
    #
    #     Args:
    #         state (tuple): Board status + mark
    #         ava_actions (list): Available actions
    #
    #     Returns:
    #         int: Selected action
    #     """
    #     assert len(ava_actions) > 0
    #
    #     ava_values = []
    #     for action in ava_actions:
    #         nstate = after_action_state(state, action)
    #         nval = self.ask_value(nstate)
    #         ava_values.append(nval)
    #         vcnt = st_visits[nstate]
    #         logging.debug("  nstate {} val {:0.2f} visits {}".
    #                       format(nstate, nval, vcnt))
    #
    #     # select most right action for 'O' or 'X'
    #     if self.mark == 'O':
    #         indices = best_val_indices(ava_values, max)
    #     else:
    #         indices = best_val_indices(ava_values, min)
    #
    #     # tie breaking by random choice
    #     aidx = random.choice(indices)
    #     logging.debug("greedy_action mark {} ava_values {} indices {} aidx {}".
    #                   format(self.mark, ava_values, indices, aidx))
    #
    #     action = ava_actions[aidx]
    #
    #     return action

    def ask_value(self, state):
        """Returns value of given state.

        If state is not exists, set it as default value.

        Args:
            state (tuple): State.

        Returns:
            float: Value of a state.
        """
        if state not in st_values:
            logging.debug("ask_value - new state {}".format(state))
            gstatus = check_game_status(state[0])
            val = DEFAULT_VALUE
            # win
            if gstatus > 0:
                val = O_REWARD if self.mark == 'O' else X_REWARD
            set_state_value(state, val)
        return st_values[state]

    def backup(self, state, nstate, ava_actions, reward, gamma, action):
        # Update do TD
        q_values_next = self.estimator.predict(nstate)
        q_values_next = ajusta_q_value(q_values_next, ava_actions)

        # Q-value para o TD Target
        td_target = reward + gamma * np.max(q_values_next)
        # Atualiza o aproximador usando o td_target
        self.estimator.update(state, action, td_target)
        """Backup value by difference and step size.

        Execute an action then backup Q by best value of next state.

        Args:
            state (tuple): Current state
            nstate (tuple): Next state
            reward (int): Immediate reward from action
        """
        # logging.debug("backup state {} nstate {} reward {}".
        #               format(state, nstate, reward))
        #
        # val = self.ask_value(state)
        # nval = self.ask_value(nstate)
        # diff = nval - val
        # val2 = val + self.alpha * diff
        #
        # logging.debug("  value from {:0.2f} to {:0.2f}".format(val, val2))
        # set_state_value(state, val2)


@click.group()
@click.option('-v', '--verbose', count=True, help="Increase verbosity.")
@click.pass_context
def cli(ctx, verbose):
    global tqdm

    set_log_level_by(verbose)
    if verbose > 0:
        tqdm = lambda x: x  # NOQA


@cli.command(help="Learn and save the model.")
@click.option('-p', '--episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--save-file', default=MODEL_FILE, show_default=True,
              help="Save model data as file name.")
def learn(max_episode, epsilon, alpha, save_file, epsilon_decay=1.0):
    _learn(max_episode, epsilon, alpha, save_file, epsilon_decay=1.0)


def _learn(max_episode, epsilon, alpha, save_file, epsilon_decay=1.0, gamma=1.0):
    """Learn by episodes.

    Make two TD agent, and repeat self play for given episode count.
    Update state values as reward coming from the environment.

    Args:
        max_episode (int): Episode count.
        epsilon (float): Probability of exploration.
        alpha (float): Step size.
        save_file: File name to save result.
    """
    reset_state_values()

    env = TicTacToeEnv()

    agents = [TDAgent('O', epsilon, alpha, Estimator(env)),
              TDAgent('X', epsilon, alpha, Estimator(env))]

    jog1wins = 0
    jog2wins = 0
    empates = 0
    # Mantém as estatísticas
    stats = EpisodeStats(
        episode_lengths=np.zeros(max_episode),
        episode_rewards=np.zeros(max_episode))

    start_mark = 'O'
    for i in tqdm(range(max_episode)):
        episode = i + 1
        #env.show_episode(False, episode)

        # reset agent for new episode
        for agent in agents:
            agent.episode_rate = episode / float(max_episode)
            # A política greedy com base no estado atual do estimador
            agent.policy = agent.make_epsilon_greedy_policy(
                epsilon * epsilon_decay ** i, env.action_space.n)

        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        t = 0
        while not done:
            t += 1
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            env.show_turn(False, mark)

            action = agent.act(state, ava_actions)

            # update (no rendering)
            nstate, reward, done, info = env.step(action)

            # Atualiza as estatísticas
            stats.episode_rewards[i] += reward
            stats.episode_lengths[i] = t

            #agent.backup(state, nstate, reward)
            agent.backup(state, nstate, ava_actions, reward, gamma, action)

            if done:
                status = env.show_result(False, mark, reward)
                if status == 0:
                    empates += 1
                elif status == 1:
                    jog1wins += 1
                else:
                    jog2wins += 1

                if i % 100 == 0:
                    print(f'Episódio: {i+1} \t Empates = {empates/(i+1)*100:.2f}% \t ' \
                          f'Vitórias: jog1 = {jog1wins/(i+1)*100:.2f}% \t ' \
                          f'jog2 = {jog2wins/(i+1)*100:.2f}%')

                # set terminal state value
                set_state_value(state, reward)

            _, mark = state = nstate

        # rotate start
        start_mark = next_mark(start_mark)

    # save states
    save_model(save_file, max_episode, epsilon, alpha)

    #salva o estimator do jogador de melhor performance
    if jog1wins > jog2wins:
        agent = agent_by_mark(agents, 'O')
    else:
        agent = agent_by_mark(agents, 'X')
    save_estimator_model('estimator.pkl', agent.estimator)

def save_model(save_file, max_episode, epsilon, alpha):
    with open(save_file, 'wt') as f:
        # write model info
        info = dict(type="td", max_episode=max_episode, epsilon=epsilon,
                    alpha=alpha)
        # write state values
        f.write('{}\n'.format(json.dumps(info)))
        for state, value in st_values.items():
            vcnt = st_visits[state]
            f.write('{}\t{:0.3f}\t{}\n'.format(state, value, vcnt))

def save_estimator_model(save_file, estimator):
    #salvando o modelo do estimator
    with open(save_file, 'wb') as f:
        pkl.dump(estimator, f)

def load_model(filename):
    with open(filename, 'rb') as f:
        # read model info
        info = json.loads(f.readline().decode('ascii'))
        for line in f:
            elms = line.decode('ascii').split('\t')
            state = eval(elms[0])
            val = eval(elms[1])
            vcnt = eval(elms[2])
            st_values[state] = val
            st_visits[state] = vcnt
    return info

def load_estimator_model(filename):
    with open(filename, 'rb') as f:
        return pkl.load(f)

@cli.command(help="Play with human.")
@click.option('-f', '--load-file', default=MODEL_FILE, show_default=True,
              help="Load file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def play(load_file, show_number):
    _play(load_file, HumanAgent('O'), show_number)


def _play(load_file, vs_agent, show_number):
    """Play with learned model.

    Make TD agent and adversarial agnet to play with.
    Play and switch starting mark when the game finished.
    TD agent behave no exploring action while in play mode.

    Args:
        load_file (str):
        vs_agent (object): Enemy agent of TD agent.
        show_number (bool): Whether show grid number for visual hint.
    """
    load_model(load_file)
    env = TicTacToeEnv(show_number=show_number)

    estimator = load_estimator_model('estimator.pkl')

    td_agent = TDAgent('X', 0, 0, estimator)  # prevent exploring
    start_mark = 'O'
    agents = [vs_agent, td_agent]

    while True:
        # start agent rotation
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False

        # show start board for human agent
        if mark == 'O':
            env.render(mode='human')

        while not done:
            agent = agent_by_mark(agents, mark)
            human = isinstance(agent, HumanAgent)

            env.show_turn(True, mark)
            ava_actions = env.available_actions()
            if human:
                action = agent.act(ava_actions)
                if action is None:
                    sys.exit()
            else:
                #action = agent.act(state, ava_actions)
                action = agent.act_play(state, ava_actions, env.action_space.n, estimator)
            state, reward, done, info = env.step(action)

            env.render(mode='human')
            if done:
                env.show_result(True, mark, reward)
                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)


@cli.command(help="Learn and benchmark.")
@click.option('-p', '--learn-episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Learn episode count.")
@click.option('-b', '--bench-episode', "max_bench_episode",
              default=BENCH_EPISODE_CNT, show_default=True, help="Bench "
              "episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model data file name.")
def learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file):
    _learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file)


def _learnbench(max_episode, max_bench_episode, epsilon, alpha, model_file,
                show=True):
    if show:
        print("Learning...")
    _learn(max_episode, epsilon, alpha, model_file)
    if show:
        print("Benchmarking...")
    return _bench(max_bench_episode, model_file, show)


@cli.command(help="Benchmark agent with base agent.")
@click.option('-p', '--episode', "max_episode", default=BENCH_EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model data file name.")
def bench(model_file, max_episode):
    _bench(max_episode, model_file)


def _bench(max_episode, model_file, show_result=True):
    """Benchmark given model.

    Args:
        max_episode (int): Episode count to benchmark.
        model_file (str): Learned model file name to benchmark.
        show_result (bool): Output result to stdout.

    Returns:
        (dict): Benchmark result.
    """
    minfo = load_model(model_file)
    agents = [BaseAgent('O'), TDAgent('X', 0, 0)]
    show = False

    start_mark = 'O'
    env = TicTacToeEnv()
    env.set_start_mark(start_mark)

    episode = 0
    results = []
    for i in tqdm(range(max_episode)):
        env.set_start_mark(start_mark)
        state = env.reset()
        _, mark = state
        done = False
        while not done:
            agent = agent_by_mark(agents, mark)
            ava_actions = env.available_actions()
            action = agent.act(state, ava_actions)
            state, reward, done, info = env.step(action)
            if show:
                env.show_turn(True, mark)
                env.render(mode='human')

            if done:
                if show:
                    env.show_result(True, mark, reward)
                results.append(reward)
                break
            else:
                _, mark = state

        # rotation start
        start_mark = next_mark(start_mark)
        episode += 1

    o_win = results.count(1)
    x_win = results.count(-1)
    draw = len(results) - o_win - x_win
    mfile = model_file.replace(CWD + os.sep, '')
    minfo.update(dict(base_win=o_win, td_win=x_win, draw=draw,
                      model_file=mfile))
    result = json.dumps(minfo)

    if show_result:
        print(result)
    return result


@cli.command(help="Learn and play with human.")
@click.option('-p', '--episode', "max_episode", default=EPISODE_CNT,
              show_default=True, help="Episode count.")
@click.option('-e', '--epsilon', "epsilon", default=EPSILON,
              show_default=True, help="Exploring factor.")
@click.option('-a', '--alpha', "alpha", default=ALPHA,
              show_default=True, help="Step size.")
@click.option('-f', '--model-file', default=MODEL_FILE, show_default=True,
              help="Model file name.")
@click.option('-n', '--show-number', is_flag=True, default=False,
              show_default=True, help="Show location number when play.")
def learnplay(max_episode, epsilon, alpha, model_file, show_number):
    _learn(max_episode, epsilon, alpha, model_file)
    _play(model_file, HumanAgent('O'), show_number)


@cli.command(help="Grid search hyper-parameters.")
@click.option('-q', '--quality', type=click.Choice(['high', 'mid', 'low']),
              default='mid', show_default=True, help="Grid search"
              " quality.")
@click.option('-r', '--reproduce-test', "rtest_cnt", default=3,
              show_default=True, help="Reproducibility test count.")
def gridsearch(quality, rtest_cnt):
    """Find and output best hyper-parameters.

    Grid search consists of two phase:
        1. Select best 10 candidates of parameter combination.
        1. Carry out reproduce test and output top 5 parameters.

    Args:
        quality (str): Select preset of parameter combination. High quality
            means more granularity in parameter space.
        rtest_cnt (int): Reproduce test count
    """
    st = time.time()
    _gridsearch_candidate(quality)
    _gridsearch_reproduce(rtest_cnt)
    print("Finished in {:0.2f} seconds".format(time.time() - st))


def _gridsearch_reproduce(rtest_cnt):
    """Refine parameter combination by reproduce test, and output best 5.

    Reproduce test is a learn & bench process from each parameter combination.

    1. Select top 10 parameters from previous step.
    2. Execute reproduce test.
    3. Sort by lose rate.
    4. Output best 5 parameters.

    Args:
        rtest_cnt (int): Reproduce test count

    Todo:
        Apply multiprocessor worker
    """
    print("Reproducibility test.")
    with open(os.path.join(CWD, 'gsmodels/result.json'), 'rt') as fr:
        df = pd.DataFrame([json.loads(line) for line in fr])
        top10_df = df.sort_values(['base_win', 'max_episode'])[:10]

    index = []
    vals = []
    # for each candidate
    pbar = _tqdm(total=len(top10_df) * rtest_cnt)
    for idx, row in top10_df.iterrows():
        index.append(idx)
        base_win_sum = 0
        total_play = 0
        # bench repeatedly
        for i in range(rtest_cnt):
            pbar.update()
            learn_episode = row.max_episode
            epsilon = row.epsilon
            alpha = row.alpha
            with NamedTemporaryFile() as tmp:
                res = _learnbench(learn_episode, BENCH_EPISODE_CNT, epsilon,
                                  alpha, tmp.name, False)
            res = json.loads(res)
            base_win_sum += res['base_win']
            total_play += BENCH_EPISODE_CNT
        lose_pct = float(base_win_sum) / rtest_cnt / total_play * 100
        vals.append(round(lose_pct, 2))

    top10_df['lose_pct'] = pd.Series(vals, index=index)

    df = top10_df.sort_values(['lose_pct', 'max_episode']).reset_index()[:5]
    print(df[['lose_pct', 'max_episode', 'alpha', 'epsilon', 'model_file']])


def _gridsearch_candidate(quality):
    """Select best hyper-parameter candiadates by grid search.

    1. Generate parameter combination by product each parameter space.
    2. Spawn processors to learn & bench each combination.
    3. Wait and write result to a file.

    Args:
        quality (str): Choice among 'high', 'mid', 'low'

    Todo:
        Progress bar estimation is not even.
    """
    # disable sub-process's progressbar
    global tqdm
    tqdm = lambda x: x  # NOQA

    if quality == 'high':
        # high
        epsilons = [e * 0.01 for e in range(8, 25, 2)]
        alphas = [a * 0.1 for a in range(2, 8)]
        episodes = [e for e in range(8000, 31000, 3000)]
    elif quality == 'mid':
        # mid
        epsilons = [e * 0.01 for e in range(10, 20, 5)]
        alphas = [a * 0.1 for a in range(3, 7)]
        episodes = [e for e in range(10000, 30000, 5000)]
    else:
        # low
        epsilons = [e * 0.01 for e in range(9, 13, 2)]
        alphas = [a * 0.1 for a in range(4, 6)]
        episodes = [e for e in range(10000, 25000, 10000)]

    alphas = [round(a, 2) for a in alphas]
    _args = list(product(episodes, epsilons, alphas))
    args = []
    for i, arg in enumerate(_args):
        arg = list(arg)
        arg.insert(1, BENCH_EPISODE_CNT)  # bench episode count
        arg.append(os.path.join(CWD, 'gsmodels/model_{:03d}.dat'.format(i)))
        arg.append(False)  # supress print
        args.append(arg)  # model file name
    prev_left = total = len(args)

    print("Grid search for {} parameter combinations.".format(total))
    pbar = _tqdm(total=total)
    pool = Pool()
    result = pool.starmap_async(_learnbench, args)
    while True:
        if result.ready():
            break
        if prev_left != result._number_left:
            ucnt = prev_left - result._number_left
            pbar.update(ucnt)
            prev_left = result._number_left
        time.sleep(1)

    ucnt = prev_left - result._number_left
    pbar.update(ucnt)
    pbar.close()

    with open(os.path.join(CWD, 'gsmodels/result.json'), 'wt') as f:
        for r in result.get():
            f.write('{}\n'.format(r))


if __name__ == '__main__':
    cli()
