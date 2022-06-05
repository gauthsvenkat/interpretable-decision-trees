# Viper adapted from original authors: https://github.com/obastani/viper
# Bastani, O., Pu, Y., & Solar-Lezama, A. (2018). Verifiable reinforcement
# learning via policy extraction. arXiv preprint arXiv:1805.08328.

import os
from pathlib import Path

import gym
import numpy as np

from src.DTPolicy import DTPolicy


def get_rollout(env, policy):
    obs, done = env.reset(), False
    rollout = []

    while not done:
        # Action
        act = policy.predict(np.array([obs]))[0]

        # Step
        next_obs, rew, done, info = env.step(act)

        # Rollout (s, a, r)
        rollout.append((obs, act, rew))

        # Update (and remove LazyFrames)
        obs = np.array(next_obs)

    return rollout


def get_rollouts_as_list_of_lists(env, policy, n_batch_rollouts):
    return [get_rollout(env, policy) for _ in range(n_batch_rollouts)]


def get_rollouts(env, policy, n_batch_rollouts):
    rollouts = []
    for i in range(n_batch_rollouts):
        rollouts.extend(get_rollout(env, policy))
    return rollouts


def _sample(obss, acts, qs, max_samples):
    # Step 1: Compute probabilities
    ps = np.max(qs, axis=1) - np.min(qs, axis=1)
    ps = ps / np.sum(ps)

    idx = np.random.choice(len(obss), size=min(max_samples, np.sum(ps > 0)), p=ps)
    # idx = np.random.choice(len(obss), size=min(max_samples, np.sum(ps > 0)))

    return obss[idx], acts[idx], qs[idx]


def test_policy(env, policy, n_test_rollouts):
    cum_rew = 0.0
    for i in range(n_test_rollouts):
        student_trace = get_rollout(env, policy)
        cum_rew += sum((rew for _, _, rew in student_trace))
    return cum_rew / n_test_rollouts


def identify_best_policy(env, policies, n_test_rollouts):
    print('Initial policy count: {}'.format(len(policies)))
    # cut policies by half on each iteration
    while len(policies) > 1:
        # Step 1: Sort policies by current estimated reward
        policies = sorted(policies, key=lambda entry: -entry[1])

        # Step 2: Prune second half of policies
        n_policies = int((len(policies) + 1) / 2)
        print('Current policy count: {}'.format(n_policies))

        # Step 3: build new policies
        new_policies = []
        for i in range(n_policies):
            policy, rew = policies[i]
            new_rew = test_policy(env, policy, n_test_rollouts)
            new_policies.append((policy, new_rew))
            print('Reward update: {} -> {}'.format(rew, new_rew))

        policies = new_policies

    if len(policies) != 1:
        raise Exception()

    return policies[0][0], policies[0][1]


def train_viper(env, student, oracle, max_iters, n_rollouts, train_frac, max_samples, n_test_rollouts):
    # Step 0: setup
    env.reset()

    obss, acts, qs = [], [], []
    students = []

    # Step 1: Generate traces
    trace = get_rollouts(env, oracle, n_rollouts)
    obss.extend((obs for obs, _, _ in trace))
    acts.extend((act for _, act, _ in trace))
    qs.extend(oracle.predict_q(np.array([obs for obs, _, _ in trace])))

    for i in range(max_iters):
        print('Iteration {}/{}'.format(i+1, max_iters))
        # Train from subset of aggregated data
        cur_obss, cur_acts, cur_qs = _sample(np.array(obss), np.array(acts), np.array(qs), max_samples)
        print('Training student with {} points'.format(len(cur_obss)))
        student.train(cur_obss, cur_acts, train_frac)

        # Get new observations

        student_trace = get_rollouts(env, student, n_rollouts)
        student_obss = [obs for obs, _, _ in student_trace]

        oracle_acts = oracle.predict(student_obss)
        oracle_qs = oracle.predict_q(student_obss)

        obss.extend(student_obss)
        acts.extend(oracle_acts)
        qs.extend(oracle_qs)

        cur_rew = sum((rew for _, _, rew in student_trace)) / n_rollouts

        print('Student reward: {}'.format(cur_rew))

        students.append((student.clone(), cur_rew))

    env.close()
    return identify_best_policy(env, students, n_test_rollouts)


class ViperEnvConfig:
    def __init__(self, student_max_depth, viper_max_iters, viper_max_rollouts, viper_train_frac, viper_max_samples,
                 viper_n_test_rollouts):
        self.student_max_depth = student_max_depth
        self.viper_max_iters = viper_max_iters
        self.viper_max_rollouts = viper_max_rollouts
        self.viper_train_frac = viper_train_frac
        self.viper_max_samples = viper_max_samples
        self.viper_n_test_rollouts = viper_n_test_rollouts

    @staticmethod
    def get_viper_config(env_name):
        if env_name == 'MountainCar-v0':
            return ViperEnvConfig(
                student_max_depth=1,
                viper_max_iters=20,
                viper_max_rollouts=100,
                viper_train_frac=0.8,
                viper_max_samples=40000,
                viper_n_test_rollouts=100
            )
        if env_name == 'CartPole-v1':
            return ViperEnvConfig(
                student_max_depth=3,
                viper_max_iters=20,
                viper_max_rollouts=100,
                viper_train_frac=0.8,
                viper_max_samples=80000,
                viper_n_test_rollouts=100
            )
        if env_name == 'Acrobot-v1':
            return ViperEnvConfig(
                student_max_depth=1,
                viper_max_iters=20,
                viper_max_rollouts=100,
                viper_train_frac=0.8,
                viper_max_samples=80000,
                viper_n_test_rollouts=100
            )

        return ViperEnvConfig(
            student_max_depth=4,
            viper_max_iters=60,
            viper_max_rollouts=60,
            viper_train_frac=0.8,
            viper_max_samples=200000,
            viper_n_test_rollouts=100
        )


def get_student(env, oracle, train=True, save_path_specifier="",depth=-1):
    dt_save_folder = Path('student', env.unwrapped.spec.id)
    os.makedirs(str(dt_save_folder), exist_ok=True)
    config = ViperEnvConfig.get_viper_config(env.unwrapped.spec.id)
    if(depth != -1):
        config.student_max_depth = depth
    
    if train:
        student = DTPolicy(config.student_max_depth, optimal_tree=optimal_tree)
        student, _ = train_viper(env, student, oracle, config.viper_max_iters, config.viper_max_rollouts,
                                 config.viper_train_frac, config.viper_max_samples, config.viper_n_test_rollouts)
        student.save_dt_policy(Path(dt_save_folder, 'policy{}.pk'.format(save_path_specifier)))
        student.save_dt_policy_viz(Path(dt_save_folder, 'policy{}.png'.format(save_path_specifier)))
    else:
        student = DTPolicy.load_dt_policy(Path(dt_save_folder, 'policy{}.pk'.format(save_path_specifier)))
        student.save_dt_policy_viz(Path(dt_save_folder, 'policy{}.png'.format(save_path_specifier)))

    return student
