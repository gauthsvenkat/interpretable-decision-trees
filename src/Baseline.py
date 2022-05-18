import os
from pathlib import Path

import gym
from stable_baselines.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN

from src.utils import create_test_env

PARAMS = {
    "CartPole-v1": {
        'buffer': 50000,
        'learning_starts': 1000,
        'exp_frac': 0.1,
        'exp_final_eps': 0.02,
        'lr': 1e-3,
        'prio_replay': True,
        'param_noise': False,
        'N': 50000,
        'target_network_update_freq': 500,
        'train_freq': 1,
    },
    "MountainCar-v0": {
        'buffer': 50000,
        'exp_frac': 0.1,
        'exp_final_eps': 0.1,
        'lr': 1e-4,
        'prio_replay': False,
        'param_noise': True,
        'N': 2000000,
        'learning_starts': 1000,
        'reward_threshold': -180,
        'target_network_update_freq': 500,
        'train_freq': 1,
    },
    "Acrobot-v1": {
        'buffer': 50000,
        'exp_frac': 0.1,
        'exp_final_eps': 0.02,
        'lr': 1e-3,
        'prio_replay': True,
        'param_noise': False,
        'N': 100000,
        'learning_starts': 1000,
        'target_network_update_freq': 500,
        'train_freq': 1,
    },
}


class BaselineOracle:
    def __init__(self, env_name, specifier=""):
        path = Path("/Users/rmadhwal/IdeaProjects/ViperInterpretability/DQN", env_name + specifier)
        try:
            self.model = DQN.load(str(path))
        except ValueError:
            p = PARAMS[env_name]
            env = gym.make(env_name)
            self.model = DQN(MlpPolicy, env,
                             verbose=1,
                             buffer_size=p['buffer'],
                             exploration_fraction=p['exp_frac'],
                             exploration_final_eps=p['exp_final_eps'],
                             learning_rate=p['lr'],
                             prioritized_replay=p['prio_replay'],
                             param_noise=p['param_noise'],
                             learning_starts=p['learning_starts'],
                             target_network_update_freq=p['target_network_update_freq'],
                             train_freq=p['train_freq'],
                             double_q=False)

            print('Learning model for {}'.format(env_name))
            if 'reward_threshold' in p:
                callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=p['reward_threshold'], verbose=1)
                eval_callback = EvalCallback(env, callback_on_new_best=callback_on_best, verbose=1, eval_freq=1500)
                self.model.learn(total_timesteps=p['N'], callback=eval_callback)
            else:
                self.model.learn(total_timesteps=p['N'])
            os.makedirs(path.parent, exist_ok=True)
            self.model.save(str(path))

    def predict(self, obss):
        return [self.model.predict(obs)[0] for obs in obss]

    def predict_q(self, obss):
        return [(self.model.action_probability(obs)) for obs in obss]
