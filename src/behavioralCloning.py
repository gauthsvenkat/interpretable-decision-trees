import os
from pathlib import Path

import numpy as np

from src.DTPolicy import DTPolicy
from src.viper import get_rollouts, ViperEnvConfig


def get_student(env, oracle, train=True, save_path_specifier=""):
    n_rollouts = 200
    dt_save_folder = Path('bc', env.unwrapped.spec.id)
    os.makedirs(str(dt_save_folder), exist_ok=True)

    config = ViperEnvConfig.get_viper_config(env.unwrapped.spec.id)

    if train:
        trace = get_rollouts(env, oracle, n_rollouts)
        obss = np.array([obs for obs, _, _ in trace])
        acts = np.array([act for _, act, _ in trace])
        student = DTPolicy(config.student_max_depth)
        print("Training Behavioral Cloning tree with {} points".format(len(obss)))

        student.train(obss, acts, config.viper_train_frac)
        student.save_dt_policy(Path(dt_save_folder, 'policy{}.pk'.format(save_path_specifier)))
        student.save_dt_policy_viz(Path(dt_save_folder, 'policy.png'))
    else:
        student = DTPolicy.load_dt_policy(Path(dt_save_folder, 'policy{}.pk'.format(save_path_specifier)))
        student.save_dt_policy_viz(Path(dt_save_folder, 'policy{}.png'.format(save_path_specifier)))

    return student

