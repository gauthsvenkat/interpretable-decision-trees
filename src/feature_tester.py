#file for experimenting features
import gym
import argparse
import os
from src.DTPolicy import SimpleAcrobotDT, SimpleMCDT, SimpleCartPoleDT
from src import viper, behavioralCloning, Baseline
from src.evaluate import Evaluate


def play(env, policy):
    for _ in range(10):
        state = env.reset()
        done = False
        while not done:

            state_adj = state

            action = policy.predict([state_adj])[0]
            env.render()
            state2, reward, done, info = env.step(action)

            state = state2


def main(args):

    env_name = args.env_name
    env = gym.make(env_name)
    env.reset()

    oracle = Baseline.BaselineOracle(env_name, specifier="")
    student = viper.get_student(env, oracle, train=False, save_path_specifier=args.student_path,depth=args.max_depth,optimal_tree=args.optimal_tree)
    bc = behavioralCloning.get_student(env, oracle, train=False, save_path_specifier=args.bc_path,depth=args.max_depth,optimal_tree=args.optimal_tree)

    if env_name == 'MountainCar-v0':
        simple = SimpleMCDT()
    elif env_name == 'CartPole-v1':
        simple = SimpleCartPoleDT()
    elif env_name == 'Acrobot-v1':
        simple = SimpleAcrobotDT()
        
    e = Evaluate(env, oracle, [student, bc, simple], n_rollouts=args.rollouts, policy_names=["Student", "Bc"], experiment = args.experiment, no_print=args.no_print, optimal = args.optimal_tree)

    e.evaluate()
    #play(env, oracle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the expert and student policies')
    parser.add_argument('--optimal_tree', action='store_true', help='Play the environment with the generated policies')
    parser.add_argument('--no_print',type=bool, default=False, help='Set to true to not print results')
    parser.add_argument('--env_name', type=str, default='Acrobot-v1', help='Name of the environment to use')
    parser.add_argument('--rollouts', type=int, default=200, help='How many rollouts to use')
    parser.add_argument('--max_depth', type=int, default=1, help='The max depth for the generated trees')
    parser.add_argument('--experiment',type=str,default="",help='pass the <environment name>_depth_<depth>')
    parser.add_argument('--student_path',type=str, default="", help='the policy name for the generated student tree. Use something like _depth_<VALUE>')
    parser.add_argument('--bc_path',type=str, default="",help='the policy name for the generated bc tree. Use something like _depth_<VALUE>')
    os. environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    args = parser.parse_args()
    main(args)
