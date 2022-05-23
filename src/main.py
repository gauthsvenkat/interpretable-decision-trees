import gym
import argparse

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
    student = viper.get_student(env, oracle, train=True, save_path_specifier="")
    bc = behavioralCloning.get_student(env, oracle, train=True, save_path_specifier="")

    if env_name == 'MountainCar-v0':
        simple = SimpleMCDT()
    elif env_name == 'CartPole-v1':
        simple = SimpleCartPoleDT()
    elif env_name == 'Acrobot-v1':
        simple = SimpleAcrobotDT()
    
    e = Evaluate(env, oracle, [student, bc, simple], n_rollouts=200, policy_names=["Student", "Bc", "Simple"])

    print(e.evaluate())
    play(env, oracle)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and evaluate the expert and student policies')
    parser.add_argument('--env_name', type=str, default='Acrobot-v1', help='Name of the environment to use')
    
    args = parser.parse_args()
    main(args)

