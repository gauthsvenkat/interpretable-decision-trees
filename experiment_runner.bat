REM Generating trees for Acrobat with different depth
py -W ignore -m src.feature_tester --env_name Acrobot-v1 --max_depth 1 --student_path _depth_1 --bc_path _depth_1 --experiment acrobot_depth_1 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name Acrobot-v1 --max_depth 2 --student_path _depth_2 --bc_path _depth_2 --experiment acrobot_depth_2 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name Acrobot-v1 --max_depth 3 --student_path _depth_3 --bc_path _depth_3 --experiment acrobot_depth_3 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name Acrobot-v1 --max_depth 4 --student_path _depth_4 --bc_path _depth_4 --experiment acrobot_depth_4 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name Acrobot-v1 --max_depth 5 --student_path _depth_5 --bc_path _depth_5 --experiment acrobot_depth_5 --rollouts 200 --no_print True

REM Generating trees for CartPole with different depth
py -W ignore -m src.feature_tester --env_name CartPole-v1 --max_depth 1 --student_path _depth_1 --bc_path _depth_1 --experiment cartpole_depth_1 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name CartPole-v1 --max_depth 2 --student_path _depth_2 --bc_path _depth_2 --experiment cartpole_depth_2 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name CartPole-v1 --max_depth 3 --student_path _depth_3 --bc_path _depth_3 --experiment cartpole_depth_3 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name CartPole-v1 --max_depth 4 --student_path _depth_4 --bc_path _depth_4 --experiment cartpole_depth_4 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --env_name CartPole-v1 --max_depth 5 --student_path _depth_5 --bc_path _depth_5 --experiment cartpole_depth_5 --rollouts 200 --no_print True


REM Generating trees for MountainCar with different depth
py -W ignore -m src.feature_tester --max_depth 1 --env_name MountainCar-v0 --student_path _depth_1 --bc_path _depth_1 --experiment mountaincar_depth_1 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --max_depth 2 --env_name MountainCar-v0 --student_path _depth_2 --bc_path _depth_2 --experiment mountaincar_depth_2 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --max_depth 3 --env_name MountainCar-v0 --student_path _depth_3 --bc_path _depth_3 --experiment mountaincar_depth_3 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --max_depth 4 --env_name MountainCar-v0 --student_path _depth_4 --bc_path _depth_4 --experiment mountaincar_depth_4 --rollouts 200 --no_print True
py -W ignore -m src.feature_tester --max_depth 5 --env_name MountainCar-v0 --student_path _depth_5 --bc_path _depth_5 --experiment mountaincar_depth_5 --rollouts 200 --no_print True