REM Generating trees for Acrobat with different depth
py -W ignore -m src.main --env_name Acrobot-v1 --max_depth 1 --student_path _depth_1 --bc_path _depth_1
py -W ignore -m src.main --env_name Acrobot-v1 --max_depth 2 --student_path _depth_2 --bc_path _depth_2
py -W ignore -m src.main --env_name Acrobot-v1 --max_depth 3 --student_path _depth_3 --bc_path _depth_3
py -W ignore -m src.main --env_name Acrobot-v1 --max_depth 4 --student_path _depth_4 --bc_path _depth_4
py -W ignore -m src.main --env_name Acrobot-v1 --max_depth 5 --student_path _depth_5 --bc_path _depth_5

REM Generating trees for CartPole with different depth
py -W ignore -m src.main --env_name CartPole-v1 --max_depth 1 --student_path _depth_1 --bc_path _depth_1
py -W ignore -m src.main --env_name CartPole-v1 --max_depth 2 --student_path _depth_2 --bc_path _depth_2
py -W ignore -m src.main --env_name CartPole-v1 --max_depth 3 --student_path _depth_3 --bc_path _depth_3
py -W ignore -m src.main --env_name CartPole-v1 --max_depth 4 --student_path _depth_4 --bc_path _depth_4
py -W ignore -m src.main --env_name CartPole-v1 --max_depth 5 --student_path _depth_5 --bc_path _depth_5


REM Generating trees for MountainCar with different depth
py -W ignore -m src.main --env_name MountainCar-v0 --max_depth 1 --student_path _depth_1 --bc_path _depth_1
py -W ignore -m src.main --env_name MountainCar-v0 --max_depth 2 --student_path _depth_2 --bc_path _depth_2
py -W ignore -m src.main --env_name MountainCar-v0 --max_depth 3 --student_path _depth_3 --bc_path _depth_3
py -W ignore -m src.main --env_name MountainCar-v0 --max_depth 4 --student_path _depth_4 --bc_path _depth_4
py -W ignore -m src.main --env_name MountainCar-v0 --max_depth 5 --student_path _depth_5 --bc_path _depth_5