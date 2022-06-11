FOR %%G in (Acrobot-v1, CartPole-v1, MountainCar-v0) DO (
    FOR %%H in (1,2,3,4,5) DO (
        FOR %%J in (0.0,0.5,1.0) DO (
            py -W ignore -m src.feature_tester --env_name %%G --max_depth %%H --student_path _depth_%%H --bc_path _depth_%%H --experiment %%G_depth_%%H --rollouts 100 --no_print true --cp %%J --optimal
        )
    )
)

FOR %%G in (Acrobot-v1, CartPole-v1, MountainCar-v0) DO (
    FOR %%H in (1,2,3,4,5) DO (
        py -W ignore -m src.feature_tester --env_name %%G --max_depth %%H --student_path _depth_%%H --bc_path _depth_%%H --experiment %%G_depth_%%H --rollouts 100 --no_print true
    )
)


