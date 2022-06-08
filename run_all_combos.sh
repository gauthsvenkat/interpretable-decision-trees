#!/bin/bash

for env in "MountainCar-v0" "CartPole-v1" "Acrobot-v1"; do
    for i in {1..5}; do
        for cp in 0.5 1; do
            ./scripts/run_docker.sh python -m src.main --optimal_tree --env_name ${env} --max_depth ${i} --cp ${cp} --student_path _depth_${i} --bc_path _depth_${i} 2>&1 | tee ./logs/${env}_depth${i}_cp${cp}.log
        done
    done
done