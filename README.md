# Interpretability and Performance of Decision Trees Extracted by Viper

Code used to evaluate the interpretability of decision trees produced by Viper,
an Imitation Learning algorithm[[1]](https://arxiv.org/abs/1805.08328).

Q-Learning code adapted from [https://github.com/guillaumefrd/q-learning-mountain-car](https://github.com/guillaumefrd/q-learning-mountain-car).
Viper code adapted from [https://github.com/obastani/viper](https://github.com/obastani/viper)

## How to run

- Install dependencies from `Pipfile` using `pipenv`
- Run `main.main()` to train Viper and Behavioral Cloning trees. Oracles have been provided, but can be retrained.
- To change environment, set `env_name` to one of `MountainCar-v0`, `Acrobot-v1`, `CartPole-v1`.

### Docker Images

Build docker image (CPU):
```
./scripts/build_docker.sh
```

GPU:
```
USE_GPU=True ./scripts/build_docker.sh
```

Pull built docker image (CPU):
```
docker pull stablebaselines/rl-baselines-zoo-cpu
```

GPU image:
```
docker pull stablebaselines/rl-baselines-zoo
```

Run script in the docker image:

```
./scripts/run_docker_cpu.sh python -m src.main
```
