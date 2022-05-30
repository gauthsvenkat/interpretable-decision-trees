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

To get started, you want to build the docker image and use the command to run the script in the docker image (i.e. the first and last step)
Please note that the docker image is configured to run in daemon mode, so to check logs you need to use docker logs {container name}
Also note that since it is running in daemon mode, you can ssh into the container using docker exec -it {container name} /bin/bash - the source code is located at /root/code/rl_zoo
After that the trained dt models can be found at /root/code/rl_zoo/student/ in the container

Build docker image (CPU):
```
./scripts/build_docker.sh
```

GPU:
```
USE_GPU=True ./scripts/build_docker.sh
```

```

Run script in the docker image:

```
./scripts/run_docker_cpu.sh python -m src.main
```
