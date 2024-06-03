# Data Collection

## Getting the expert driver

We will use the [Roach](https://github.com/zhejz/carla-roach) Reinforcement Learning (RL) Coach agent to collect our training data. Note that this agent has access to privileged information, so we must be careful to check the data afterwards to make sure no causal confusion may occur with our model (such as stopping due to a red traffic light that the RGB cameras aren't able to see).

Download the checkpoint for the model available in their repository [here](https://github.com/zhejz/carla-roach?tab=readme-ov-file#trained-models). This must be placed in the `TRAINING_RESULTS_ROOT` path that was defined during the [environment setup](../README.md#setting-environment-variables). We provide this subdirectory, but should there be a need (or if the `TRAINING_RESULTS_ROOT` variable be different than the one we have defined above), then it can be created like so:

```bash
mkdir -p $TRAINING_RESULTS_ROOT/_results/Roach_rl_birdview/checkpoints
```

To download the Roach RL agent checkpoint, run:

```bash
wget https://api.wandb.ai/files/iccv21-roach/trained-models/1929isj0/ckpt/ckpt_11833344.pth -P $TRAINING_RESULTS_ROOT/_results/Roach_rl_birdview/checkpoints 
```

We also provide a configuration for the above checkpoint, whose name contains the checkpoint number: `config11833344.json`. This file (and all similar ones obtained during training of a model) will contain the following:

```json
# TODO
```

This way, all that is needed now is to set the experiments (maps, sensors, vehicles, pedestrians, weather conditions) and run the proper scripts for data collection.

## Setting the experiments

TODO