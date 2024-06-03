# Training and Validation

## Training
### Setting up the experiment configuration

We will define a configuration file for training a model. In it, we will define the typical training hyperparameters, as well as network setup and training data to use. All of the available configurations are in the [`configs/_global.py`](./configs/_global.py), so if more should be added for training or evaluation, they should also be added there. 

Some experiments are provided in the `configs` directory.

TBD

### Training an experiment

To train the defined experiment above, we can do so either with a single GPU or multiple GPUs. If available, it's recommended to use multiple GPUs in order to accelerate the training.

For a single GPU with ID `0`, we run:

```bash
python3 main.py --process-type=train_val --gpus 0 --folder CILv2 --exp CILv2_3cam_smalltest
```

For multiple GPUs, we need their IDs and total number of GPUs. For example, in a cluster with many GPUs, we wish to use three GPUs with IDs `0,3,5`, so we run:

```bash
torchrun --nproc_per_node=3 main.py --gpus 0,3,5 --folder CILv2 --exp CILv2_3cam_smalltest
```

Results will be saved in the `TRAINING_RESULTS_ROOT` directory, separated by `folder` and `experiment`, that is, `$TRAINING_RESULTS_ROOT/_results/{folder}/{exp}`. 

## Validation

Finally, in order to know the generalization capabilities of your model, you must run some real-time driving of it in unseen environments and new weather conditions. We have available two benchmarks, although more could be defined in other maps or even weather environments.

> [!NOTE]
> Both benchmarks are run in [synchronous mode](https://carla.readthedocs.io/en/0.9.13/adv_synchrony_timestep/#setting-synchronous-mode). Please refer to CARLA's documentation for more information on this setting. 


### `NoCrash` benchmark

For models trained with single lane data, we typically test their capabilites using the `NoCrash` benchmark defined in the [CILRS paper](https://arxiv.org/abs/1904.08980). Basically, we define some routes that the model should complete within a certain time limit, and should a crash be detected, we stop the 

To run the `NoCrash` benchmark, we have defined some scripts. There are two maps available, each with different scenarios (number of agents roaming in the map):

| | **`Town01`** | **`Town02`** |
| --- |   ---    |    ---   |
| **Scenario** | `empty`, `regular`, `dense` | `empty`, `regular`, `busy` |

Thus, if we wish to run the `NoCrash` benchmark in `Town01` under new weather conditions with a `regular` scenario for the checkpoint `45` of the `CILv2_3cam_smalltest` experiment above, using the GPU with ID `0`, we run the following:

```bash
./run_CARLA_driving/scripts/run_evaluation/CILv2/nocrash_newweather_Town01_lbc.sh 0 CILv2 CILv2_3cam_smalltest 45 regular
```

Note that this allows you to run multiple checkpoints in parallel using different GPUs, should they be available. The results of the driving will be saved in `./run_CARLA_driving/results` (Route Completion, Driving Score, etc.). By default, a random seed of `0` will be used to spawn and control the other agents, but we can change this to another one by e.g. adding `--random-seed 42` in the above command. 

If you wish to save the driving during this evaluation (such as the one in the [demo video](#demo-video), although that one showcases a route of the [offline Leaderboard](#offline-leaderboard) found below), then you also need to add the flag  `--save-driving-vision`. This will save each frame to the `$SENSOR_SAVE_PATH` defined above. Naturally, this will make the driving run much slower, so use it for models you wish to better visualize.

For running the benchmark in `Town02`, the command will be the same, but the script to be run is:

```bash
./run_CARLA_driving/scripts/run_evaluation/CILv2/nocrash_newweathertown_Town02_lbc.sh 0 CILv2 CLIv2_3cam_smalltest 45 regular --random-seed 42 --save-driving-vision
```

### Offline Leaderboard benchmark

For models trained with multi-lane data (such as `Town03` onwards), it'd be more interesting to test the model in more complex scenarios than those found in the `NoCrash` benchmark above. For these, we have adapted the [offline Leaderboard benchmark](https://github.com/zhejz/carla-roach?tab=readme-ov-file#the-offline-leaderboard) for validating and testing the models under new weather conditions. Note that the available towns are `Town03` and `Town05` and there is a different script for each (to keep things separate, but they could be joined into the same script). As above, we run the following:

```bash
./run_CARLA_driving/scripts/leaderboard_Town05.py 0 CILv2 CILv2_3cam_smalltest 45
```

Note that the number of pedestrians and vehicles is now fixed to the same amount found in the Leaderboard. As before, the results will be saved in `./run_CARLA_driving/results`, and we can change the random seed or save the driving with the respective commands: `--random-seed=42` and `--save-driving-vision`. A complete route is shown in the [demo video](../README.md#demo-video).
