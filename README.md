# Guiding Attention in End-to-End Driving Models
Official repository for the paper "Guiding Attention in End-to-End Driving Models". 

<div align="center">
[[Project Page]](https://blog.diegoporres.com/guiding-attention-e2e/) [[Arxiv]](https://arxiv.org/abs/2405.00242)
</div>

-----------------------------------------

## Demo Video
<div align="center">
<video width="1000" controls autoplay loop muted markdown="1">
    <source src="docs/short_drive.mp4" type="video/mp4">
</video>
</div>

-----------------------------------------

## About

Vision-based end-to-end driving models trained by imitation learning can lead to affordable solutions for autonomous driving. However, training these well-performing models usually requires a huge amount of data, while still lacking explicit and intuitive activation maps to reveal the inner workings of these models while driving. In this paper, we study how to guide the attention of these models to improve their driving quality and obtain more intuitive activation maps by adding a loss term during training using salient semantic maps. In contrast to previous work, our method does not require these salient semantic maps to be available during testing time, as well as removing the need to modify the model's architecture to which it is applied. We perform tests using perfect and noisy salient semantic maps with encouraging results in both, the latter of which is inspired by possible errors encountered with real data. Using CIL++ as a representative state-of-the-art model and the CARLA simulator with its standard benchmarks, we conduct experiments that show the effectiveness of our method in training better autonomous driving models, especially when data and computational resources are scarce.

## Getting started

### Requirements

 * **Hardware:** A computer with a dedicated GPU capable of running Unreal Engine.
 * **OS:** This code was developed in Ubuntu 22.04. Previous versions should also work, but no other OS was tested.

### Code and CARLA Installation

The easiest way to get started is with [`miniconda`](https://docs.anaconda.com/free/miniconda/). After installation, clone and move to the root directory of this repository, and run the following lines in a terminal:

```bash
conda create -n cilv2 python=3.7
conda activate cilv2
pip3 install -r requirements.txt
```

Another requirement of the code is to have CARLA `0.9.13` installed locally and its `docker` container. For the former, you can download (and extract) the simulator [here](https://github.com/carla-simulator/carla/releases/tag/0.9.13). For the latter, make sure you have properly installed `docker` and run the following in a terminal:

```bash
docker pull carlasim/carla:0.9.13
```

Remember to follow the [post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/) to remove the need for `sudo`, else our code won't be able to run when collecting data or running evaluations.

### Setting Environment Variables

To run the code, we will need to set some environment variables. Make sure you are *in the root of this repository*:

```bash
export CARLAHOME=/path/to/local/carla  # Root of the local CARLA 0.9.13
export PYTHONPATH=${CARLAHOME}/PythonAPI/carla/:${CARLAHOME}/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg:`pwd`/run_CARLA_driving:`pwd`/scenario_runner:`pwd`
export TRAINING_RESULTS_ROOT=`pwd`/../VisionTFM
export DATASET_PATH=/path/to/dataset/root  # Can be general and complemented in the `config.yml` file for the experiment
export SENSOR_SAVE_PATH=${DATASET_PATH}/driving_record  # Recommended to save the evaluation next to your dataset
export DRIVING_TEST_ROOT=`pwd`/run_CARLA_driving/

```

## Data Collection

TBD

## Data Preparation

After collecting a dataset, it must be thoroughly cleaned. This implies multiple parts, which we explain in the following:

### Command Fix

This is the first thing that should be done. The command given to the model (i.e., follow the lane, turn left, turn right, etc.) is sometimes given too late, resulting in poor performance for the model as it cannot learn to make the maneuver in time. 

To fix this, we can see where the command was given and rewrite the CAN bus data so that this command is given some seconds before. We do this by running:

```bash
python3 data_analysis/data_tools.py command-fix --dataset-path=/path/to/your/dataset
```

This will create for each `can_bus%06d.json` file a new `cmd_fix_can_bus%06d.json` file, which we will use for training.

### Resize RGB Images

If the collected images are at a higher resolution than the ones we will train with, we can resize them offline so that training takes less time. We do so by running:

```bash
python3 data_analysis/data_tools.py resize-dataset --dataset-path=/path/to/your/dataset --res=300x300 --img-ext=png --processes-per-cpu=4
```

This will create, for each of the RGB images in the dataset, an equivalent one with same frame number and image extension. For example, if we have `rgb_central000345.png`, the code will create `resized_rgb_central000345.png` at `300x300` resolution. The number of processes should speed things up, but these will depend on your hardware configuration.

### Data Visualization

We can visualize the routes and create videos of the driving to more easily see if the collected data has some faults. 

```bash
python3 data_analysis/data_tools.py visualize-routes --dataset-path=/path/to/your/dataset --fps=20 --camera-name=resized_rgb --processes-per-cpu=1
```

If we don't sepcify the output directory (`--out`), then the script will create a `videos` directory in the provided `--dataset-path`. An example of the created videos in a dataset collected in `Town04` with `WetNoon` weather condition is shown here:

<div align="center">
<video width="600" controls autoplay loop muted markdown="1">
    <source src="docs/Town04_WetNoon_route00011_faster.mp4" type="video/mp4">
</video>
</div>

We will have the most important information read from the `cmd_fix_can_bus` data: at the top, the frame number, command to follow, and current speed in meters per second, and in the bottom the two values we will learn to predict: the steering angle and acceleration (throttle - brake values). 

This step is useful if we wish to verify the collected data is correct, or if we should remove specific frames. We do this in the following section.

### Clean Route

Sometimes there are frames where the ego vehicle is not performing well, whether by mistake or design. For example, the Roach agent knows that it should stop due to a red light that the cameras cannot see, resulting in a causal confusion. Another more egregious behavior is colliding with other vehicles or pedestrians, which we do not wish to train on.

We can annotate which frames this happens at (or ranges of frames) when we visualize the routes above, and remove them via the following script:

```bash
python3 data_analysis/data_tools.py clean-route --route-path=/path/to/your/dataset/route --remove-ticks=90-1020,1500-1700
```

In the above, all of the data (RGB, depth, can_bus, etc.) will be removed that have ticks in the ranges `[90, 1020]` and `[1500, 1700]` (note the endpoints are inclusive). On the other hand, if we pass an integer like `--remove-ticks=205`, this will remove all the data with ticks in the range `[0, 205)` (note it's non-inclusive). Due to the fact you are deleting data, the script above will ask for you to input `yes`, along showing the amount of files that will be deleted (not the amount of frames, but the amount of data).

### Mask Generation

To creat the *synthetic attention masks* explained in the paper, we must collect a semantic segmentation and depth images for each frame and for each RGB sensor. 

TBD

## Experiment Configuration

We will define a configuration file for training a model. In it, we will define the typical training hyperparameters, as well as network setup and training data to use. All of the available configurations are in the [`configs/_global.py`](./configs/_global.py), so if more should be added for training or evaluation, they should also be added there. 

Some experiments are provided in the `configs` directory.

TBD

## Running an experiment

<!-- To train the defined experiment above -->

TBD


## Citation

Should you find this work useful, please cite (to be updated with the IV citation):

```bibtex
@misc{porres2024guiding,
      title={Guiding Attention in End-to-End Driving Models}, 
      author={Diego Porres and Yi Xiao and Gabriel Villalonga and Alexandre Levy and Antonio M. López},
      year={2024},
      eprint={2405.00242},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements

This research is supported by project TED2021-132802BI00 funded by MCIN/AEI/10.13039/501100011033 and the European Union NextGenerationEU/PRTR. Antonio M. Lopez acknowledges the financial support to his general research activities given by ICREA under the ICREA Academia Program. Antonio and Gabriel thank the synergies, in terms of research ideas, arising from the project PID2020-115734RB-C21 funded by MCIN/AEI/10.13039/501100011033. The authors acknowledge the support of the Generalitat de Catalunya CERCA Program and its ACCIO agency to CVC’s general activities.

<div align="center">
 <img src="docs/logos_inst_whitebg.png" height="150">
</div>