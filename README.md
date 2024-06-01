# Guiding Attention in End-to-End Driving Models

<div align="center">

[[Project Page]](https://blog.diegoporres.com/guiding-attention-e2e/) [[Arxiv]](https://arxiv.org/abs/2405.00242) <br>
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

See the [Data collection](./docs/data_collection.md) guide for how to properly set up the agent that will collect the training and validation data.

## Data Cleanup

See the [Data cleanup](./docs/data_cleanup.md) guide for how to prepare the collected data for training a new agent. 

## Training and Validation

### Experiment Configuration

We will define a configuration file for training a model. In it, we will define the typical training hyperparameters, as well as network setup and training data to use. All of the available configurations are in the [`configs/_global.py`](./configs/_global.py), so if more should be added for training or evaluation, they should also be added there. 

Some experiments are provided in the `configs` directory.

TBD

### Running an experiment

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