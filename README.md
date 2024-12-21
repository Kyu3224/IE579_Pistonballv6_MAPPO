# MAPPO Implementation for Pistonball Environment

This project implements the **MAPPO** (Multi-Agent Proximal Policy Optimization) algorithm using the **SKRL** library (https://skrl.readthedocs.io/en/latest/). It is based on the **Pistonball** environment from **PettingZoo** (https://pettingzoo.farama.org/environments/butterfly/pistonball/).

## Installation

To get started, clone the repository and install the required dependencies:
It is recommended to use a virtual environment, such as **conda**.

```bash
git clone <repository_url>
cd <repository_directory>
pip install skrl["torch"]
```

## Train
To train the model, run the following command with your preferred algorithm (either ppo or mappo):
```bash
python train.py --alg {ALG_NAME:ppo or mappo}
```

## Play
To test the trained model, run the following command with your preferred algorithm (either ppo or mappo):
```bash
python play.py --alg {ALG_NAME:ppo or mappo}
```
