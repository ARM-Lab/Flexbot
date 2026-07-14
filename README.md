# Flexbot - research and education workspace for intelligent flexible robotics
This repository contains the mechanical design files, experimental datasets, motor-control software, neural-network models, and reinforcement-learning implementation developed for the FLEXBOT cable-driven continuum robot platform.

The repository is intended for students and researchers who want to:

- understand the mechanical design of the FLEXBOT platform,
- control DYNAMIXEL motors through cable displacement commands,
- train forward and inverse kinematic neural-network models,
- train and evaluate a PPO controller,
- reproduce, analyse, and extend the current results.

The current software implementation focuses on a **single continuum segment actuated by three cables**.

---

## Repository structure

```text
FLEXBOT/
├── CAD/
│   └── CAD and manufacturing files
│
├── src/
│   ├── data/
│   │   ├── continuum_robot_dataset.csv
│   │   └── continuum_robot_dataset.txt
│   │
│   ├── models_training/
│   │   ├── train_forward_model.py
│   │   └── train_ik_model.py
│   │
│   ├── motor_control/
│   │   ├── dynamixel_cable.py
│   │   ├── dynamixels_init_testing.py
│   │   └── example_controller.py
│   │
│   └── reinforcement_learning/
│       ├── data/
│       ├── models/
│       ├── results/
│       ├── __init__.py
│       ├── check_setup.py
│       ├── common.py
│       ├── config.yaml
│       ├── continuum_env.py
│       ├── evaluate_ppo.py
│       ├── forward_model.py
│       ├── plot_results.py
│       ├── plot_trajectories.py
│       ├── README.md
│       ├── requirements.txt
│       └── train_ppo.py
│
├── .gitignore
└── README.md
```

---

## System overview

FLEXBOT is a cable-driven continuum robot platform designed for research and education in continuum robotics, mechatronics, learning-based modelling, and intelligent control.

Each robot segment is actuated by three cables. Cable commands are represented in the range:

```text
0–100
```

where:

```text
0   = no cable shortening
100 = 10 mm cable shortening
```

Therefore:

```text
1 command unit = 0.1 mm cable shortening
```

The experimental dataset was generated using combinations in which **one or a maximum of two cables are shortened simultaneously**, 
and respectiove XYZ coordinates were measure using mocap system. For more information about mocap system please ask your supervisor.

---

## Main repository components

### CAD

The `CAD/` directory contains the mechanical design and manufacturing files of the continuum robot platform.

Before manufacturing or modifying any component, check:

- dimensions,
- cable routing,
- motor mounting,
- pulley or spool geometry,
- backbone geometry,
- tolerances between moving parts.

Any mechanical modification may change the relationship between motor rotation, cable displacement, and robot tip position.

### Experimental data

The measured datasets are stored in:

```text
src/data/
```

Available files:

```text
continuum_robot_dataset.csv
continuum_robot_dataset.txt
```

The dataset describes the relationship between cable actuation and the measured Cartesian position of the robot tip.

Before using the dataset, verify:

- column order,
- input and output variables,
- units,
- presence or absence of a header,
- number of samples,
- minimum and maximum values.

Position and cable-displacement values are expressed in millimetres unless stated otherwise in the corresponding script.

### Neural-network training

The neural-network training scripts are stored in:

```text
src/models_training/
```

#### Forward model

```text
train_forward_model.py
```

The forward model estimates the Cartesian tip position from the cable commands:

```text
[cable_1, cable_2, cable_3] -> [x, y, z]
```

Run from the repository root:

```bash
python src/models_training/train_forward_model.py
```

The trained forward model can be used for:

- prediction of robot motion,
- validation of inverse-model outputs,
- simulation without direct access to the physical robot,
- reinforcement-learning environment dynamics.

#### Inverse kinematic model

```text
train_ik_model.py
```

The inverse model estimates the required cable commands for a desired Cartesian tip position:

```text
[x, y, z] -> [cable_1, cable_2, cable_3]
```

Run from the repository root:

```bash
python src/models_training/train_ik_model.py
```

Before training either model, verify that the dataset path inside the script points to the correct file in `src/data/`.

### Motor control

The motor-control implementation is stored in:

```text
src/motor_control/
```

#### Initial DYNAMIXEL test

```text
dynamixels_init_testing.py
```

This script is used for initial communication and motor testing.

Run:

```bash
python src/motor_control/dynamixels_init_testing.py
```

Before execution, verify:

- serial port,
- motor IDs,
- baud rate,
- protocol version,
- operating mode,
- movement direction,
- current and velocity limits,
- initial motor position.

The first tests should be performed with the cables disconnected or mechanically unloaded.

#### Reusable motor-control module

```text
dynamixel_cable.py
```

This file contains the reusable cable-control implementation. It is intended to be imported by other Python scripts.

Its main purpose is to convert a requested cable displacement in millimetres into the corresponding DYNAMIXEL motor position command.

The conversion depends on:

- spool or pulley diameter,
- transmission ratio,
- encoder resolution,
- cable winding direction,
- motor zero position.

These parameters must be recalibrated whenever the mechanical setup is changed.

#### Controller example

```text
example_controller.py
```

This script demonstrates how to import and use the motor-control module from another Python program.

Run:

```bash
python src/motor_control/example_controller.py
```

Use this script as the starting point when integrating the motor controller into another experiment or control algorithm.

### Reinforcement learning

The PPO implementation is stored in:

```text
src/reinforcement_learning/
```

The current reinforcement-learning task is point reaching with one continuum segment and three cables. A trained forward model is used to estimate the robot tip position resulting from the selected cable commands.

Main files:

| File | Purpose |
|---|---|
| `config.yaml` | Experiment configuration |
| `continuum_env.py` | Reinforcement-learning environment |
| `forward_model.py` | Forward-model loading and inference |
| `train_ppo.py` | PPO training |
| `evaluate_ppo.py` | Evaluation of a trained PPO agent |
| `plot_results.py` | Training and evaluation plots |
| `plot_trajectories.py` | Desired and achieved trajectory plots |
| `check_setup.py` | Dependency and file-path verification |
| `common.py` | Shared utility functions |
| `models/` | Saved PPO models |
| `results/` | Evaluation outputs and generated plots |
| `data/` | Data used specifically by the RL implementation |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/PeterJanSincak/Flexbot.git
cd Flexbot
```

Create a Python virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows:

```powershell
.venv\Scripts\activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the reinforcement-learning dependencies:

```bash
pip install -r src/reinforcement_learning/requirements.txt
```

For motor control, install the DYNAMIXEL SDK if it is not already included:

```bash
pip install dynamixel-sdk
```

Additional packages may be required by the model-training scripts, depending on their current implementation.

---

## PPO workflow

### 1. Open the reinforcement-learning directory

```bash
cd src/reinforcement_learning
```

### 2. Check the setup

```bash
python check_setup.py --config config.yaml
```

The setup check should verify:

- required Python packages,
- dataset path,
- forward-model path,
- output directories,
- configuration values.

### 3. Review the configuration

Open:

```text
config.yaml
```

Check at least:

- dataset path,
- forward-model path,
- random seed,
- total training steps,
- PPO hyperparameters,
- cable-command limits,
- target-position range,
- model output path,
- results output path.

### 4. Train PPO

```bash
python train_ppo.py --config config.yaml
```

The trained model should be saved in:

```text
models/
```

Training logs and numerical outputs should be saved in:

```text
results/
```

### 5. Evaluate the trained agent

```bash
python evaluate_ppo.py --config config.yaml
```

Recommended evaluation quantities include:

- mean Euclidean position error,
- median position error,
- maximum position error,
- success rate,
- episode reward,
- number of steps required to reach the target.

### 6. Plot the results

```bash
python plot_results.py --config config.yaml
```

### 7. Plot trajectories

```bash
python plot_trajectories.py --config config.yaml
```

Generated plots should be available in:

```text
results/
```

---

## Recommended workflow for a new student

### Stage 1 – Understand the robot

1. Read this README.
2. Identify the robot segment, cables, backbone, motor mounts, and spools.
3. Understand how motor rotation produces cable shortening.
4. Identify the safe mechanical limits of the robot.

### Stage 2 – Inspect the dataset

1. Open the files in `src/data/`.
2. Identify the input and output columns.
3. Check the units and value ranges.
4. Plot the measured workspace.
5. Plot the distribution of cable commands.
6. Check for missing, duplicated, or invalid samples.

### Stage 3 – Train the forward model

1. Run `train_forward_model.py`.
2. Inspect training and validation losses.
3. Evaluate Cartesian error on each axis.
4. Evaluate Euclidean tip-position error.
5. Save the trained model and all preprocessing parameters.

### Stage 4 – Train the inverse model

1. Run `train_ik_model.py`.
2. Evaluate cable-command prediction error.
3. Pass predicted cable commands through the forward model.
4. Evaluate the resulting Cartesian position error.
5. Compare the inverse-model result with the requested target position.

### Stage 5 – Test the motors

1. Confirm communication with each motor.
2. Test each motor separately.
3. Verify movement direction.
4. Verify the conversion from millimetres to encoder counts.
5. Start with small cable displacements.
6. Confirm that cables are not overtensioned.
7. Test the reusable controller through `example_controller.py`.

### Stage 6 – Train PPO

1. Check `config.yaml`.
2. Verify the forward-model path.
3. Run `check_setup.py`.
4. Train the PPO agent.
5. Evaluate the trained agent.
6. Plot rewards, errors, and trajectories.
7. Repeat training with multiple random seeds before drawing conclusions.

### Stage 7 – Physical validation

Only after successful software validation should the predicted commands be tested on the physical robot.

Physical tests must begin with:

- small displacement commands,
- low motor velocity,
- conservative current limits,
- sufficient cable slack,
- an accessible emergency power switch.

### Stage 8 – Own implementation

Based on the example implementations, you are now free to implement your own controller. 

---

## Safety

Incorrect motor commands can overtension the cables, damage the robot, or overload the motors.

Before every physical experiment:

1. Check the cable routing.
2. Check the cable tension.
3. Verify all motor IDs.
4. Verify the command sign and movement direction.
5. Set safe velocity and current limits.
6. Start from the known initial position.
7. Keep the emergency power-off mechanism accessible.
8. Do not leave the robot unattended while torque is enabled.

---

## Reproducibility

For every experiment, record:

- dataset version,
- Git commit,
- configuration file,
- random seed,
- Python version,
- package versions,
- training device,
- number of epochs or training steps,
- saved model filename,
- evaluation results.

Do not overwrite previous experiments. Store each experiment in a separate directory, for example:

```text
results/
└── ppo_seed_42_2026-07-14/
    ├── metrics.csv
    ├── training_curve.png
    ├── evaluation_results.csv
    └── trajectory.png
```
---

## Possible future work

Possible extensions include:

- physical validation of the PPO controller,
- closed-loop control using external position sensing,
- automatic cable calibration,
- multi-segment continuum robot control,
- comparison of PPO with other reinforcement-learning algorithms,
- uncertainty estimation for the learned models,
- workspace and repeatability analysis,
- integration with OptiTrack or another motion-capture system,
- sim-to-real adaptation.

---

## Acknowledgements

This work was supported by Tatra Bank Foundation, 
Education grant program - Digital for Higher Education 2025.

---