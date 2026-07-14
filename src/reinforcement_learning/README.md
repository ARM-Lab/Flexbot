# PPO point reaching: one continuum segment and three cables

This module is configured for the supplied forward-model training script. PPO is
trained in a learned simulation:

```text
PPO action -> valid cable state [a1,a2,a3] -> forward neural model -> XYZ -> reward
```

## Exact forward-model compatibility

The supplied training script saves these files under `forward_model_output/`:

```text
forward_model_checkpoint.pt
x_scaler.joblib
y_scaler.joblib
```

The PPO loader reproduces the exact network:

```text
3 -> 256 -> 256 -> 128 -> 3
```

including ReLU, BatchNorm and Dropout. It loads `model_state_dict` from
`forward_model_checkpoint.pt` and applies the original input and output
`StandardScaler` objects.

## Installation

```bash

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 1. Verify the complete setup

Run this before PPO training:

```bash
python check_setup.py --config config.yaml
```

It verifies:

- the headerless dataset format;
- cable values and the maximum-two-active-cables constraint;
- checkpoint architecture and state dictionary;
- both StandardScaler files;
- forward-model prediction errors on dataset samples;
- one Gymnasium reset and step.

The reported forward error should be close to the validation error produced by
the original forward-model training. A large difference indicates a wrong
checkpoint, scaler or dataset path. All of these parameters has to be set and 
adjusted according to the needs of the user.

## 2. Short PPO test

```bash
python train_ppo.py --config config.yaml --timesteps 10000
```

This confirms that the environment and PPO loop work without committing to the
full run.

## 3. Full PPO training

```bash
python train_ppo.py --config config.yaml
```

The default configuration runs 500,000 timesteps. Results are stored in:

```text
results/ppo_one_segment/
├── models/best_model.zip
├── models/ppo_continuum_final.zip
├── checkpoints/
├── evaluation/
└── monitor/
```

## Evaluation

Random valid initial states:

```bash
python evaluate_ppo.py --config config.yaml
```

Straight initial state `[0,0,0]`:

```bash
python evaluate_ppo.py --config config.yaml --neutral-start
```

Create result plots:

```bash
python plot_results.py --config config.yaml
```

## Cable-domain constraint

After every PPO action, the environment clips values to `0--100` and subtracts
the smallest of the three cable values. This guarantees at least one zero cable
and therefore at most two shortened cables. The forward model is never queried
with three simultaneously shortened cables.
