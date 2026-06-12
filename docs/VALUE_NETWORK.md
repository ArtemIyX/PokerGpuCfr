# Value Network

## What it is

The value network is a small neural model that estimates the expected value of a poker state near the depth limit of the solver.

It does not replace CFR.
It gives CFR a fast approximation for leaf nodes the solver cannot expand further in real time.

## Why we need it

6-max no-limit holdem has too many states to solve fully at runtime.

We need the value network because:

- it gives leaf EVs when the tree is cut off
- it lets runtime re-solving stay fast on an RTX 5080
- it reduces the need to traverse large subtrees
- it helps offline training by bootstrapping deeper states

Without it, runtime solving would either be too slow or too shallow.

## What it predicts

The first version should predict one of these:

- per-player EV at a leaf state
- or a value vector over hand buckets for the acting player

Start simple.
Use scalar EV output first if the solver pipeline is not finished yet.

## Input and output

Inputs:

- public state: street, pot, stacks, action history, positions
- board encoding
- active player count
- range vectors or bucket summaries for each player

Outputs:

- estimated EV for each active player, or
- estimated value for the acting player from each bucket

## How we will write it

Use a small MLP or residual MLP.

Recommended shape:

- fixed-size numeric input
- embedding or one-hot for categorical features
- 3 to 6 hidden layers
- LayerNorm or BatchNorm if needed
- fp16 or bf16 inference on GPU

Keep it small enough to fit comfortably in 16 GB VRAM.

Do not build a giant transformer first.
That would be slower to train, harder to debug, and unnecessary for the first usable version.

## Training data

Train it from solver-generated labels.

Sources:

- small solved subgames
- abstracted CFR rollouts
- depth-limited re-solving outputs
- sampled best-response or terminal EV labels

Each sample should look like:

- state features
- ranges
- target EV or target bucket values

## Training method

Use a staged approach.

1. Generate a small dataset from easy solved spots.
2. Train a baseline supervised model.
3. Validate on held-out solver states.
4. Add harder spots and retrain.
5. Periodically refresh labels from improved solver runs.

This is not one quick run.
It is an iterative pipeline.

## Hardware limits

We only have:

- RTX 5080 with 16 GB VRAM
- 64 GB RAM

So the training must be bounded:

- use small batches
- use mixed precision
- keep the model compact
- cache data on disk
- stream batches from CPU memory
- avoid full-game end-to-end training at once

Target training time should be hours, not weeks.
If a run is likely to take days, the model or dataset is too large.

## How to avoid using the GPU for weeks

Use these rules:

- start with a tiny model
- train on a small curated dataset first
- run short experiments and compare validation loss
- stop early when validation stops improving
- checkpoint often
- use a time budget per training run
- cap the number of training samples per phase

The goal is a practical approximation, not a massive foundation model.

## Saving the model

Save:

- model weights
- optimizer state
- scheduler state
- training step or epoch
- random seed state
- feature normalization stats
- best validation checkpoint

Use one folder per run and keep the best checkpoint separate from the latest checkpoint.

## Monitoring libraries

Add tools for training visibility:

- `tensorboard`
- `psutil`
- `pynvml` or `nvidia-ml-py`
- `rich`


These help track:

- GPU memory usage
- GPU utilization
- loss curves
- validation error
- checkpoint timing
- batch throughput


## Rule of thumb

The value network should be good enough to support solving, not good enough to replace solving.
