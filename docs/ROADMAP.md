# PokerGPU Roadmap

This roadmap turns the solver into a practical system in stages.
It prioritizes correctness first, then trained leaf evaluation, then larger abstractions, then multiway approximation.

## Goal

Build a GPU-assisted poker solver that can:

- solve small postflop subgames correctly
- use a trained value network at depth limits
- reuse blueprint strategies as warm starts
- extend to limited 6-max approximation
- keep runtime latency low enough for real use

## Main rule

Do not expand scope until the current layer is validated.

If a smaller game is not correct, the larger game will only hide the bug.

## Phase 1: Make small-game solving trustworthy

### Scope

- heads-up only
- postflop only at first
- tiny abstraction
- fixed action sets
- deterministic CPU baseline

### Deliverables

- a solver path that builds a public tree from a real state
- correct terminal EV calculation
- correct card removal and range masking
- correct canonical board handling
- correct regret updates on a tiny tree
- a stable leaf-evaluator interface

### Validation

- compare solver EV against exact or manually checked toy cases
- run symmetry tests on isomorphic boards
- run range conservation tests after every chance node
- run regression tests on one fixed state

### Exit condition

The solver can resolve a small postflop spot end to end and the outputs are stable across repeated runs.

## Phase 2: Build the first usable value network

### Scope

- scalar EV prediction first
- small MLP or residual MLP
- supervised labels only from trusted solver outputs

### Data source

- tiny solved subgames
- depth-limited states with reliable labels
- held-out validation spots from the same abstraction family

### Deliverables

- feature builder for public state + ranges
- dataset export pipeline
- training loop
- checkpointing
- inference path usable by the runtime solver

### Validation

- train/validation loss curves
- held-out EV error
- inference shape and range checks
- parity against CPU stub on fixed states

### Exit condition

The value network can replace the leaf stub in the small solver without breaking EV sanity or runtime stability.

## Phase 3: Connect value network to depth-limited solving

### Scope

- runtime depth-limited re-solving
- value net at the frontier
- warm-start from cached blueprint data where available

### Deliverables

- frontier batch builder
- GPU inference call in the runtime path
- deterministic fallback when the model is missing
- time-budgeted solve loop

### Validation

- latency stays inside target budget
- action distribution is stable across repeated resolves
- network output does not produce obvious EV spikes

### Exit condition

The runtime solver can take a live state and return an action using the trained leaf evaluator.

## Phase 4: Expand abstraction carefully

### Scope

- richer postflop bet sizes
- more streets
- stronger bucket design
- blueprint export and reuse

### Deliverables

- position-aware preflop abstraction
- IP/OOP postflop sizing sets
- board canonicalization shared across all modules
- bucketed postflop representation
- blueprint storage format

### Validation

- blueprint lookup matches canonical keys
- bucket mapping is consistent across training and runtime
- action abstraction changes do not break previous tests

### Exit condition

The solver can handle larger abstracted heads-up and limited 3-way subgames with the same core pipeline.

## Phase 5: Add limited multiway approximation

### Scope

- 3-way first
- then 4-6 players with approximation
- coalition or aggregated-range handling

### Deliverables

- reduced-player subgame solving
- shared policy for unresolved branches
- heuristic fallback for hard multiway spots
- value network labels for multiway states where available

### Validation

- monotonicity checks when players are added
- range coherence checks
- sampled best-response checks on toy multiway games

### Exit condition

The system can produce stable, non-absurd decisions in multiway spots even if it is not exact equilibrium.

## Phase 6: Make the offline blueprint real

### Scope

- offline CFR or MCCFR on the abstracted tree
- export averaged strategy tables
- use blueprint as warm start for runtime

### Deliverables

- long-run training loop
- checkpoint and resume
- exploitability sampling
- compressed strategy export

### Validation

- convergence curves
- sampled exploitability trend
- reloaded blueprint matches the saved strategy

### Exit condition

Offline training produces reusable strategy tables that improve runtime solving.

## Phase 7: Optimize for GPU throughput

### Scope

- reduce Python overhead
- batch tree work
- move hot loops to dense or block-sparse GPU ops

### Deliverables

- GPU reach propagation
- GPU regret update
- batched leaf evaluation
- benchmark suite

### Validation

- GPU path beats CPU path on the target tree size
- GPU and CPU results stay numerically close

### Exit condition

The solver is fast enough to be useful on the intended hardware.

## What not to do yet

- do not try full exact 6-max equilibrium
- do not start with a giant model
- do not train on unverified labels
- do not optimize GPU kernels before correctness is locked
- do not add multiway complexity before HU postflop is stable

## Suggested order of work

1. Fix the small solver path.
2. Train the first value network on trusted labels.
3. Wire the network into runtime depth-limited solving.
4. Expand abstraction and blueprint handling.
5. Add limited multiway approximation.
6. Build offline blueprint training.
7. Optimize the hot path on GPU.

## Current best focus

The best next step is:

- HU postflop correctness
- then a small trusted dataset
- then value network inference inside the runtime solver

