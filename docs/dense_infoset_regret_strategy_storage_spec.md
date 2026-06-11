# Dense Infoset, Regret, and Strategy Storage Spec

## Goal

Replace dynamic per-infoset lookup paths with dense contiguous solver tables.

The storage layer must support:
- flat regret arrays
- flat strategy-sum arrays
- precomputed infoset offsets
- contiguous action slices
- fast access in CFR traversal and updates

## Scope

This spec covers:
- infoset layout
- regret and strategy storage
- hot-path access patterns
- contiguous table invariants
- tests for layout correctness

This spec does not cover:
- game tree construction
- card abstraction
- action abstraction
- leaf evaluation

## Current Code Entry Point

The current storage layer already exists in:
- `src/pokergpu/cfr/infosets.py`
- `src/pokergpu/cfr/iteration.py`
- `src/pokergpu/cfr/traversal.py`

The existing types are the right foundation:
- `InfosetLayout`
- `InfosetStore`
- `regret_matching`

## Definitions

- `infoset`: public game state bucket for a player
- `action slice`: contiguous subrange for one infoset in a flat array
- `contiguous table`: one 1D array with all infoset action entries back to back
- `hot path`: code executed every CFR traversal and regret update

## Requirements

### 1. Replace dynamic lookup paths in hot code

The CFR hot path must avoid:
- dict lookups per action
- per-node table reshaping
- nested containers for regrets or strategy sums
- repeated allocation in traversal loops

Hot code should use:
- precomputed `InfosetLayout`
- direct slice views into 1D arrays
- integer infoset ids

### 2. Use flat offsets for regrets and strategy sums

All solver state must be stored in contiguous 1D arrays.

Required layout:
- `regrets: float32[total_actions]`
- `strategy_sums: float32[total_actions]`

For each infoset:
- `offsets[i]` marks the start of that infoset's action slice
- `action_counts[i]` gives the slice length
- `offsets` must be contiguous and monotonic

### 3. Precompute infoset to action-count mapping

The solver must build the infoset layout before iteration begins.

Requirements:
- every infoset has a fixed action count
- action count lookup must be O(1)
- no dynamic action-count discovery in the CFR loop
- the layout must validate itself once, then remain immutable

### 4. Keep all solver tables contiguous

The solver must not split regrets or strategy sums across multiple arrays per infoset.

All contiguous tables must:
- live in one 1D array each
- preserve infoset ordering
- be sliceable without copies
- support vectorized operations

This applies to:
- regrets
- strategy sums
- any future per-action accumulator arrays

## Data Model

### InfosetLayout

`InfosetLayout` should remain the authoritative mapping from infoset id to action range.

It must guarantee:
- matching lengths for action counts and offsets
- contiguous offsets
- positive action counts
- total action count consistency

### InfosetStore

`InfosetStore` should own:
- `layout`
- `regrets`
- `strategy_sums`

It should expose:
- `regrets_for_infoset(i)`
- `strategy_sums_for_infoset(i)`
- `current_strategy(i)`
- `average_strategy(i)`

These accessors should return views into contiguous arrays, not copies.

## Hot-Path Rules

### CFR iteration

Inside the CFR iteration loop:
- use infoset integer ids only
- use direct array slices for action utilities
- update regrets in place
- update strategy sums in place
- avoid allocating new per-infoset containers

### Traversal

Traversal code should pass around:
- infoset ids
- action slices
- action utilities as dense arrays

It should not build nested dicts unless returning debug or summary data.

## Memory Model

The solver state should remain compact and GPU-friendly.

Recommended arrays:
- `regrets`: fp32
- `strategy_sums`: fp32
- action utilities: fp32 temporary buffers

If a future optimization compresses data:
- compression must preserve a contiguous export path
- decompression must restore flat arrays before hot use

## Layout Construction

Before solving:
1. Walk the tree.
2. Collect infoset ids in stable order.
3. Determine each infoset's action count.
4. Build `InfosetLayout.from_action_counts(...)`.
5. Allocate `InfosetStore.zeros(layout)`.

The layout must be stable for identical tree structures and abstraction ids.

## Validation Rules

The layout must reject:
- empty infoset action counts
- mismatched offsets
- non-contiguous offsets
- incorrect total action count

The store must reject:
- arrays with incorrect length
- non-1D arrays

## Suggested API Extensions

If needed, add helpers for:
- building an `InfosetLayout` from tree data
- flattening action-count maps into contiguous vectors
- returning an infoset slice without extra indexing
- bulk access to all regrets or strategy sums

Recommended helper names:
- `build_infoset_layout`
- `infoset_action_offsets`
- `infoset_action_slice`
- `flatten_action_counts`

## Tests To Add

Add tests for:
- contiguous offsets
- total action count consistency
- slice ranges match action counts
- store arrays have the right length
- per-infoset slices are views into the flat arrays
- regret updates stay in place
- average strategy falls back to uniform on zero sums
- invalid layouts raise errors

## Acceptance Criteria

This task is complete when:
- all solver tables are flat and contiguous
- infoset-to-action mapping is precomputed
- CFR hot paths avoid dynamic lookup
- tests cover layout and storage invariants
