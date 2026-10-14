# Poker TouchGuide batch pairing

Steps 1 and 2 are implemented: matched observations/actions, CPU batch assembly,
identity labels, fixed validation groups and source-reference examples.
Similarity scores, the contrastive loss, action noise and training are later steps.

Data root:

```text
/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1
```

Load the new entry explicitly; `CPMDataset`'s default root still refers to the
older dataset. `CPMBatchLoader.from_config` reads this entry's configuration:

```python
from openpi.touchguide.batching import CPMBatchLoader

root = "/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1"
loader = CPMBatchLoader.from_config(root, "train", epoch=0, num_workers=0)
batch = next(iter(loader))
# Model inputs: head, wrist, tactile_map_raw, actions.
# targets = arange(B), used for both score rows and score columns.
# positive_mask has True on the diagonal; negative_mask is its complement.
# sample_ids and metadata preserve the same order in both towers.
loader.set_epoch(1)  # Call this when a future training loop starts another epoch.
loader.dataset.close()
```

The initial grouping size is 64, pending the encoder/training memory decision.
Training shuffles without replacement using `SeedSequence([42, epoch])`,
producing 614 complete groups and omitting 37 shuffled tail samples per epoch.
Validation uses all 4,372 samples: 68 groups of 64 plus a final group of 20.
Its exact permutation and group boundaries are in `index/val_batches.npz`.
Changing only the validation JSON batch size is rejected; fixed groups must
also be regenerated if the grouping size changes.

Actions keep their stored arm-delta-7/hand-absolute-20 representation normalized
with the 20k pi05 statistics. No donor action is rebased to another observation.
There is no old eight-negative pool or noise helper on this loading path.
State/contact auxiliary tensors are omitted by default; metadata remains available.
Off-diagonal labels indicate mismatched demonstration identities, not measured
physical failure. Duplicate identities and singleton batches are rejected.

Inspect `examples/batch_examples.json` for two three-sample illustrative batches.
`examples/pair_examples.jsonl` records their 18 pairs by ID; it does not duplicate
the original images, taxels or trajectories, nor export all possible training pairs.

A small CPU preview (no model or training) can be run from the repository root:

```bash
PYTHONPATH=src:packages/openpi-client/src \
  /path/to/simulator-env/bin/python \
  -m openpi.touchguide.batching --root /path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1 \
  --split val --preview-size 3 --no-images
```

`scripts/touchguide/prepare_batch_pairing.py` creates the saved grouping and
configuration. The duplicate shell wrapper was removed. From the repository root:

```bash
PYTHONPATH=src:packages/openpi-client/src PYTHONDONTWRITEBYTECODE=1 \
  /path/to/simulator-env/bin/python \
  scripts/touchguide/prepare_batch_pairing.py
```

The current dataset is already prepared. The builder refuses to overwrite
existing pairing artifacts.
`scripts/touchguide/validate_batch_pairing.py` checks real samples, identity
labels, fixed coverage and rejection of the legacy negative path.
