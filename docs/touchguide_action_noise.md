# TouchGuide CPM action noise (step 3)

Action noise is applied online **after** clean batch collation. Source H5, RGB,
taxels, normalized clean actions, sample IDs and identity labels remain unchanged.
No CPM model, scores, contrastive loss, training or simulation is implemented by this step.

Dataset entry:
`/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1`

## Formula and local settings

Given normalized clean action chunks `actions [B,30,27]`, use

```text
K = 10, p = 0.3
P(k) = p(1-p)^k / [1-(1-p)^K], k in {0,...,9}
flow_time = k/K
epsilon ~ N(0,I), shape [B,30,27]
noisy_actions = (1-flow_time)*actions + flow_time*epsilon
```

The geometric distribution is truncated and renormalized, not sampled unbounded
then clamped. One time is shared by a batch. Each action column receives independent
Gaussian noise; the same resulting action column is reused for all observation rows
of the future B-by-B score matrix. No extra negative-action tensor is built.
Time 0 is exactly clean; 1 is pure noise and is excluded from random sampling.
Forced time 1 remains available only for checking the interpolation boundary.
Do not clip noisy actions or normalize them a second time.

TouchGuide describes geometric sampling of noising steps and linear flow interpolation
([paper, V-C1](https://arxiv.org/html/2601.20239v6)). The geometric p, ten-level grid,
mapping k/K and batch-shared time are our local choices, not specified paper defaults.
This is CPM augmentation; it does not change pi0.5's own time sampler or denoising schedule.

## Data interface

Clean reader and `CPMBatchLoader` remain available. `noise_on_read` stays false.
Use the wrapper explicitly for CPM augmentation:

```python
from openpi.touchguide.noise import NoiseAugmentedBatchLoader

train = NoiseAugmentedBatchLoader.from_config(root, "train", epoch=0)
batch = next(iter(train))
# CPM: head, wrist, tactile_map_raw, noisy_actions.
# Retained reference: actions. Added diagnostics: flow_time, noise_level, noise_epsilon.
# A later model may also consume flow_time for explicit time conditioning.
# targets, positive_mask and negative_mask still identify the original sample.
train.set_epoch(1)
train.dataset.close()
```

`config/{train,val}.json` contains `noise_augmentation`, including
`cpm_action_input_key="noisy_actions"`. `model_input_keys` describes the clean reader;
`augmented_model_input_keys` describes the augmented CPM input. A future trainer must
select the latter. Noise itself never changes which observation/action pair is positive.
Negatives mean identity mismatch, not verified task failure.

Training grouping uses the existing seed/epoch permutation. Noise uses a separate
CPU generator seeded from noise seed and epoch, so it does not consume global RNG.
Changing epoch changes both grouping and noise; repeated iteration at the same epoch
reproduces the augmentation. Call `set_epoch` once per new epoch.

Validation keeps the saved grouping and computes a fixed epsilon per sample ID
using a stable blake2b-derived seed. The same epsilon is reused at each noise time
and is independent of batch order or size. Evaluate all configured validation times
0,0.1,...,0.9 separately; the default time=0 is only the clean baseline:

```python
for time in [i / 10 for i in range(10)]:
    val = NoiseAugmentedBatchLoader.from_config(root, "val", validation_time=time)
    for batch in val:
        pass  # Future CPM validation; no model is run by the loader.
    val.dataset.close()
```

The seed recipe is recorded in `meta/noise_augmentation.json` and code in `noise.py`.
Exact cross-device/precision RNG identity is not promised; this stage uses CPU float32.

## Artifacts and checks

- `meta/noise_augmentation.json`: settings, probabilities and status.
- `examples/noise_examples.json`: source IDs and references for three validation samples.
- `examples/noise_examples.npz`: clean actions, fixed epsilon, ten noisy levels and original masks.
- `meta/noise_validation.json`: independent CPU formula, frequency and reproducibility checks.
- `meta/noise_loader_validation.json`: actual configured wrapper checks and saved-example consistency.

Only small examples are saved; no full noisy dataset is materialized. The current entry
already has this step prepared. The one-time preparation script refuses to overwrite
existing noise artifacts:

```bash
PYTHONPATH=src:packages/openpi-client/src PYTHONDONTWRITEBYTECODE=1 \
  /path/to/simulator-env/bin/python \
  scripts/touchguide/prepare_action_noise.py
```

For a read-only preview (no model or training):

```bash
PYTHONPATH=src:packages/openpi-client/src PYTHONDONTWRITEBYTECODE=1 \
  /path/to/simulator-env/bin/python \
  -m openpi.touchguide.noise --root /path/to/datasets/sim/poker-draw/pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1 \
  --split val --preview-size 3 --flow-time 0.4 --no-images
```
