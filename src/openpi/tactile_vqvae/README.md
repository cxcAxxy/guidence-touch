# T-Rex F6 VQ-VAE for openpi

This package ports the T-Rex F6 history representation as an independent
module.  The model consumes one hand at a time with shape `[B, 16, 5, 6]` and,
in its default `hand` granularity, emits one code/embedding per 16-frame
window.  The pi0.5 policy is deliberately untouched in this stage.

The poker sidecar at `pi05_trex/0920_200` is read directly from its
`episodes/*/raw.h5` files.  The loader uses the stored `history_indices`,
selects left or right five-fingertip slices, and applies the sidecar's per-
fingertip q01/q99 ranges.  A constant channel is mapped to zero, which keeps
the unused all-zero left hand from becoming a fixed -1 input.

Example training command:

```bash
PYTHONPATH=src python -m openpi.tactile_vqvae.train \
  --data-root /path/to/datasets/sim/poker-draw/pi05_trex/0920_200 \
  --output-dir /path/to/datasets/sim/poker-draw/pi05_trex/vqvae_runs/hand_k128 \
  --codebook-size 128
```

The default trains on the right hand, because the current poker episodes
have an unused all-zero left hand.  `--hand-mode both` is available once
both hands contain meaningful F6.  At inference, normalize the full
`[B, 16, 10, 6]` window with checkpoint stats, then
`model.encode_both_hands(window)` returns two hand codes and embeddings.

For the current 200-episode poker set, `K=64/128/256` is a sensible initial
sweep.  The checkpoint uses T-Rex's `model_state`/`optim_state` keys and
contains the model configuration plus F6 normalization ranges.

## Poker F6 training with W&B

Use a GPU-enabled Python environment containing `torch`, `h5py`, and `wandb`.
Log in once with `wandb login` in that environment.  This poker run defaults
to the authenticated account's actual default entity
`xianchichen8-hunan-university-library`; the login username is `xianchichen8`.
`WANDB_ENTITY` can override it for another account.

```bash
bash scripts/train_poker_f6_vqvae.sh
```

The script checks CUDA availability before training, runs on the right hand
with 16-frame history, stride 4, K=128, embedding width 256, batch size 256,
30 epochs, AdamW at 3e-4, and an episode-level 90/10 train/validation split.
It logs train reconstruction/VQ losses, codebook usage and perplexity, and
validation loss to `poker-draw-f6-vqvae`.  Checkpoints and `run_config.json`
are saved under a timestamped run directory in `pi05_trex/vqvae_runs`.
`WANDB_PROJECT`, `WANDB_MODE`, `PYTHON_BIN`, `RUN_NAME`, and `OUTPUT_DIR` may be
set in the shell to override the defaults.

## Three-axis tactile-map encoder

`TactileMapEncoder` accepts channel-normalized tactile maps shaped
`[B, F, 3, 7, 5]` (normal, tangent-0, tangent-1) and emits `[B, F, 256]`:
one token per fingertip. It uses separate small convolutional stems for the
three physical axes, fuses their feature maps, and processes the 7x5 grid with
residual blocks without downsampling. Coarse 3x3 spatial pooling retains taxel
location before projection. Both five- and ten-finger batches are supported;
pass a `[B, F]` mask to zero missing-finger tokens (the current poker data has
an unused left hand). Normalize each channel with the sidecar's
`meta/tactile_stats.json` q01/q99 values and clip to [-1, 1] before calling
this encoder.

This module is randomly initialized. The F6 VQ-VAE checkpoint does not include
map-encoder weights; these must be learned in a later tactile-expert training
stage or by a separate map-reconstruction pretraining run.

### Pretraining the tactile-map encoder

The map branch can be pretrained separately with a denoising autoencoder. The
script below only starts when called by the user:

```bash
cd /path/to/guidence-touch
bash scripts/train_poker_tactile_map.sh
```

The default run uses the right-hand maps only, samples every fourth 30 Hz frame,
and splits 180/20 episodes for training/validation with seed 42. A batch of 128
frames contains 640 fingertip maps. The encoder yields one 256D token per
fingertip; its training-only decoder reconstructs each normalized 3x7x5 map.
During training, 10% of taxel locations are replaced with the normalized
physical zero-map value; validation uses unmasked maps. Nonzero-contact maps
have 3x weight in the reconstruction MSE. W&B logs overall, contact, and free
reconstruction errors under the project `poker-draw-tactile-map-ae`. The run
saves `latest.pt` and `epoch_XXXX.pt` each epoch, plus `best.pt` when validation
loss improves, under `pi05_trex/map_runs/<timestamped run name>`. The checkpoint
contains a separate `encoder_state` for later policy integration, along with
the normalization ranges used here. The sidecar's q01/q99 ranges were computed
from all source episodes; for a strict held-out distribution study, recompute
these ranges on training episodes only.
