# Poker TouchGuide: fixed pi0.5 baseline

Stage 1 prepares an independent **20,000-step pi0.5 PyTorch baseline**. It does not train a guidance network, load the old tactile expert, or run the simulator.

- JAX source: `/path/to/datasets/sim/poker-draw/pi05/0920_200/checkpoints/pi05_kaihand/poker_draw_bs128_noskip_25k/20000`.
- Converted weights: `/path/to/datasets/sim/poker-draw/pi05_trex/pi05_pytorch_20k/model.safetensors`.
- Verification artifacts: `/path/to/datasets/sim/poker-draw/pi05_trex/touchguide/baseline_20k/verification`.
- Source normalization: `20000/assets/normalizer/norm_stats.json`, copied into the converted checkpoint and checked by SHA256.

The physical action has 27 dimensions: 7 right-arm joint targets encoded as increments from the current observation, followed by 20 right-hand absolute joint targets. The model pads this to 32 dimensions. The action horizon is 30 and denoising uses 10 Euler steps. The future simulator baseline retains the requested execution horizon of 30; these offline checks do not execute actions.

## Verification result

Three recorded rollout observations were processed once with the original KaiHand transforms and shared between JAX and PyTorch, with the same saved float32 Gaussian noise. Images were already resized to 224×224; only their layout changes between NHWC for JAX and NCHW for PyTorch. They cover no contact, stronger contact, and a later observation. They are rollout observations, not demonstrations or new simulator episodes.

- Normalized first-27 action RMSE: **0.002132**; maximum absolute difference: **0.009766**.
- Right-arm mean absolute difference: **0.000667 rad**; maximum: **0.003963 rad (0.227°)**.
- Right-hand mean absolute difference: **0.000389 rad**; maximum: **0.005211 rad (0.299°)**.
- Explicit guidance-disabled Euler loop versus stock PyTorch sampler: maximum difference **0**.
- Base model parameters are frozen. A direct `denoise_step` permits finite, nonzero gradients with respect to noisy action inputs, with gradient RMS **0.003714** in the diagnostic.

These checks establish a usable conversion baseline on these fixed observations. Simulator task success is a separate evaluation. The numerical screen is an engineering criterion, not an official OpenPI tolerance: each observation must have normalized RMSE ≤ 0.01 and maximum absolute difference ≤ 0.05.

The PyTorch check uses the standard OpenPI inference precision policy: selected VLM patch/norm parameters and top-level projection parameters in FP32, other VLM weights in BF16, with CUDA BF16 autocast. Original JAX inference restores parameters in BF16 and uses its original pi05_kaihand compute configuration. Small backend numerical differences remain.

## Reproduce

```bash
cd /path/to/guidence-touch
bash scripts/prepare_poker_touchguide_baseline.sh
```

The script header contains the interpreter paths, source checkpoint, converted checkpoint, and verification directory. Existing converted weights are reused only after checking their source checkpoint and configuration. Existing fixed observations are reused; outputs and the manifest are refreshed deterministically. A conversion will refuse to overwrite an existing output directory.

`baseline_manifest.json` contains source/converted weight hashes, normalization hash, fixed input hash, per-observation/per-joint/per-action-step error statistics, configuration and gradient checks. `fixed_inputs_manifest.json` records the exact source requests, camera timestamps, prompt, tokenizer and preprocessing sources.

The source converter now selects pi0.5 normalization from the model configuration, validates missing/unexpected parameters, preserves FP32 loading before the requested final cast, and copies checkpoint-local normalization assets. Vocabulary output heads omitted by the source mapping are explicitly recorded; they do not participate in action inference.

The existing T-Rex tactile expert was trained with the 24,999-step base. It is not part of this 20,000-step TouchGuide baseline. Existing encoder and tactile-expert training outputs are unchanged.
