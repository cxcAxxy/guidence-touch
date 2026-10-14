"""Fixed-input, guidance-disabled pi0.5 PyTorch baseline verification."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import safetensors.torch
import torch

from openpi.models import model as model_api
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline check")
    torch.manual_seed(42)
    torch.set_num_threads(8)
    cfg = Pi0Config(pi05=True, action_dim=32, action_horizon=30, max_token_len=200,
                    pytorch_compile_mode=None, dtype=args.precision)
    print("Constructing PyTorch model", flush=True)
    model = PI0Pytorch(cfg)
    safetensors.torch.load_model(model, str(args.checkpoint / "model.safetensors"), strict=True)
    print("Safetensors loaded", flush=True)
    # Use the same dtype policy as OpenPI's PyTorch inference loader.
    model.paligemma_with_expert.to_bfloat16_for_selected_params(args.precision)
    model.eval().requires_grad_(False).to("cuda")
    arrays = np.load(args.inputs, allow_pickle=False)
    obs = model_api.Observation(
        images={k: torch.from_numpy(arrays["image_" + k]).permute(0, 3, 1, 2).contiguous().to("cuda") for k in IMAGE_KEYS},
        image_masks={k: torch.from_numpy(arrays["mask_" + k]).to("cuda") for k in IMAGE_KEYS},
        state=torch.from_numpy(arrays["state"]).to("cuda"),
        tokenized_prompt=torch.from_numpy(arrays["tokenized_prompt"]).long().to("cuda"),
        tokenized_prompt_mask=torch.from_numpy(arrays["tokenized_prompt_mask"]).to("cuda"),
    )
    noise = torch.from_numpy(arrays["noise"]).to("cuda")
    print("Loaded fixed observations:", noise.shape, flush=True)
    autocast_enabled = args.precision == "bfloat16"
    started = time.monotonic()
    traces = []
    original_denoise = model.denoise_step
    def traced_denoise(*a, **kw):
        v = original_denoise(*a, **kw)
        traces.append(v.detach().float().cpu().numpy())
        return v
    model.denoise_step = traced_denoise
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        actions = model.sample_actions("cuda", obs, noise=noise.clone(), num_steps=10)
    model.denoise_step = original_denoise
    if len(traces) != 10:
        raise AssertionError(f"Expected 10 denoising calls, got {len(traces)}")
    # Explicit Euler loop is the future insertion point for guidance. At strength
    # zero it must reproduce the stock sampler with exactly the same inputs.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        images, masks, tokens, token_mask, state = model._preprocess_observation(obs, train=False)
        prefix, pad, att = model.embed_prefix(images, masks, tokens, token_mask)
        positions = torch.cumsum(pad, dim=1) - 1
        mask4d = model._prepare_attention_masks_4d(make_att_2d_masks(pad, att))
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, cache = model.paligemma_with_expert.forward(attention_mask=mask4d,
            position_ids=positions, inputs_embeds=[prefix, None], use_cache=True)
        x = noise.clone()
        t = torch.tensor(1., device="cuda", dtype=torch.float32)
        dt = torch.tensor(-.1, device="cuda", dtype=torch.float32)
        for _ in range(10):
            x = x + dt * model.denoise_step(state, pad, cache, x, t.expand(x.shape[0]))
            t = t + dt
    zero_diff = float((x - actions).abs().max())
    if zero_diff != 0.:
        raise AssertionError(f"Guidance-disabled Euler loop differs from stock: {zero_diff}")
    # Confirm a frozen action expert still admits gradients with respect to its
    # noisy action input. Do not differentiate image prefix or model parameters.
    with torch.enable_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        noisy = noise.detach().clone().requires_grad_(True)
        velocity = model.denoise_step(state, pad, cache, noisy, torch.full((noisy.shape[0],), .3, device="cuda"))
        gradient, = torch.autograd.grad(velocity.float().square().mean(), noisy)
    if not torch.isfinite(gradient).all() or not torch.any(gradient != 0):
        raise AssertionError("No finite nonzero action-input gradient")
    if any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise AssertionError("Base model was not fully frozen")
    out = actions.detach().float().cpu().numpy()
    if out.shape != arrays["noise"].shape or not np.isfinite(out).all():
        raise AssertionError("Invalid action output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, actions=out, velocities=np.stack(traces))
    metadata = {
        "dtype_policy": "OpenPI loader: selected VLM patch/norm FP32, other VLM BF16, projections FP32, CUDA BF16 autocast",
        "image_layout": "NCHW transposed from shared NHWC inputs without resampling",
        "backend": "pytorch", "precision": args.precision, "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0), "denoising_steps": 10,
        "guidance_disabled_max_abs": zero_diff, "base_frozen": True,
        "action_input_gradient_rms": float(gradient.float().square().mean().sqrt()),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "duration_seconds": time.monotonic() - started,
        "checkpoint": str(args.checkpoint), "inputs": str(args.inputs),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
