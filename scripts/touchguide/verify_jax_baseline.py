#!/usr/bin/env python3
"""Evaluate original pi05_kaihand on already transformed, fixed-noise inputs.

This is an offline diagnostic: inputs can be recorded rollout observations;
they are not assumed to be demonstrations. Output actions stay in normalized
32-D model space (arm delta convention), before policy output transforms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OPENPI_ROOT = Path('/path/to/workspace/code/openpi')
IMAGE_KEYS = ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--checkpoint20k', required=True, type=Path,
                        help='Original Orbax checkpoint directory containing params/.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--restore-dtype', choices=('bf16', 'float32'), default='bf16',
                        help='Parameter restoration dtype; model compute dtype stays as configured.')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Process input observations sequentially by default to limit VRAM.')
    parser.add_argument('--trace', action='store_true',
                        help='Additionally recompute Euler states/velocities and verify its final result.')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    # Set before importing JAX or any OpenPI module that imports it.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ.setdefault('JAX_PLATFORMS', 'cuda')
    os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(OPENPI_ROOT / 'src'), str(OPENPI_ROOT / 'packages/openpi-client/src')]

    import numpy as np
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as model_lib
    from openpi.models import pi0 as pi0_lib
    from openpi.shared import nnx_utils
    from openpi.training import config as config_lib

    loaded_source = Path(model_lib.__file__).resolve()
    if not loaded_source.is_relative_to((OPENPI_ROOT / 'src').resolve()):
        raise RuntimeError(f'Wrong OpenPI source imported: {loaded_source}')
    gpus = jax.devices('gpu')
    if not gpus:
        raise RuntimeError('No JAX GPU is available')
    print('JAX devices:', gpus, flush=True)

    with np.load(args.inputs, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    state = np.asarray(data['state'], dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 32 or len(state) == 0:
        raise ValueError(f'state must be nonempty [B,32], got {state.shape}')
    batch_size = len(state)
    noise = np.asarray(data['noise'], dtype=np.float32)
    if noise.shape != (batch_size, 30, 32):
        raise ValueError(f'noise must be [B,30,32], got {noise.shape}')
    images, masks = {}, {}
    for key in IMAGE_KEYS:
        image = np.asarray(data[f'image_{key}'])
        if image.shape != (batch_size, 224, 224, 3) or not np.issubdtype(image.dtype, np.floating):
            raise ValueError(f'image_{key} must be floating [B,224,224,3], got {image.shape}/{image.dtype}')
        if not np.isfinite(image).all() or image.min() < -1.0001 or image.max() > 1.0001:
            raise ValueError(f'image_{key} must be finite and normalized to [-1,1]')
        images[key] = image.astype(np.float32)
        mask = np.asarray(data[f'mask_{key}'])
        if mask.shape != (batch_size,) or mask.dtype != np.bool_:
            raise ValueError(f'mask_{key} must be bool [B], got {mask.shape}/{mask.dtype}')
        masks[key] = mask
    tokens = np.asarray(data['tokenized_prompt'])
    token_mask = np.asarray(data['tokenized_prompt_mask'])
    if tokens.shape != (batch_size, 200) or not np.issubdtype(tokens.dtype, np.integer):
        raise ValueError(f'tokenized_prompt must be integer [B,200], got {tokens.shape}/{tokens.dtype}')
    if token_mask.shape != tokens.shape or token_mask.dtype != np.bool_:
        raise ValueError('tokenized_prompt_mask must be bool with tokenized_prompt shape')
    if not np.isfinite(state).all() or not np.isfinite(noise).all():
        raise ValueError('state and noise must be finite')

    params_dir = args.checkpoint20k.resolve() / 'params'
    if not params_dir.is_dir():
        raise FileNotFoundError(params_dir)
    config = config_lib.get_config('pi05_kaihand').model
    print('Restoring:', params_dir, 'dtype:', args.restore_dtype, 'config:', config, flush=True)
    start = time.monotonic()
    dtype = jnp.bfloat16 if args.restore_dtype == 'bf16' else jnp.float32
    # Use a single GPU even if several devices are visible.
    with jax.default_device(gpus[0]):
        params = model_lib.restore_params(
            params_dir, dtype=dtype, sharding=jax.sharding.SingleDeviceSharding(gpus[0])
        )
        model = config.load(params)
    del params
    sample = nnx_utils.module_jit(model.sample_actions)

    class TraceSampler(nnx.Module):
        """Copy the original Euler loop with diagnostic output buffers only."""
        def __init__(self, base):
            self.base = base

        def run(self, observation, fixed_noise):
            base = self.base
            observation = model_lib.preprocess_observation(None, observation, train=False)
            b = observation.state.shape[0]
            prefix, prefix_mask, prefix_ar_mask = base.embed_prefix(observation)
            _, kv_cache = base.PaliGemma.llm(
                [prefix, None], mask=pi0_lib.make_attn_mask(prefix_mask, prefix_ar_mask),
                positions=jnp.cumsum(prefix_mask, axis=1) - 1,
            )
            dt = -1.0 / 10
            states = jnp.zeros((11, *fixed_noise.shape), dtype=fixed_noise.dtype).at[0].set(fixed_noise)
            velocities = jnp.zeros((10, *fixed_noise.shape), dtype=fixed_noise.dtype)
            times = jnp.zeros((10,), dtype=jnp.float32)

            def step(carry):
                x_t, t, i, xs, vs, ts = carry
                suffix, suffix_mask, suffix_ar_mask, cond = base.embed_suffix(
                    observation, x_t, jnp.broadcast_to(t, b)
                )
                suffix_attn_mask = pi0_lib.make_attn_mask(suffix_mask, suffix_ar_mask)
                prefix_attn_mask = jnp.broadcast_to(
                    prefix_mask[:, None, :], (b, suffix.shape[1], prefix.shape[1])
                )
                mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
                (_, hidden), _ = base.PaliGemma.llm(
                    [None, suffix], mask=mask, positions=positions, kv_cache=kv_cache,
                    adarms_cond=[None, cond],
                )
                velocity = base.action_out_proj(hidden[:, -base.action_horizon:])
                x_next = x_t + dt * velocity
                return x_next, t + dt, i + 1, xs.at[i + 1].set(x_next), vs.at[i].set(velocity), ts.at[i].set(t)

            result = jax.lax.while_loop(
                lambda carry: carry[1] >= -dt / 2,
                step, (fixed_noise, 1.0, jnp.int32(0), states, velocities, times),
            )
            return result[3], result[4], result[5]

    trace = nnx_utils.module_jit(TraceSampler(model).run) if args.trace else None
    actions, trace_states, trace_velocities, trace_times, trace_errors = [], [], [], [], []
    for first in range(0, batch_size, args.batch_size):
        last = min(first + args.batch_size, batch_size)
        sl = slice(first, last)
        obs = model_lib.Observation.from_dict({
            'state': jnp.asarray(state[sl]),
            'image': {k: jnp.asarray(v[sl]) for k, v in images.items()},
            'image_mask': {k: jnp.asarray(v[sl]) for k, v in masks.items()},
            'tokenized_prompt': jnp.asarray(tokens[sl], dtype=jnp.int32),
            'tokenized_prompt_mask': jnp.asarray(token_mask[sl]),
        })
        fixed_noise = jnp.asarray(noise[sl])
        prediction = np.asarray(sample(jax.random.key(0), obs, num_steps=10, noise=fixed_noise), dtype=np.float32)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f'Nonfinite predictions for input rows {first}:{last}')
        actions.append(prediction)
        if trace is not None:
            xs, vs, ts = (np.asarray(v) for v in trace(obs, fixed_noise))
            trace_states.append(np.moveaxis(xs, 1, 0))
            trace_velocities.append(np.moveaxis(vs, 1, 0))
            trace_times.append(np.broadcast_to(ts, (last - first, len(ts))))
            trace_errors.extend(np.max(np.abs(xs[-1] - prediction), axis=(1, 2)).tolist())
        print(f'Completed input rows {first}:{last} / {batch_size}', flush=True)
    result = {'actions': np.concatenate(actions), 'noise': noise}
    if trace is not None:
        result.update(
            trace_states=np.concatenate(trace_states),
            trace_velocities=np.concatenate(trace_velocities),
            trace_times=np.concatenate(trace_times),
            trace_final_max_abs_difference=np.asarray(trace_errors),
        )
    metadata = {
        'source': str(OPENPI_ROOT), 'checkpoint': str(args.checkpoint20k.resolve()),
        'config': 'pi05_kaihand', 'restore_dtype': args.restore_dtype,
        'compute_dtype': str(config.dtype), 'num_steps': 10,
        'input_sha256': hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        'input_origin': 'caller-provided model-ready observations; may be recorded rollout states',
        'actions_space': 'normalized model 32-D output before unnormalize/absolute-arm transforms',
        'elapsed_seconds': time.monotonic() - start,
        'trace_states_shape': '[B,11,30,32]: initial noise plus each Euler update',
        'trace_velocities_shape': '[B,10,30,32]',
        'source_sha256': {
            rel: hashlib.sha256((OPENPI_ROOT / rel).read_bytes()).hexdigest()
            for rel in ['src/openpi/models/pi0.py', 'src/openpi/models/model.py',
                        'src/openpi/models/pi0_config.py', 'src/openpi/shared/nnx_utils.py']
        },
    }
    result['metadata_json'] = np.asarray(json.dumps(metadata, sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('wb') as stream:
        np.savez_compressed(stream, **result)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('Saved:', args.output, 'actions32:', result['actions'].shape, flush=True)
    if trace_errors:
        print('Trace vs original final max abs difference:', max(trace_errors), flush=True)


if __name__ == '__main__':
    main()
