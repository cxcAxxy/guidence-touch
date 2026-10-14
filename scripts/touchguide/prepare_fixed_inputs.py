#!/usr/bin/env python3
"""Build shared JAX/Torch inputs using original KaiHand transforms, without a policy model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest import mock

DEFAULT_ROOT = Path('/path/to/datasets/sim/poker-draw')
DEFAULT_EPISODE = DEFAULT_ROOT / 'pi05_trex/eval/pi05_vs_trex_20260924_150754_3384781/pi05+trex/seed_000'
DEFAULT_CHECKPOINT = DEFAULT_ROOT / 'pi05/0920_200/checkpoints/pi05_kaihand/poker_draw_bs128_noskip_25k/20000'
DEFAULT_REPO = Path('/path/to/workspace/code/openpi')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument('--episode-dir', type=Path, default=DEFAULT_EPISODE)
    p.add_argument('--original-repo', type=Path, default=DEFAULT_REPO)
    p.add_argument('--tokenizer-model', type=Path)
    p.add_argument('--request-rows', nargs=3, type=int,
                   help='Three zero-based indices among policy records after filtering event rows')
    p.add_argument('--noise-seed', type=int, default=20260927)
    args = p.parse_args()

    # Only CPU image preprocessing and SentencePiece encoding are needed here.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.original_repo / 'src'))
    sys.path.insert(1, str(args.original_repo / 'packages/openpi-client/src'))

    import h5py
    import numpy as np
    from openpi import transforms
    from openpi.models import tokenizer
    from openpi.policies import kaihand_policy
    from openpi.shared import download
    from openpi.shared import normalize

    source_h5 = args.episode_dir / 'raw/episode.h5'
    requests_path = args.episode_dir / 'requests.jsonl'
    stats_path = args.checkpoint / 'assets/normalizer/norm_stats.json'
    tasks_path = DEFAULT_ROOT / 'pi05/0920_200/meta/tasks.jsonl'
    for source in (source_h5, requests_path, stats_path, tasks_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    candidates = ([args.tokenizer_model] if args.tokenizer_model else [
        Path(os.getenv('OPENPI_DATA_HOME', '~/.cache/openpi')).expanduser() / 'big_vision/paligemma_tokenizer.model',
        Path('$HOME/.cache/openpi/big_vision/paligemma_tokenizer.model'),
    ])
    tokenizer_path = next((path.resolve() for path in candidates if path is not None and path.is_file()), None)
    if tokenizer_path is None:
        raise FileNotFoundError('Cached PaliGemma tokenizer is missing; use --tokenizer-model. Network is disabled.')

    def offline_download(url: str, **_kwargs):
        if str(url) == 'gs://big_vision/paligemma_tokenizer.model':
            return tokenizer_path
        local = Path(url)
        if '://' not in str(url) and local.exists():
            return local.resolve()
        raise RuntimeError(f'Remote access prohibited while preparing fixed inputs: {url}')

    with mock.patch.object(download, 'maybe_download', side_effect=offline_download):
        text_tokenizer = tokenizer.PaligemmaTokenizer(200)
    # This is the original pi05_kaihand inference transform sequence. DeltaActions
    # is absent because inference observations do not contain target actions.
    stats = normalize.load(stats_path.parent)
    input_pipeline = transforms.compose([
        kaihand_policy.KaiHandInputs(),
        transforms.Normalize(stats, use_quantiles=True),
        transforms.ResizeImages(224, 224),
        transforms.TokenizePrompt(text_tokenizer, discrete_state_input=True),
        transforms.PadStatesAndActions(32),
    ])
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip()]
    if len(tasks) != 1:
        raise ValueError('Expected exactly one poker task prompt')
    prompt = tasks[0]['task']
    source_rows = [(row, json.loads(line)) for row, line in enumerate(requests_path.read_text().splitlines())
                   if line.strip()]
    # Rollout logs may interleave event records with actual policy observations.
    # Preserve source line indices while ignoring events without state or time.
    policy_rows = [(row, record) for row, record in source_rows
                   if isinstance(record, dict) and 'state' in record and 'sim_time' in record]
    requests = [record for _, record in policy_rows]
    if len(requests) < 3:
        raise ValueError('Need at least three request records')
    if args.request_rows:
        rows = args.request_rows
    else:
        contact = max(range(len(requests)), key=lambda i: float(requests[i].get('right_map_abs_max_n', 0)))
        if contact in (0, len(requests) - 1):
            contact = len(requests) // 2
        rows = [0, contact, len(requests) - 1]
    if len(set(rows)) != 3 or any(row < 0 or row >= len(requests) for row in rows):
        raise ValueError(f'Invalid representative request rows: {rows}')

    arrays: dict[str, list] = {}
    frame_records = []
    def add(key, value):
        arrays.setdefault(key, []).append(value)

    with h5py.File(source_h5, 'r') as episode:
        state_times = np.asarray(episode['state/timestamp'])
        joint_names = [name.decode() if isinstance(name, bytes) else str(name)
                       for name in episode['state/joint_names'][:]]
        joint_indices = [joint_names.index(name) for name in kaihand_policy.RIGHT_JOINT_NAMES]
        camera_times = {name: np.asarray(episode[f'cameras/{name}/timestamp'])
                        for name in ('head', 'right_wrist')}
        for row in rows:
            request = requests[row]
            sim_time = float(request['sim_time'])
            raw_state = np.asarray(request['state'], dtype=np.float32)
            if raw_state.shape != (27,) or not np.isfinite(raw_state).all():
                raise ValueError(f'Invalid 27-D state in request row {row}')
            record = {'request_row': policy_rows[row][0], 'filtered_policy_request_index': row,
                      'request': request.get('request'), 'sim_time_s': sim_time,
                      'stage': request.get('stage'), 'right_map_abs_max_n': request.get('right_map_abs_max_n'),
                      'state_source': 'exact requests.jsonl state used by policy'}
            images = {}
            for camera in camera_times:
                index = int(np.argmin(np.abs(camera_times[camera] - sim_time)))
                images[camera] = np.asarray(episode[f'cameras/{camera}/rgb'][index], dtype=np.uint8)
                record[camera] = {'camera_index': index, 'timestamp_s': float(camera_times[camera][index]),
                                  'time_offset_s': float(camera_times[camera][index] - sim_time)}
            state_index = int(np.argmin(np.abs(state_times - sim_time)))
            recorded_state = np.asarray(episode['state/robot_joint_position'][state_index])[joint_indices]
            record['nearest_h5_state_index'] = state_index
            record['nearest_h5_state_time_s'] = float(state_times[state_index])
            record['request_vs_nearest_h5_state_max_abs_rad'] = float(np.max(np.abs(raw_state - recorded_state)))
            processed = input_pipeline({'head_image': images['head'], 'right_wrist_image': images['right_wrist'],
                                        'state': raw_state.copy(), 'prompt': prompt})
            add('state', np.asarray(processed['state'], dtype=np.float32))
            add('raw_state', raw_state)
            add('tokenized_prompt', np.asarray(processed['tokenized_prompt'], dtype=np.int32))
            add('tokenized_prompt_mask', np.asarray(processed['tokenized_prompt_mask'], dtype=np.bool_))
            for camera in ('base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'):
                image = np.asarray(processed['image'][camera])
                if image.dtype != np.uint8 or image.shape != (224, 224, 3):
                    raise ValueError(f'Unexpected processed image {camera}: {image.shape}/{image.dtype}')
                add('image_' + camera, image.astype(np.float32) / np.float32(127.5) - np.float32(1.0))
                add('mask_' + camera, np.asarray(processed['image_mask'][camera], dtype=np.bool_))
            frame_records.append(record)

    result = {key: np.stack(value) for key, value in arrays.items()}
    result['noise'] = np.random.default_rng(args.noise_seed).standard_normal((3, 30, 32)).astype(np.float32)
    if result['state'].shape != (3, 32) or result['tokenized_prompt'].shape != (3, 200):
        raise ValueError('Incorrect fixed-input shapes')
    for key, value in result.items():
        if value.dtype.kind == 'f' and not np.isfinite(value).all():
            raise ValueError(f'Non-finite input: {key}')
    npz_path = args.output_dir / 'fixed_inputs.npz'
    manifest_path = args.output_dir / 'fixed_inputs_manifest.json'
    if npz_path.exists() or manifest_path.exists():
        raise FileExistsError('Fixed-input output exists; choose a new --output-dir')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **result)
    manifest = {
        'format': 'touchguide_jax_pytorch_fixed_inputs_v1',
        'checkpoint': str(args.checkpoint.resolve()), 'original_repo': str(args.original_repo.resolve()),
        'config': {'name': 'pi05_kaihand', 'pi05': True, 'action_dim': 32, 'action_horizon': 30,
                   'max_token_len': 200, 'discrete_state_input': True, 'denoising_steps': 10},
        'prompt': prompt, 'noise_seed': args.noise_seed,
        'normalization': 'original checkpoint quantile q01/q99, no state clipping, tokenize 27-D before pad32',
        'camera_selection': 'nearest recorded camera timestamp to exact request sim_time; offsets recorded',
        'frames': frame_records,
        'request_filter': {'total_json_records': len(source_rows), 'policy_records': len(policy_rows),
                           'skipped_event_records': len(source_rows) - len(policy_rows),
                           'rule': 'only dictionaries with state and sim_time',
                           'policy_record_to_source_row': [row for row, _ in policy_rows]},
        'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype)} for key, value in result.items()},
        'files': {label: {'path': str(path.resolve()), 'sha256': sha256_file(path)}
                  for label, path in {'fixed_inputs': npz_path, 'source_h5': source_h5,
                                      'requests': requests_path, 'norm_stats': stats_path,
                                      'tokenizer': tokenizer_path, 'tasks': tasks_path}.items()},
        'transform_source_files': {label: str(Path(module.__file__).resolve())
                                   for label, module in {'transforms': transforms,
                                                         'kaihand_policy': kaihand_policy,
                                                         'tokenizer': tokenizer}.items()},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'fixed_inputs': str(npz_path), 'sha256': manifest['files']['fixed_inputs']['sha256'],
                      'manifest': str(manifest_path), 'frames': frame_records}, indent=2))


if __name__ == '__main__':
    main()
