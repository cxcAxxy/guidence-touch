"""Build current-image/current-taxel CPM pairs without modifying source data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import h5py
import numpy as np
import pyarrow.parquet as pq

ROOT = Path('/path/to/datasets/sim/poker-draw')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def normalize(actions_absolute, state_raw, q01, q99):
    relative = actions_absolute.copy()
    relative[..., :7] -= state_raw[..., None, :7]
    return (2 * (relative - q01) / (q99 - q01 + 1e-6) - 1).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pi05-root', type=Path, default=ROOT/'pi05/0920_200')
    p.add_argument('--tactile-root', type=Path, default=ROOT/'pi05_trex/0920_200')
    p.add_argument('--base-checkpoint', type=Path, default=ROOT/'pi05/0920_200/checkpoints/pi05_kaihand/poker_draw_bs128_noskip_25k/20000')
    p.add_argument('--output-dir', type=Path, default=ROOT/'pi05_trex/touchguide/cpm_datasets/poker_0920_200_v1')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--negative-pool-size', type=int, default=8)
    p.add_argument('--minimum-negative-rmse', type=float, default=.05)
    args = p.parse_args()
    if args.stride <= 0 or args.negative_pool_size <= 0 or args.minimum_negative_rmse <= 0:
        raise ValueError('Invalid sampling settings')
    started = time.monotonic()
    out = args.output_dir
    staging = out.with_name(out.name + '.building')
    if out.exists() or staging.exists():
        raise FileExistsError(f'Output/staging already exists: {out}, {staging}')
    info_path = args.pi05_root/'meta/info.json'
    info = json.loads(info_path.read_text())
    if info['fps'] != 30 or info['total_episodes'] != 200:
        raise ValueError('Expected the paired 200-episode 30 Hz dataset')
    side_meta = args.tactile_root/'meta/episodes.jsonl'
    episodes = [json.loads(x) for x in side_meta.read_text().splitlines() if x.strip()]
    if [x['episode_index'] for x in episodes] != list(range(200)):
        raise ValueError('Unexpected tactile episode order')
    norm_path = args.base_checkpoint/'assets/normalizer/norm_stats.json'
    norm = json.loads(norm_path.read_text())['norm_stats']
    q01, q99 = (np.asarray(norm['actions'][k], dtype=np.float32) for k in ('q01', 'q99'))
    if q01.shape != (27,) or q99.shape != (27,):
        raise ValueError('Expected original 27-dimensional action normalization')
    distance_dims = q99 - q01 > 1e-6
    order = np.random.default_rng(args.seed).permutation(200)
    selected = {'val': sorted(order[:20].tolist()), 'train': sorted(order[20:].tolist())}
    split_for = {e: split for split, ids in selected.items() for e in ids}
    for directory in ('meta', 'index', 'episodes', 'negatives'):
        (staging/directory).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(norm_path, staging/'meta/action_norm_stats.json')
    split_rows = {'train': [], 'val': []}
    # Compact endpoint means allow inexpensive rejection of obviously duplicate
    # negative trajectories. Accepted pairs are checked with the complete chunk.
    split_chunks = {'train': [], 'val': []}
    split_states = {'train': [], 'val': []}
    sources, inventory = [], []
    sums = np.zeros(3, dtype=np.float64)
    squares = np.zeros(3, dtype=np.float64)
    minima, maxima = np.full(3, np.inf), np.full(3, -np.inf)
    count = 0
    contact_hist = {'train': np.zeros(6, dtype=np.int64), 'val': np.zeros(6, dtype=np.int64)}
    total_source_frames = 0
    for meta in episodes:
        eid = int(meta['episode_index'])
        n = int(meta['num_frames'])
        split = split_for[eid]
        path = args.pi05_root/info['data_path'].format(episode_chunk=eid//info['chunks_size'], episode_index=eid)
        table = pq.read_table(path)
        def column(name, dtype):
            return np.asarray(table[name].to_pylist(), dtype=dtype)
        frame = column('frame_index', np.int32)
        if len(frame) != n or not np.array_equal(frame, np.arange(n)) or not np.all(column('episode_index', np.int32) == eid):
            raise ValueError(f'Parquet episode/frame alignment failed: {eid}')
        state = column('observation.state', np.float32)
        action = column('action', np.float32)
        if state.shape != (n, 27) or action.shape != (n, 27):
            raise ValueError(f'Unexpected state/action shape: {eid}')
        source_index = column('provenance.source_state_index', np.int64)
        camera_times = column('provenance.source_camera_timestamp_s', np.float64)
        source_action_frames = column('provenance.action_source_camera_frame_index', np.int64)
        source_obs_frames = column('provenance.source_camera_frame_index', np.int64)
        if not np.array_equal(source_action_frames, source_obs_frames+1):
            raise ValueError(f'Unexpected next-frame action target mapping: {eid}')
        anchors = np.arange(0, n-29, args.stride, dtype=np.int32)
        if not len(anchors):
            raise ValueError(f'No complete chunk in episode {eid}')
        chunk_indices = anchors[:, None] + np.arange(30)
        absolute = action[chunk_indices]
        normalized = normalize(absolute, state[anchors], q01, q99)
        side_path = args.tactile_root/meta['file']
        with h5py.File(side_path, 'r') as side:
            if side['tactile_map'].shape != (n, 10, 3, 7, 5):
                raise ValueError(f'Unexpected taxel shape: {eid}')
            if not np.array_equal(np.asarray(side['frame_index']), frame):
                raise ValueError(f'Tactile frame alignment failed: {eid}')
            if not np.array_equal(np.asarray(side['source_state_index']), source_index):
                raise ValueError(f'Tactile state-source alignment failed: {eid}')
            if not np.allclose(np.asarray(side['source_camera_timestamp_s']), camera_times, rtol=0, atol=1e-4):
                raise ValueError(f'Tactile camera-time alignment failed: {eid}')
            maps = np.asarray(side['tactile_map'][anchors, 5:10], dtype=np.float32)
        if any(not np.isfinite(v).all() for v in (state, action, normalized, maps)):
            raise ValueError(f'Nonfinite data in episode {eid}')
        contact = np.any(np.abs(maps) > 1e-6, axis=(2, 3, 4))
        timestamp = column('timestamp', np.float64)[anchors]
        videos = {}
        for key, camera in (('head_video','observation.images.head'), ('right_wrist_video','observation.images.right_wrist')):
            video = args.pi05_root/info['video_path'].format(episode_chunk=eid//info['chunks_size'], episode_index=eid, video_key=camera)
            if not video.is_file():
                raise FileNotFoundError(video)
            videos[key] = str(video.resolve())
        artifact = staging/f'episodes/episode_{eid:06d}.h5'
        values = {'frame_index': anchors, 'timestamp_s': timestamp, 'state_raw': state[anchors],
                  'actions_absolute': absolute, 'actions': normalized, 'tactile_map_raw': maps,
                  'contact_fingers': contact, 'source_state_index': source_index[anchors],
                  'source_camera_timestamp_s': camera_times[anchors]}
        with h5py.File(artifact, 'w') as target:
            for key, value in values.items():
                target.create_dataset(key, data=value, compression='gzip', compression_opts=1, shuffle=True)
            target.attrs.update(videos)
            target.attrs.update(episode_index=eid, split=split, source_parquet=str(path.resolve()),
                                source_tactile=str(side_path.resolve()), source_num_frames=n,
                                fps=30, action_horizon=30, action_dim=27)
        split_rows[split].extend((eid, i, int(t), float(ts)) for i, (t, ts) in enumerate(zip(anchors, timestamp)))
        split_chunks[split].append(absolute)
        split_states[split].append(state[anchors])
        contact_hist[split] += np.bincount(contact.sum(axis=1), minlength=6)
        if split == 'train':
            # All selected training maps, including physical zeros. No validation
            # data and no quantile clipping are used to estimate these scales.
            axis_values = np.moveaxis(maps.astype(np.float64), 2, 0).reshape(3, -1)
            sums += axis_values.sum(axis=1)
            squares += np.square(axis_values).sum(axis=1)
            minima = np.minimum(minima, axis_values.min(axis=1))
            maxima = np.maximum(maxima, axis_values.max(axis=1))
            count += axis_values.shape[1]
        sources.append({'episode_index': eid, 'num_frames': n, 'parquet': str(path.resolve()),
                        'parquet_sha256': digest(path), 'tactile': str(side_path.resolve()),
                        'tactile_sha256': digest(side_path), **videos})
        inventory.append({'path': str(artifact.relative_to(staging)), 'samples': len(anchors), 'sha256': digest(artifact)})
        total_source_frames += n
        if eid % 10 == 0 or eid == 199:
            print(f'Copied aligned arrays: episode {eid+1}/200', flush=True)

    if total_source_frames != info['total_frames']:
        raise ValueError('Source total frame mismatch')
    scale = np.sqrt(squares/count)
    save_json(staging/'meta/tactile_scale.json', {
        'unit': 'N', 'channel_order': ['normal', 'tangent0', 'tangent1'],
        'estimated_from': 'train split selected current right-hand maps only',
        'values_per_channel': count, 'mean': (sums/count).tolist(), 'rms': scale.tolist(),
        'std': np.sqrt(np.maximum(squares/count-(sums/count)**2, 0)).tolist(),
        'min': minima.tolist(), 'max': maxima.tolist(),
        'optional_normalization': 'x / max(rms, 1e-6), preserving physical zero; no clipping',
        'dataset_stores': 'raw float32 forces only; scaling is an explicit encoder/training choice',
    })
    negative_stats = {}
    for split in ('train', 'val'):
        rows = split_rows[split]
        ids = np.asarray([r[0] for r in rows], dtype=np.int32)
        frames = np.asarray([r[2] for r in rows], dtype=np.int32)
        states = np.concatenate(split_states[split])
        chunks = np.concatenate(split_chunks[split])
        means = chunks.mean(axis=1)
        np.savez_compressed(staging/f'index/{split}.npz', episode_index=ids,
            sample_row=np.asarray([r[1] for r in rows], np.int32), frame_index=frames,
            timestamp_s=np.asarray([r[3] for r in rows], np.float64))
        rng = np.random.default_rng(args.seed + (101 if split == 'train' else 202))
        pool = np.empty((len(rows), args.negative_pool_size), dtype=np.int64)
        margins = np.empty(len(rows), dtype=np.float32)
        observed_min = np.inf
        for i in range(len(rows)):
            accepted = []
            batch_attempts = 0
            positive = normalize(chunks[i], states[i], q01, q99)
            distances = []
            while len(accepted) < args.negative_pool_size:
                batch_attempts += 1
                if batch_attempts > 100:
                    raise RuntimeError(f'Insufficient distinct negative candidates: {split}/{i}')
                candidates = rng.integers(0, len(rows), size=32)
                eligible = (ids[candidates] != ids[i]) | (np.abs(frames[candidates]-frames[i]) >= 30)
                for j in candidates[eligible]:
                    j = int(j)
                    if j in accepted:
                        continue
                    candidate = normalize(chunks[j], states[i], q01, q99)
                    rmse = float(np.sqrt(np.mean((candidate[:, distance_dims]-positive[:, distance_dims])**2)))
                    if rmse < args.minimum_negative_rmse:
                        continue
                    accepted.append(j)
                    distances.append(rmse)
                    if len(accepted) == args.negative_pool_size:
                        break
            pool[i] = accepted
            margins[i] = min(distances)
            observed_min = min(observed_min, min(distances))
            if i % 5000 == 0:
                print(f'Negative pair pool: {split} {i}/{len(rows)}', flush=True)
        np.savez_compressed(staging/f'negatives/{split}.npz', negative_sample_index=pool,
                            minimum_normalized_rmse=margins)
        negative_stats[split] = {'count': len(rows), 'pool_size': args.negative_pool_size,
            'observed_minimum_normalized_rmse': float(observed_min),
            'contact_finger_count_histogram': contact_hist[split].tolist()}

    save_json(staging/'meta/sources.json', sources)
    save_json(staging/'meta/splits.json', {'seed': args.seed, 'episode_indices': selected,
        'note': 'CPM train/validation only; base pi0.5 may have seen all 200 episodes.'})
    config = {'fps': 30, 'action_horizon': 30, 'action_dim': 27, 'stride': args.stride,
        'seed': args.seed, 'negative_pool_size': args.negative_pool_size,
        'noise_levels': [0., .1, .2, .3, .4], 'noise_generated_at_training_time': True,
        'noise_formula': '(1-tau)*normalized_action + tau*N(0,I)',
        'minimum_negative_rmse': args.minimum_negative_rmse,
        'same_episode_negative_minimum_frame_separation': 30,
        'contact_taxel_threshold_newtons': 1e-6}
    save_json(staging/'meta/manifest.json', {
        'schema': 'touchguide_cpm_dataset_v1', 'dataset_root': str(out.resolve()),
        'sources': {'pi05_root': str(args.pi05_root.resolve()), 'tactile_root': str(args.tactile_root.resolve()),
                    'base_checkpoint': str(args.base_checkpoint.resolve()),
                    'pi05_info_sha256': digest(info_path), 'sidecar_episodes_sha256': digest(side_meta),
                    'normalizer_sha256': digest(norm_path)},
        'normalized_action_checkpoint_step': int(args.base_checkpoint.name),
        'configuration': config, 'splits': {split: {'episode_indices': selected[split],
                    'episodes': len(selected[split]), **negative_stats[split]} for split in ('train', 'val')},
        'model_inputs': ['current_head_image', 'current_right_wrist_image', 'current_right_tactile_map', 'candidate_action'],
        'auxiliary_only': ['state_raw', 'episode_index', 'frame_index', 'timestamps', 'contact_fingers'],
        'f6_included': False, 'tactile_history_included': False,
        'action_convention': 'first7 absolute_target-current_anchor_state; last20 absolute targets; 20k q01/q99, no clip',
        'negative_convention': 'donor absolute targets re-expressed relative to recipient anchor arm state, then same normalization',
        'negative_label_semantics': 'mismatched contrastive pairs, not simulator-verified task failures',
        'excluded': 'anchors without a complete 30-step action chunk; no duplicated end padding',
        'image_storage': 'existing original videos referenced by absolute path; frame_index/30 is video time',
        'tactile_storage': 'current raw normal/tangent0/tangent1 N, right five fingers, no clipping',
        'artifacts': inventory, 'duration_seconds': time.monotonic()-started,
    })
    readme = f'''# TouchGuide CPM poker dataset v1

Current visual images + current right-hand tactile array + candidate action.
No F6, tactile history, or independently encoded joint-state input.

Source: {args.pi05_root}; tactile sidecar: {args.tactile_root}.
180 train episodes / 20 validation episodes, episode split seed {args.seed}.
Complete 30-step action chunks, stride {args.stride}. These are CPM validation
episodes, not an assertion that the pretrained pi0.5 never saw them.

Clean physical targets and normalized actions are saved in episodes/*.h5.
Current tactile maps are raw float32 forces in N with shape [5,3,7,5].
Images reference the existing videos; this dataset therefore requires the
source video files to remain available. Each video's zero-based frame index
is recorded; images are resized to 224x224 by the reader.

index/train.npz: {len(split_rows['train'])} samples.
index/val.npz: {len(split_rows['val'])} samples.
negatives/*.npz: {args.negative_pool_size} reproducible candidate donor IDs per
sample. They are same-split mismatches, with >=30 frame separation within the
same episode and normalized action RMSE >={args.minimum_negative_rmse}.
They are not experimentally verified failures; similar motions can remain
physically reasonable and should be considered when evaluating the scorer.
Donor arm targets must be re-expressed relative to the recipient's current
state; the provided reader performs this conversion.

meta/action_norm_stats.json is an exact copy of the 20k checkpoint normalizer.
meta/tactile_scale.json provides optional train-only, non-clipping RMS scales;
no tactile normalization is irreversibly applied to the saved arrays.
Noisy action x_tau=(1-tau)*A+tau*epsilon is generated dynamically at training
time; no noisy trajectories or pi0.5 predictions are saved.

Contact flags are sampling/audit metadata derived from raw taxels, not CPM
success labels. Both contact and no-contact frames are retained.
'''
    (staging/'README.md').write_text(readme)
    staging.rename(out)
    print(json.dumps({'dataset_root': str(out), 'train_samples': len(split_rows['train']),
        'val_samples': len(split_rows['val']), 'duration_seconds': time.monotonic()-started}, indent=2), flush=True)


if __name__ == '__main__':
    main()
