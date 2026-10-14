"""Index contact anchors and their recorded geometry; do not synthesize data.

Reads the existing CPM entry, paired tactile sidecars and native recordings.
No simulator, model, video decoder, training, or source writes are involved.
All observations/actions remain referenced by H5 path and sample row.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import multiprocessing as mp
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE = Path('/path/to/datasets/sim/poker-draw')
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
LINKS = tuple('hand_r_' + name + ('_link6' if name == 'thumb' else '_link4') for name in FINGERS)
JOINTS = tuple(f'right_arm_joint{i}' for i in range(1, 8)) + tuple(
    f'hand_r_{finger}_joint{i}' for finger in FINGERS
    for i in ((1, 2, 3, 5) if finger == 'thumb' else (1, 2, 3, 4))
)
VERSION = 'poker_fingertip_contact_anchors_v1'


def strings(dataset):
    return dataset.asstr()[:].tolist()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    return dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'),
                    np.quantile(x, [0, .01, .05, .5, .95, .99, 1]).tolist()))


def durations(contact, times):
    """Age and remaining duration to the first/last camera sample in a run."""
    age = np.zeros(contact.shape, dtype=np.float64)
    remaining = np.zeros_like(age)
    for finger in range(5):
        edges = np.diff(np.r_[False, contact[:, finger], False].astype(np.int8))
        for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            age[start:stop, finger] = times[start:stop] - times[start]
            remaining[start:stop, finger] = times[stop - 1] - times[start:stop]
    return age, remaining


def vectors_or_null(values, valid):
    return [v.tolist() if ok else None for v, ok in zip(values, valid, strict=True)]


def schema():
    f = pa.float64()
    xyz = pa.list_(f, 3)
    five_xyz = pa.list_(xyz, 5)
    columns = []
    for name in ('anchor_uid', 'split', 'demonstration_phase', 'cpm_h5_path', 'source_raw_h5',
                 'source_tactile_h5', 'head_video', 'right_wrist_video', 'geometry_source'):
        columns.append(pa.field(name, pa.string()))
    for name in ('episode_index', 'sample_row', 'frame_index', 'source_camera_frame_index',
                 'source_action_first_camera_frame_index', 'source_state_index', 'source_tactile_index',
                 'state_interp_left_index', 'state_interp_right_index'):
        columns.append(pa.field(name, pa.int64()))
    for name in ('timestamp_s', 'source_camera_timestamp_s', 'source_state_timestamp_s',
                 'source_tactile_timestamp_s', 'fingertip_pose_timestamp_s',
                 'state_camera_skew_s', 'tactile_camera_skew_s', 'pose_camera_skew_s',
                 'pose_tactile_skew_s'):
        columns.append(pa.field(name, f))
    columns.append(pa.field('observation_state_timestamp_s', f))
    columns.append(pa.field('state_interp_right_weight', f))
    for name in ('finger_normal_force_n', 'contact_age_s', 'contact_remaining_s',
                 'normal_coherence', 'away_vs_pad_normal_dot'):
        columns.append(pa.field(name, pa.list_(f, 5)))
    for name in ('contact_fingers', 'eligible_fingers', 'contact_geometry_valid'):
        columns.append(pa.field(name, pa.list_(pa.bool_(), 5)))
    columns.append(pa.field('contact_event_count', pa.list_(pa.int64(), 5)))
    for name in ('fingertip_site_position_world_m',
                 'pad_normal_axis_world', 'pad_tangent0_world', 'pad_tangent1_world'):
        columns.append(pa.field(name, five_xyz))
    # Parquet round-tripping nullable fixed-size child lists fails on our PyArrow.
    # Keep inactive contact vectors genuinely null, using variable-size xyz lists.
    for name in ('contact_centroid_world_m', 'contact_normal_away_world',
                 'contact_tangent0_world', 'contact_tangent1_world'):
        columns.append(pa.field(name, pa.list_(pa.list_(f), 5)))
    columns.append(pa.field('fingertip_site_rotation_world', pa.list_(pa.list_(f, 9), 5)))
    columns.append(pa.field('contact_basis_handedness', pa.list_(pa.int64(), 5)))
    columns.append(pa.field('card_pose_xyz_wxyz', pa.list_(f)))
    columns.append(pa.field('card_pose_timestamp_s', f))
    return pa.schema(columns, metadata={
        b'schema': VERSION.encode(), b'finger_order': ','.join(FINGERS).encode(),
        b'normal_semantics': b'+n moves the pad away from the object, using sign-corrected contact reaction',
        b'observation_action_storage': b'reference cpm_h5_path/sample_row; no copies or perturbations',
    })


def contact_geometry(raw, tactile_indices, raw_link_indices, min_coherence):
    """Use native contact frames, sign-corrected to the force on each pad.

    Contact rows are world-space axes; the native wrench acts on geom2.
    Preserve grid-col/grid-row tangent signs. Handedness may be -1: this
    contact chart must never be used as a fingertip SO(3) target rotation.
    """
    group = raw['tactile_contact_force']
    metadata = json.loads(group.attrs['metadata_json'])
    names = strings(raw['model/geom_names'])
    name_to_id = {name: i for i, name in enumerate(names)}
    target_ids = {name_to_id[name] for name in metadata['target_geom_names']}
    pads = {name_to_id[link + '_tactile_pad_col']: i for i, link in enumerate(LINKS)}
    axes = group['normal_axis_world'][:][:, raw_link_indices]
    tangents = group['tangent_basis_world'][:][:, raw_link_indices]
    fn = group['normal_force_n'][:][:, raw_link_indices]
    normal_force = group['normal_force_world_n'][:][:, raw_link_indices]
    native_counts = group['contact_count'][:][:, raw_link_indices]
    total_force = group['force_world_n'][:][:, raw_link_indices]
    proxy = raw['tactile_proxy']
    proxy_ids = [strings(proxy['link_names']).index(link) for link in LINKS]
    proxy_fn = proxy['normal_force'][:][:, proxy_ids]
    proxy_force = proxy['force_world'][:][:, proxy_ids]
    proxy_counts = proxy['contact_count'][:][:, proxy_ids]
    proxy_centroid = proxy['centroid_world'][:][:, proxy_ids]
    # The solver proxy includes all contacts on each distal body. Pad/object
    # contacts are a subset, so equal counts, Fn and force establish that its
    # recorded centroid uses the same events. Avoid re-reading every event.
    matches = (native_counts == proxy_counts) & np.isclose(fn, proxy_fn, atol=1e-7, rtol=1e-5)
    matches &= np.all(np.isclose(total_force, proxy_force, atol=1e-7, rtol=1e-5), axis=-1)
    matches &= (fn <= 0) | np.isfinite(proxy_centroid).all(axis=-1)
    matched_rows = matches.all(axis=1)
    selected = np.asarray(tactile_indices, dtype=np.int64)
    fallback_rows = selected[~matched_rows[selected]]
    if len(fallback_rows):
        events = raw['contacts/events']
        g1, g2 = events['geom1_id'][:], events['geom2_id'][:]
        positions, frames = events['position_world'][:], events['frame_world'][:]
        wrenches, event_state = events['wrench_contact_on_geom2'][:], events['state_index'][:]
        starts, counts = raw['contacts/frame_start'][:], raw['contacts/frame_count'][:]
    result = []
    max_force_error = 0.0
    max_vector_error = 0.0
    for idx in tactile_indices:
        idx = int(idx)
        total = np.zeros(5)
        weighted_position = np.zeros((5, 3))
        reaction = np.zeros((5, 3))
        count = np.zeros(5, dtype=np.int64)
        if matched_rows[idx]:
            total, reaction, count = fn[idx], normal_force[idx], native_counts[idx]
            weighted_position = np.nan_to_num(proxy_centroid[idx]) * total[:, None]
            start, stop = 0, 0
            source = 'native_tactile_reaction_and_matching_solver_centroid'
        else:
            start, stop = int(starts[idx]), int(starts[idx] + counts[idx])
            if start < 0 or stop > len(g1):
                raise ValueError('Native contact event range is invalid')
            source = 'native_contact_events_force_weighted'
        for event in range(start, stop):
            if int(event_state[event]) != idx:
                raise ValueError('Contact frame/event state index disagrees')
            pad1, pad2 = pads.get(int(g1[event])), pads.get(int(g2[event]))
            if pad1 is not None and int(g2[event]) in target_ids:
                finger, sign = pad1, -1.0
            elif pad2 is not None and int(g1[event]) in target_ids:
                finger, sign = pad2, 1.0
            else:
                continue
            magnitude = abs(float(wrenches[event, 0]))
            direction = sign * frames[event, 0] * float(wrenches[event, 0])
            if not np.isfinite(np.r_[magnitude, direction, positions[event]]).all():
                raise ValueError('Nonfinite native contact event')
            total[finger] += magnitude
            reaction[finger] += direction
            weighted_position[finger] += magnitude * positions[event]
            count[finger] += 1
        max_force_error = max(max_force_error, float(np.max(np.abs(total - fn[idx]))))
        max_vector_error = max(max_vector_error, float(np.max(np.abs(reaction - normal_force[idx]))))
        if not np.allclose(total, fn[idx], atol=1e-7, rtol=1e-5):
            raise ValueError('Contact events do not recover recorded fingertip Fn')
        if not np.allclose(reaction, normal_force[idx], atol=1e-7, rtol=1e-5):
            raise ValueError('Contact normal sign/rotation does not recover recorded pad force')
        if not np.array_equal(count, native_counts[idx]):
            raise ValueError('Native tactile/contact event counts disagree')
        length = np.linalg.norm(reaction, axis=1)
        coherence = np.divide(length, total, out=np.zeros(5), where=total > 0)
        if not np.isfinite(np.r_[total, reaction.reshape(-1)]).all() or np.any(coherence > 1 + 1e-5):
            raise ValueError('Invalid native reaction vector or summed normal force')
        valid = (length > 1e-12) & (coherence >= min_coherence)
        normal = np.full((5, 3), np.nan)
        centroid = np.full((5, 3), np.nan)
        t0, t1 = np.full((5, 3), np.nan), np.full((5, 3), np.nan)
        parity = np.zeros(5, dtype=np.int64)
        dot_pad = np.full(5, np.nan)
        for finger in np.flatnonzero(valid):
            n = reaction[finger] / length[finger]
            u = tangents[idx, finger, 0]
            col = u - np.dot(u, n) * n
            col_len = np.linalg.norm(col)
            if col_len < 1e-8:
                valid[finger] = False
                continue
            col /= col_len
            row = np.cross(n, col)
            row_alignment = float(np.dot(row, tangents[idx, finger, 1]))
            if abs(row_alignment) < 1e-8:
                valid[finger] = False
                continue
            if row_alignment < 0:
                row = -row
            normal[finger], t0[finger], t1[finger] = n, col, row
            centroid[finger] = weighted_position[finger] / total[finger]
            parity[finger] = 1 if np.dot(np.cross(col, row), n) > 0 else -1
            dot_pad[finger] = np.dot(n, axes[idx, finger])
            basis = np.stack([n, col, row])
            if not np.allclose(basis @ basis.T, np.eye(3), atol=1e-7):
                raise ValueError('Contact chart is not orthonormal')
            if np.dot(n, normal_force[idx, finger]) <= 0:
                raise ValueError('Away normal must follow the reaction on the pad')
        result.append(dict(normal=normal, centroid=centroid, tangent0=t0, tangent1=t1,
                           valid=valid, coherence=coherence, parity=parity, dot_pad=dot_pad,
                           count=count, pad_axis=axes[idx], pad_tangents=tangents[idx], source=source))
    return result, dict(contact_event_force_max_abs_error_n=max_force_error,
                        contact_event_vector_max_abs_error_n=max_vector_error,
                        target_geom_names=metadata['target_geom_names'],
                        matching_solver_aggregate_rows=int(matched_rows[selected].sum()),
                        native_event_fallback_rows=len(fallback_rows),
                        native_event_verification_performed=bool(len(fallback_rows)),
                        proxy_normal_force_max_abs_error_n=float(np.max(np.abs(fn[selected] - proxy_fn[selected]))) if len(selected) else 0.,
                        proxy_total_force_vector_max_abs_error_n=float(np.max(np.abs(total_force[selected] - proxy_force[selected]))) if len(selected) else 0.)


def index_episode(task):
    root, side_root, i, meta, split, episode_forces, threshold, args = task
    records, audits, exclusions = [], [], Counter()
    offsets = {k: [] for k in ('state_camera', 'tactile_camera', 'pose_camera', 'pose_tactile')}
    geometry_dots, all_coherences = [], []
    by_split, by_phase = Counter(), Counter()
    per_finger = np.zeros(5, dtype=np.int64)
    per_count = np.zeros(6, dtype=np.int64)
    if args.max_episodes:
        print(f'Preview episode {i}: reading and verifying sources', flush=True)
    cpm_path = root / f'episodes/episode_{i:06d}.h5'
    side_path = side_root / meta['file']
    raw_path = Path(meta['source_raw_h5'])
    with h5py.File(cpm_path, 'r') as ep, h5py.File(side_path, 'r') as side, h5py.File(raw_path, 'r') as raw:
        frames = ep['frame_index'][:].astype(np.int64)
        timestamps = ep['timestamp_s'][:]
        n = len(frames)
        source_n = int(ep.attrs['source_num_frames'])
        if source_n != int(meta['num_frames']) or not np.array_equal(side['frame_index'][:], np.arange(source_n)):
            raise ValueError('Source frame length/index mismatch')
        if ep['state_raw'].shape != (n, 27) or ep['actions_absolute'].shape != (n, 30, 27) or ep['actions'].shape != (n, 30, 27):
            raise ValueError('State/action shape mismatch')
        if ep['tactile_map_raw'].shape != (n, 5, 3, 7, 5):
            raise ValueError('Tactile shape mismatch')
        candidates = frames % args.stride == 0
        if np.any(frames[candidates] + 29 >= source_n):
            raise ValueError('Source anchor has no complete 30-step chunk')
        state, actions, absolute, maps = (ep[key][:] for key in ('state_raw', 'actions', 'actions_absolute', 'tactile_map_raw'))
        finite = np.isfinite(state).all(axis=1) & np.isfinite(actions).all(axis=(1, 2)) & np.isfinite(absolute).all(axis=(1, 2)) & np.isfinite(maps).all(axis=(1, 2, 3, 4))
        exclusions['nonfinite_observation_or_action'] += int((candidates & ~finite).sum())
        active = episode_forces > threshold
        contact = active.any(axis=1)
        exclusions['no_effective_contact'] += int((candidates & finite & ~contact).sum())
        rows = np.flatnonzero(candidates & finite & contact)
        source_si = ep['source_state_index'][:].astype(np.int64)
        side_si = side['source_state_index'][:][frames]
        if not np.array_equal(source_si, side_si):
            raise ValueError('CPM and tactile state-source indices disagree')
        source_ti = side['source_tactile_index'][:][frames].astype(np.int64)
        channels = strings(side['map_channels'])
        if channels != ['normal', 'tangent0', 'tangent1']:
            raise ValueError('Unexpected tactile channel order')
        links = strings(raw['tactile_contact_force/link_names'])
        right_ids = [links.index(link) for link in LINKS]
        raw_state_names = strings(raw['state/joint_names'])
        state_ids = [raw_state_names.index(name) for name in JOINTS]
        raw_state_time = raw['state/timestamp'][:]
        force_group = raw['tactile_contact_force']
        raw_tactile_time = (force_group['timestamp'][:] if 'timestamp' in force_group else raw_state_time)
        if np.any(source_si < 0) or np.any(source_si >= len(raw_state_time)) or np.any(source_ti < 0) or np.any(source_ti >= len(raw_tactile_time)):
            raise ValueError('Raw source index out of range')
        source_camera_time = ep['source_camera_timestamp_s'][:]
        native_state = raw['state/robot_joint_position'][:][:, state_ids]
        if not np.all(np.diff(raw_state_time) > 0):
            raise ValueError('Native state timestamps are not strictly increasing')
        if source_camera_time.min() < raw_state_time[0] or source_camera_time.max() > raw_state_time[-1]:
            raise ValueError('Camera state interpolation would extrapolate')
        interp_right = np.clip(np.searchsorted(raw_state_time, source_camera_time, side='right'), 1, len(raw_state_time) - 1)
        interp_left = interp_right - 1
        interp_weight = (source_camera_time - raw_state_time[interp_left]) / (raw_state_time[interp_right] - raw_state_time[interp_left])
        interpolated_state = ((1 - interp_weight[:, None]) * native_state[interp_left]
                              + interp_weight[:, None] * native_state[interp_right])
        state_error = float(np.max(np.abs(state - interpolated_state)))
        if not np.allclose(state, interpolated_state, atol=1e-6, rtol=1e-6):
            raise ValueError('CPM state does not match native joint positions interpolated to camera time')
        if args.max_episodes:
            print(f'Preview episode {i}: state interpolation passed', flush=True)
        normal_maps = force_group['normal_taxel_force_n'][:][source_ti][:, right_ids]
        tangent_maps = force_group['tangent_taxel_force_n'][:][source_ti][:, right_ids]
        native_maps = np.stack([normal_maps, tangent_maps[..., 0], tangent_maps[..., 1]], axis=2)
        map_error = float(np.max(np.abs(maps - native_maps)))
        if not np.allclose(maps, native_maps, atol=1e-6, rtol=1e-5):
            raise ValueError('CPM taxels do not match the native force recording')
        if not np.array_equal(maps, side['right_hand_tactile_map'][:][frames]):
            raise ValueError('CPM taxels and paired tactile sidecar disagree')
        if args.max_episodes:
            print(f'Preview episode {i}: tactile alignment passed', flush=True)
        source_state_time = side['source_state_timestamp_s'][:][frames]
        source_tactile_time = side['source_tactile_timestamp_s'][:][frames]
        if not np.allclose(source_state_time, raw_state_time[source_si], atol=1e-8, rtol=0) or not np.allclose(source_tactile_time, raw_tactile_time[source_ti], atol=1e-8, rtol=0):
            raise ValueError('Stored source clock/index mismatch')
        parquet = Path(ep.attrs['source_parquet'])
        provenance = pq.read_table(parquet, columns=['provenance.source_camera_frame_index', 'provenance.action_source_camera_frame_index', 'observation.state', 'action'])
        source_states = np.asarray(provenance['observation.state'].to_pylist(), dtype=np.float32)
        source_actions = np.asarray(provenance['action'].to_pylist(), dtype=np.float32)
        if not np.array_equal(state, source_states[frames]):
            raise ValueError('CPM observation states differ from source pi05 states')
        expected_absolute = source_actions[frames[:, None] + np.arange(30)]
        if not np.array_equal(absolute, expected_absolute):
            raise ValueError('CPM absolute action chunks differ from original pi05 targets')
        camera_indices = np.asarray(provenance['provenance.source_camera_frame_index'].to_pylist(), dtype=np.int64)[frames]
        action_camera_indices = np.asarray(provenance['provenance.action_source_camera_frame_index'].to_pylist(), dtype=np.int64)[frames]
        if not np.array_equal(action_camera_indices, camera_indices + 1):
            raise ValueError('Expected original next-camera-frame action target convention')
        camera = raw['cameras/head']
        if not np.allclose(camera['timestamp'][:][camera_indices], source_camera_time, atol=1e-4, rtol=0):
            raise ValueError('Source camera frame/time mismatch')
        pose_available = 'world_from_fingertip' in camera
        if not pose_available:
            raise ValueError('Native fingertip poses missing; FK fallback must be implemented explicitly')
        side_names = json.loads(camera.attrs['side_names_json'])
        finger_names = json.loads(camera.attrs['finger_names_json'])
        if finger_names not in [list(FINGERS), ['thumb', 'index', 'middle', 'ring', 'little']]:
            raise ValueError('Native fingertip pose finger order mismatch')
        right_side = side_names.index('right')
        poses = camera['world_from_fingertip'][:][camera_indices, right_side]
        pose_time = (camera['pose_timestamp'][:] if 'pose_timestamp' in camera else camera['timestamp'][:])[camera_indices]
        rotations = poses[:, :, :3, :3]
        pose_valid = np.isfinite(poses).all(axis=(1, 2, 3)) & np.all(np.isclose(poses[:, :, 3], np.array([0., 0., 0., 1.]), atol=1e-6), axis=(1, 2))
        pose_valid &= np.all(np.isclose(rotations @ rotations.transpose(0, 1, 3, 2), np.eye(3), atol=1e-6), axis=(1, 2, 3))
        pose_valid &= np.all(np.isclose(np.linalg.det(rotations), 1., atol=1e-6), axis=1)
        skew = {'state_camera': source_state_time - source_camera_time,
                'tactile_camera': source_tactile_time - source_camera_time,
                'pose_camera': pose_time - source_camera_time,
                'pose_tactile': pose_time - source_tactile_time}
        clocks_valid = np.ones(n, dtype=bool)
        for key, value in skew.items():
            offsets[key].extend(value[candidates].tolist())
            clocks_valid &= np.isfinite(value) & (np.abs(value) <= args.max_clock_skew_s)
        exclusions['clock_skew'] += int((candidates & finite & contact & ~clocks_valid).sum())
        exclusions['invalid_fingertip_pose'] += int((candidates & finite & contact & clocks_valid & ~pose_valid).sum())
        rows = rows[clocks_valid[rows] & pose_valid[rows]]
        geometries, event_audit = contact_geometry(raw, source_ti[rows], right_ids, args.minimum_normal_coherence)
        if args.max_episodes:
            print(f'Preview episode {i}: native contact geometry passed', flush=True)
        full_fn = side['right_hand_tactile_map'][:, :, 0].sum(axis=(-2, -1), dtype=np.float64)
        full_times = side['source_camera_timestamp_s'][:]
        age, remaining = durations(full_fn > threshold, full_times)
        raw_phases = raw['commands/phase'].asstr()[:] if 'commands/phase' in raw else None
        card_pose = raw['objects/card/pose_wxyz'][:] if 'objects/card/pose_wxyz' in raw else None
        head_video, wrist_video = str(ep.attrs['head_video']), str(ep.attrs['right_wrist_video'])
        if not Path(head_video).is_file() or not Path(wrist_video).is_file():
            raise FileNotFoundError('Referenced video is missing')
        kept = 0
        for row, geo in zip(rows, geometries, strict=True):
            eligible = active[row] & geo['valid']
            if not eligible.any():
                exclusions['no_eligible_contact_geometry'] += 1
                continue
            uid = f'poker_0920_200/episode_{i:06d}/frame_{int(frames[row]):06d}'
            phase = str(raw_phases[source_si[row]]) if raw_phases is not None else 'unknown'
            card = None
            if card_pose is not None:
                value = card_pose[source_si[row]]
                if value.shape == (7,) and np.isfinite(value).all() and np.isclose(np.linalg.norm(value[3:]), 1., atol=1e-5):
                    card = value.tolist()
            record = dict(anchor_uid=uid, split=split, episode_index=i, sample_row=int(row),
                          frame_index=int(frames[row]), timestamp_s=float(timestamps[row]),
                          source_camera_frame_index=int(camera_indices[row]),
                          source_action_first_camera_frame_index=int(action_camera_indices[row]),
                          source_state_index=int(source_si[row]), source_tactile_index=int(source_ti[row]),
                          state_interp_left_index=int(interp_left[row]), state_interp_right_index=int(interp_right[row]),
                          state_interp_right_weight=float(interp_weight[row]),
                          observation_state_timestamp_s=float(source_camera_time[row]),
                          source_camera_timestamp_s=float(source_camera_time[row]),
                          source_state_timestamp_s=float(source_state_time[row]),
                          source_tactile_timestamp_s=float(source_tactile_time[row]),
                          fingertip_pose_timestamp_s=float(pose_time[row]),
                          state_camera_skew_s=float(skew['state_camera'][row]),
                          tactile_camera_skew_s=float(skew['tactile_camera'][row]),
                          pose_camera_skew_s=float(skew['pose_camera'][row]),
                          pose_tactile_skew_s=float(skew['pose_tactile'][row]),
                          cpm_h5_path=str(cpm_path.resolve()), source_raw_h5=str(raw_path.resolve()),
                          source_tactile_h5=str(side_path.resolve()), head_video=head_video, right_wrist_video=wrist_video,
                          demonstration_phase=phase, geometry_source=geo['source'],
                          finger_normal_force_n=episode_forces[row].tolist(), contact_fingers=active[row].tolist(),
                          eligible_fingers=eligible.tolist(), contact_geometry_valid=geo['valid'].tolist(),
                          contact_event_count=geo['count'].tolist(), normal_coherence=geo['coherence'].tolist(),
                          contact_age_s=age[frames[row]].tolist(), contact_remaining_s=remaining[frames[row]].tolist(),
                          contact_basis_handedness=geo['parity'].tolist(),
                          away_vs_pad_normal_dot=[float(v) if np.isfinite(v) else None for v in geo['dot_pad']],
                          fingertip_site_position_world_m=poses[row, :, :3, 3].tolist(),
                          fingertip_site_rotation_world=rotations[row].reshape(5, 9).tolist(),
                          contact_centroid_world_m=vectors_or_null(geo['centroid'], geo['valid']),
                          contact_normal_away_world=vectors_or_null(geo['normal'], geo['valid']),
                          contact_tangent0_world=vectors_or_null(geo['tangent0'], geo['valid']),
                          contact_tangent1_world=vectors_or_null(geo['tangent1'], geo['valid']),
                          pad_normal_axis_world=geo['pad_axis'].tolist(),
                          pad_tangent0_world=geo['pad_tangents'][:, 0].tolist(),
                          pad_tangent1_world=geo['pad_tangents'][:, 1].tolist(),
                          card_pose_xyz_wxyz=card,
                          card_pose_timestamp_s=float(source_state_time[row]) if card is not None else None)
            records.append(record)
            by_split[split] += 1
            by_phase[phase] += 1
            per_finger += active[row]
            per_count[int(active[row].sum())] += 1
            geometry_dots.extend(geo['dot_pad'][eligible].tolist())
            all_coherences.extend(geo['coherence'][eligible].tolist())
            kept += 1
        audits.append(dict(episode_index=i, split=split, input_anchors=n,
                           stride_candidates=int(candidates.sum()), selected_anchors=kept,
                           raw_path=str(raw_path), raw_taxel_max_abs_error_n=map_error,
                           state_interp_max_abs_error_rad=state_error,
                           native_fingertip_pose_available=pose_available, card_pose_available=card_pose is not None,
                           **event_audit))
    return dict(records=records, audits=audits, exclusions=exclusions, offsets=offsets,
                geometry_dots=geometry_dots, coherences=all_coherences, by_split=by_split,
                by_phase=by_phase, per_finger=per_finger, per_count=per_count)


def build(args):
    root, side_root, output = args.data_root.resolve(), args.tactile_root.resolve(), args.output_dir.resolve()
    if not args.allow_local_output and not str(output).startswith('/nas/'):
        raise ValueError('Production output must be under /nas; --allow-local-output is for a local preview')
    staging = output.with_name(output.name + '.building')
    if output.exists() or staging.exists():
        raise FileExistsError(f'Refusing to replace existing output/staging: {output}, {staging}')
    manifest = json.loads((root / 'meta/manifest.json').read_text())
    configuration = manifest['configuration']
    if configuration['fps'] != 30 or configuration['action_horizon'] != 30 or configuration['action_dim'] != 27:
        raise ValueError('Expected the existing 30 Hz, 30x27 CPM entry')
    source_stride = int(configuration['stride'])
    if args.stride < source_stride or args.stride % source_stride:
        raise ValueError('Selection stride must be a multiple of the source CPM stride')
    splits = json.loads((root / 'meta/splits.json').read_text())['episode_indices']
    split_for = {int(i): split for split, ids in splits.items() for i in ids}
    if len(split_for) != sum(map(len, splits.values())):
        raise ValueError('Source episode splits overlap')
    side_meta = {int(m['episode_index']): m for line in (side_root / 'meta/episodes.jsonl').read_text().splitlines()
                 if line.strip() for m in [json.loads(line)]}
    ids = sorted(split_for)
    if args.max_episodes:
        ids = ids[:args.max_episodes]
    start_time = time.monotonic()
    forces, train_force = {}, []
    for i in ids:
        with h5py.File(root / f'episodes/episode_{i:06d}.h5', 'r') as episode:
            value = np.asarray(episode['tactile_map_raw'][:, :, 0], dtype=np.float64)
            if not np.isfinite(value).all() or np.any(value < 0):
                raise ValueError(f'Invalid normal maps: episode {i}')
            if str(episode.attrs['split']) != split_for[i]:
                raise ValueError('Source H5 and split manifest disagree')
            forces[i] = value.sum(axis=(-2, -1))
            if split_for[i] == 'train':
                train_force.append(forces[i])
    positive_train = np.concatenate(train_force).reshape(-1)
    positive_train = positive_train[positive_train > 1e-6]
    if not len(positive_train):
        raise ValueError('No positive training contact force for threshold calibration')
    p01 = float(np.quantile(positive_train, .01))
    automatic = max(1e-6, min(.001, .1 * p01))
    threshold = automatic if args.contact_threshold_n == 'auto' else float(args.contact_threshold_n)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Contact threshold must be finite and positive')
    print(f'Contact threshold: {threshold:.9g} N; train positive Fn p01={p01:.9g} N', flush=True)
    (staging / 'index').mkdir(parents=True)
    (staging / 'meta').mkdir()
    records, audits, exclusions = [], [], Counter()
    offsets = {k: [] for k in ('state_camera', 'tactile_camera', 'pose_camera', 'pose_tactile')}
    geometry_dots, all_coherences = [], []
    by_split, by_phase = Counter(), Counter()
    per_finger = np.zeros(5, dtype=np.int64)
    per_count = np.zeros(6, dtype=np.int64)
    split_fingerprint = digest(root / 'meta/splits.json')
    tasks = [(root, side_root, i, side_meta[i], split_for[i], forces[i], threshold, args) for i in ids]
    with (ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('spawn'))
          if args.workers > 1 else nullcontext(None)) as executor:
        parts = executor.map(index_episode, tasks, chunksize=1) if executor else map(index_episode, tasks)
        for episode_number, part in enumerate(parts, start=1):
            records.extend(part['records'])
            audits.extend(part['audits'])
            exclusions.update(part['exclusions'])
            for key in offsets:
                offsets[key].extend(part['offsets'][key])
            geometry_dots.extend(part['geometry_dots'])
            all_coherences.extend(part['coherences'])
            by_split.update(part['by_split'])
            by_phase.update(part['by_phase'])
            per_finger += part['per_finger']
            per_count += part['per_count']
            if episode_number % 10 == 0 or episode_number == len(ids):
                print(f'Indexed {episode_number}/{len(ids)} episodes; kept {len(records)} anchors', flush=True)
    if not records:
        raise ValueError('No eligible contact anchors')
    if len({r['anchor_uid'] for r in records}) != len(records):
        raise ValueError('Duplicate anchor identities')
    table_path = staging / 'index/contact_anchors.parquet'
    table = pa.Table.from_pylist(records, schema=schema())
    pq.write_table(table, table_path, compression='zstd', row_group_size=4096)
    if pq.ParquetFile(table_path).metadata.num_rows != len(records):
        raise ValueError('Parquet row-count verification failed')
    if not pq.read_table(table_path).equals(table):
        raise ValueError('Parquet full-table round-trip verification failed')
    for split in ('train', 'val'):
        indices = np.asarray([k for k, r in enumerate(records) if r['split'] == split], dtype=np.int64)
        np.savez_compressed(staging / f'index/{split}.npz', anchor_table_row=indices,
                            anchor_uid=np.asarray([records[k]['anchor_uid'] for k in indices], dtype=str),
                            episode_index=np.asarray([records[k]['episode_index'] for k in indices], dtype=np.int32),
                            sample_row=np.asarray([records[k]['sample_row'] for k in indices], dtype=np.int32),
                            frame_index=np.asarray([records[k]['frame_index'] for k in indices], dtype=np.int32))
    sensitivity = {}
    for t in sorted({1e-6, 1e-4, .001, .005, .01, .02, threshold}):
        sensitivity[f'{t:.9g}'] = {
            split: sum(int((values > t).any(axis=1).sum()) for i, values in forces.items() if split_for[i] == split)
            for split in ('train', 'val')
        }
    config = dict(schema=VERSION, stage='contact_anchor_index_only', data_root=str(root), tactile_root=str(side_root),
                  output_dir=str(output), workers=args.workers, stride=args.stride, source_stride=source_stride, fps=30,
                  action_horizon=30, action_dim=27, finger_order=list(FINGERS), joint_order=list(JOINTS),
                  tactile_channels=['normal', 'tangent0', 'tangent1'], normal_force_unit='N', position_unit='m',
                  contact_threshold_request=args.contact_threshold_n, contact_threshold_n=threshold,
                  threshold_estimated_from='training split positive per-finger Fn only',
                  automatic_threshold_formula='max(1e-6, min(1e-3, 0.1 * training_positive_Fn_p01))',
                  threshold_semantics='conservative activity threshold, not a measured sensor-noise floor',
                  train_positive_fn_quantiles_n=quantiles(positive_train),
                  minimum_normal_coherence=args.minimum_normal_coherence, max_clock_skew_s=args.max_clock_skew_s,
                  normal_direction='+n follows sign-corrected contact reaction on pad: away; -n: approach',
                  contact_centroid='normal-force-weighted solver centroid when contact count/Fn/force match admitted pad contacts; otherwise native contact-event reconstruction',
                  tangent0='native grid-col axis projected onto contact plane',
                  tangent1='orthogonal contact-plane axis sign-aligned with native grid-row direction',
                  basis_warning='contact chart may be left-handed; use native fingertip_site_rotation_world for IK orientation',
                  fingertip_position_reference='native *_site, not pad center or contact centroid',
                  state_alignment='CPM state_raw is native joint state linearly interpolated to camera timestamp; source_state_index is provenance, not its exact value',
                  card_pose_optional=True, card_pose_order='xyz,wxyz',
                  action_convention='30 absolute future targets; arm delta from current anchor + hand absolute, then original 20k quantile normalization',
                  source_split_episode_indices=splits, split_file_sha256=split_fingerprint,
                  action_norm_stats_path=str(root / 'meta/action_norm_stats.json'),
                  action_norm_stats_sha256=digest(root / 'meta/action_norm_stats.json'),
                  generation_script_sha256=digest(Path(__file__)), preview_max_episodes=args.max_episodes,
                  success_semantics='recorded demonstration anchors; phase is not an execution-success label',
                  no_perturbations_generated=True, no_model_loaded=True)
    summary = dict(schema=VERSION, input_episodes=len(ids), input_anchors=sum(a['input_anchors'] for a in audits),
                   selected_anchors=len(records), by_split=dict(by_split), by_demonstration_phase=dict(by_phase),
                   by_contact_finger_count={str(i):int(n) for i,n in enumerate(per_count)},
                   contact_anchors_per_finger=dict(zip(FINGERS, per_finger.tolist())),
                   excluded_counts=dict(exclusions), threshold_sensitivity_before_geometry=sensitivity,
                   normal_coherence=quantiles(all_coherences), away_vs_pad_normal_dot=quantiles(geometry_dots),
                   card_pose_available_anchors=sum(r['card_pose_xyz_wxyz'] is not None for r in records),
                   elapsed_seconds=time.monotonic() - start_time)
    report = dict(status='passed', schema=VERSION, episodes=audits,
                  checks=['state/native joints interpolated to camera time', 'CPM/original pi05 observation states and absolute action chunks', 'CPM/sidecar/native taxels', 'camera frame/timestamp',
                          'source state/tactile index timestamps', 'native reaction normal and matching solver contact count/Fn/force (events reconstructed for mismatches)',
                          'native fingertip SO(3)', 'contact-chart orthonormality', 'complete action chunks',
                          'inherited episode split and unique anchor UID', 'parquet row count'],
                  clock_offsets_seconds={k:quantiles(v) for k,v in offsets.items()},
                  temporal_semantics='CPM state is native post-integration state interpolated to camera time; source_state_index only identifies its nearby provenance row. Native tactile/contacts are pre-integration; cached fingertip poses carry separate pose_timestamp. Existing interpolation and remaining offsets are preserved.',
                  global_max_state_interp_error_rad=max(a['state_interp_max_abs_error_rad'] for a in audits),
                  global_max_raw_taxel_error_n=max(a['raw_taxel_max_abs_error_n'] for a in audits),
                  native_event_fallback_rows=sum(a['native_event_fallback_rows'] for a in audits),
                  matching_solver_aggregate_rows=sum(a['matching_solver_aggregate_rows'] for a in audits),
                  global_max_proxy_normal_force_error_n=max(a['proxy_normal_force_max_abs_error_n'] for a in audits),
                  global_max_proxy_force_vector_error_n=max(a['proxy_total_force_vector_max_abs_error_n'] for a in audits),
                  global_max_contact_force_error_n=max(a['contact_event_force_max_abs_error_n'] for a in audits),
                  global_max_contact_vector_error_n=max(a['contact_event_vector_max_abs_error_n'] for a in audits))
    write_json(staging / 'meta/selection_config.json', config)
    write_json(staging / 'meta/contact_summary.json', summary)
    write_json(staging / 'meta/source_alignment_report.json', report)
    (staging / 'README.md').write_text(
        '# Poker fingertip recovery: contact anchors, step 1\n\n'
        'This directory contains contact anchor indices and recorded geometry only.\n'
        'No perturbed observations, recovery actions, synthetic tactile values, or training checkpoints were generated.\n\n'
        f'- Anchors: {len(records)}; train: {by_split["train"]}; val: {by_split["val"]}.\n'
        f'- Contact: at least one right fingertip has sum(normal taxels) > {threshold:.9g} N.\n'
        '- Read index/contact_anchors.parquet; index/train.npz and val.npz reference its row numbers.\n'
        '- Use cpm_h5_path + sample_row for state_raw, tactile_map_raw, actions_absolute and actions.\n'
        '- Camera images remain in head_video/right_wrist_video at frame_index / 30 seconds.\n'
        '- Contact +normal is the force reaction on the pad (away from the object).\n'
        '- Contact tangents preserve grid-column/grid-row signs; check handedness before constructing rotations.\n'
        '- Native fingertip positions refer to sites, not contact centroids or pad centers.\n'
        '- Native state/tactile/FK clocks differ by a few milliseconds; see source_alignment_report.json.\n'
        '- state_raw is already interpolated to camera time; interpolation indices/weights are stored separately from source_state_index.\n'
        '- Original 180/20 episode split is inherited; all future variants must inherit anchor_uid and split.\n'
        '- Phase labels describe demonstration commands, not verified task outcomes.\n'
        '- Card pose is optional auxiliary geometry and is not a CPM input.\n'
        '- This is not a training-ready CPM dataset; an augmented reader is a later stage.\n'
    )
    staging.rename(output)
    print(json.dumps({'output_dir':str(output), 'selected_anchors':len(records),
                      'by_split':dict(by_split), 'contact_threshold_n':threshold}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, default=BASE / 'pi05_trex/touchguide/cpm_datasets/poker_0920_200_touchguide_v1')
    p.add_argument('--tactile-root', type=Path, default=BASE / 'pi05_trex/0920_200')
    p.add_argument('--output-dir', type=Path, default=Path('/path/to/datasets/touchguidence/augmentation/poker_fingertip_recovery_v1'))
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--workers', type=int, default=4, help='CPU processes for independent episode reads')
    p.add_argument('--contact-threshold-n', default='auto')
    p.add_argument('--minimum-normal-coherence', type=float, default=.9)
    p.add_argument('--max-clock-skew-s', type=float, default=1 / 30)
    p.add_argument('--max-episodes', type=int, default=0, help='Nonzero limits a local preview; production uses all source episodes')
    p.add_argument('--allow-local-output', action='store_true')
    args = p.parse_args()
    if args.workers < 1 or args.stride < 1 or args.max_episodes < 0 or not 0 < args.minimum_normal_coherence <= 1 or args.max_clock_skew_s <= 0:
        p.error('Invalid stride, episode limit, normal coherence, or clock tolerance')
    build(args)


if __name__ == '__main__':
    main()
