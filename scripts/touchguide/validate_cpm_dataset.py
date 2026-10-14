"""Validate stored CPM trajectories, provenance, and negative donor pools."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
import pyarrow.parquet as pq


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(absolute, state, lo, hi):
    delta = absolute.copy()
    delta[..., :7] -= state[..., None, :7]
    return 2*(delta-lo)/(hi-lo+1e-6)-1


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    m=json.loads((root/'meta/manifest.json').read_text())
    config=m['configuration']
    assert m['schema']=='touchguide_cpm_dataset_v1'
    assert not m['f6_included'] and not m['tactile_history_included']
    norm=root/'meta/action_norm_stats.json'
    original=Path(m['sources']['base_checkpoint'])/'assets/normalizer/norm_stats.json'
    assert sha(norm)==sha(original)==m['sources']['normalizer_sha256']
    stats=json.loads(norm.read_text())['norm_stats']['actions']
    lo,hi=(np.asarray(stats[k],np.float32) for k in ('q01','q99'))
    active=hi-lo>1e-6
    sets={k:set(m['splits'][k]['episode_indices']) for k in ('train','val')}
    assert len(sets['train'])==180 and len(sets['val'])==20
    assert not sets['train']&sets['val'] and sets['train']|sets['val']==set(range(200))
    report={'schema':'touchguide_cpm_validation_v1','passed':False,'dataset_root':str(root),'splits':{}}
    indexes={}
    chunks={}
    states={}
    for split in ('train','val'):
        with np.load(root/f'index/{split}.npz') as f:
            index={k:f[k] for k in f.files}
        ids,frames=index['episode_index'],index['frame_index']
        assert set(ids.tolist())==sets[split]
        assert len(ids)==m['splits'][split]['count']
        indexes[split]=index
        all_chunks=[]
        all_states=[]
        source_checked=0
        raw_max=0.
        for eid in sorted(sets[split]):
            positions=np.flatnonzero(ids==eid)
            with h5py.File(root/f'episodes/episode_{eid:06d}.h5','r') as f:
                n=int(f.attrs['source_num_frames'])
                expected=np.arange(0,n-29,config['stride'])
                np.testing.assert_array_equal(f['frame_index'][:],expected)
                np.testing.assert_array_equal(frames[positions],expected)
                np.testing.assert_array_equal(index['sample_row'][positions],np.arange(len(expected)))
                assert f['actions'].shape==(len(expected),30,27)
                assert f['tactile_map_raw'].shape==(len(expected),5,3,7,5)
                assert not any('f6' in key or 'history' in key for key in f)
                for camera in ('head_video','right_wrist_video'):
                    assert Path(f.attrs[camera]).is_file()
                absolute=f['actions_absolute'][:]
                state=f['state_raw'][:]
                stored=f['actions'][:]
                maps=f['tactile_map_raw'][:]
                assert np.isfinite(absolute).all() and np.isfinite(state).all() and np.isfinite(stored).all() and np.isfinite(maps).all()
                np.testing.assert_allclose(stored,normalize(absolute,state,lo,hi),rtol=0,atol=1e-6)
                np.testing.assert_array_equal(f['contact_fingers'][:],np.any(np.abs(maps)>config['contact_taxel_threshold_newtons'],axis=(2,3,4)))
                raw_max=max(raw_max,float(np.abs(maps).max()))
                # Three independent source comparisons per episode, covering
                # beginning, middle and final complete action chunk.
                rows=np.unique([0,len(expected)//2,len(expected)-1])
                source=pq.read_table(f.attrs['source_parquet'])
                source_actions=np.asarray(source['action'].to_pylist(),np.float32)
                source_states=np.asarray(source['observation.state'].to_pylist(),np.float32)
                src_index=np.asarray(source['provenance.source_state_index'].to_pylist(),np.int64)
                src_time=np.asarray(source['provenance.source_camera_timestamp_s'].to_pylist(),np.float64)
                with h5py.File(f.attrs['source_tactile'],'r') as side:
                    for row in rows:
                        anchor=int(expected[row])
                        np.testing.assert_array_equal(absolute[row],source_actions[anchor:anchor+30])
                        np.testing.assert_array_equal(state[row],source_states[anchor])
                        np.testing.assert_array_equal(maps[row],side['tactile_map'][anchor,5:])
                        assert f['source_state_index'][row]==src_index[anchor]==side['source_state_index'][anchor]
                        assert abs(float(f['source_camera_timestamp_s'][row])-src_time[anchor])<1e-4
                        source_checked+=1
                all_chunks.append(absolute)
                all_states.append(state)
        chunks[split]=np.concatenate(all_chunks)
        states[split]=np.concatenate(all_states)
        with np.load(root/f'negatives/{split}.npz') as f:
            pool=f['negative_sample_index']
            saved_margins=f['minimum_normalized_rmse']
        assert pool.shape==(len(ids),config['negative_pool_size'])
        assert pool.min()>=0 and pool.max()<len(ids)
        assert np.all(np.diff(np.sort(pool,axis=1),axis=1)>0)
        assert np.all((ids[pool]!=ids[:,None])|(np.abs(frames[pool]-frames[:,None])>=30))
        minimum=np.inf
        # Vectorized full verification of every candidate, rather than only a
        # spot check of the generated donor indices.
        for first in range(0,len(ids),128):
            last=min(first+128,len(ids))
            anchor_states=states[split][first:last]
            positive=normalize(chunks[split][first:last],anchor_states,lo,hi)
            candidate_absolute=chunks[split][pool[first:last]]
            negative=normalize(candidate_absolute,anchor_states[:,None,:],lo,hi)
            distances=np.sqrt(np.mean((negative[...,active]-positive[:,None,...,active])**2,axis=(-1,-2)))
            assert np.isfinite(negative).all()
            assert np.all(distances>=config['minimum_negative_rmse']-1e-6)
            np.testing.assert_allclose(saved_margins[first:last],distances.min(axis=1),rtol=0,atol=1e-6)
            minimum=min(minimum,float(distances.min()))
        report['splits'][split]={'episodes':len(sets[split]),'samples':len(ids),
            'negative_pairs_verified':int(pool.size),'minimum_negative_rmse':minimum,
            'source_records_verified':source_checked,'raw_tactile_max_abs_newtons':raw_max}
        print(f'Verified split {split}: {len(ids)} samples, {pool.size} donor pairs',flush=True)
    report['passed']=True
    (root/'meta/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
