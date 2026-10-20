"""Preprocess an explicit trial selection to HDF5 + JSON, without neural inputs."""
import argparse, csv, json, hashlib
from pathlib import Path
from dataclasses import asdict
import h5py
import numpy as np
try:
    from .current import preprocess_trial, Config
    from .core import FLAGS
except ImportError:
    from current import preprocess_trial, Config
    from core import FLAGS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eye-root', type=Path, required=True, help='Directory containing Monkey/Eye_Data_session.mat')
    p.add_argument('--trials', type=Path, required=True, help='CSV: monkey,session,trial_id,epoch_ms,h1..h6; optional geometry_kind')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    with args.trials.open() as f: rows = list(csv.DictReader(f))
    required = {'monkey','session','trial_id','epoch_ms',*[f'h{k}' for k in range(1,7)]}
    if not rows or not required <= rows[0].keys(): raise ValueError(f'Required columns: {sorted(required)}')
    keys = [(r['monkey'],r['session'],int(r['trial_id'])) for r in rows]
    if len(set(keys)) != len(keys): raise ValueError('Duplicate trial IDs within session')
    root = args.eye_root.resolve(); out = args.output.resolve()
    if out == root or root in out.parents: raise ValueError('Output cannot be inside raw eye directory')
    out.mkdir(parents=True, exist_ok=False)
    cfg=Config(); records=[]
    manifest=dict(selection_sha256=hashlib.sha256(args.trials.read_bytes()).hexdigest(),
                  config=asdict(cfg),current_fixation_min_ms=75,current_max_euclidean_spread_deg=2,
                  raw_roi_radius_deg=2.5,quality_flags=FLAGS,no_interpolation=True,
                  clock_assumption='Raw time seconds from geo_present on task/photodiode clock; not independently synchronized',
                  source_files=[], scripts={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob('*.py')})
    for m,s in sorted({(r['monkey'],r['session']) for r in rows}):
        if Path(m).name != m or Path(s).name != s or m in {'.','..'} or s in {'.','..'}: raise ValueError('Unsafe source path')
        source=root/m/f'Eye_Data_{s}.mat';manifest['source_files'].append(dict(path=str(source),size=source.stat().st_size))
        with h5py.File(source) as f, h5py.File(out/f'{m}_{s}.h5','w') as dest:
            dest.attrs['source_eye_file']=str(source);dest.attrs['timestamp_unit']='ms from geo_present'
            refs={k:f['Eye_Data'][k][:].ravel() for k in ['eye_x','eye_y','pupil_size']}
            for r in rows:
                if (r['monkey'],r['session']) != (m,s): continue
                tid=int(r['trial_id'])
                if tid<1 or any(tid>len(v) for v in refs.values()): raise ValueError((m,s,tid,'Missing eye cell'))
                raw=[f[refs[k][tid-1]][:] for k in ['eye_x','eye_y','pupil_size']]
                h=[float(r[f'h{k}']) for k in range(1,7)];epoch=float(r['epoch_ms'])
                try: result=preprocess_trial(*raw,h,epoch,cfg)
                except Exception as e: raise RuntimeError(f'{m}/{s}/trial {tid}: {e}') from e
                g=dest.create_group(str(tid));g.attrs['epoch_end_ms']=epoch;g.attrs['saccade_status']=result['saccade_status'];g.attrs['geometry']=h
                q=result['quality']
                for k in ['t','x','y','pupil','y_index','pupil_index','y_residual_ms','pupil_residual_ms','flags','valid','segment']:
                    g.create_dataset(k,data=q[k],compression='gzip')
                g.create_dataset('epoch_segment',data=result['epoch_segment'],compression='gzip')
                g.create_dataset('velocity',data=result['velocity'],compression='gzip');g.attrs['gap_threshold_ms']=q['gap_ms']
                records.append(dict(monkey=m,session=s,trial_id=tid,geometry_kind=r.get('geometry_kind','unspecified'),
                                    saccade_status=result['saccade_status'],looks=result['looks'],saccades=result['saccades']))
                print(m,s,tid,'complete',flush=True)
    (out/'events.json').write_text(json.dumps(records,indent=2))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
