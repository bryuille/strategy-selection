"""Run timestamp-native preprocessing for Figure Five's current top two sessions/monkey."""
import argparse,csv,gzip,hashlib,json,sys,platform,importlib.metadata
from pathlib import Path
from dataclasses import asdict
import h5py
import numpy as np
from core import Config,FLAGS,ROI,ROI_NAMES,quality,detect,runs

FIELDS=['geo_present','fix_start','flash_one','flash_two','flash_three','photodiode_qc_bad',
        'geo_type','path_type','trial_fade','h1','h2','h3','h4','h5','h6']

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def ranking(path):
    with path.open() as f:rows=list(csv.DictReader(f))
    required={'monkey','session','cv_mahalanobis','status'}
    if not rows or not required<=rows[0].keys():raise ValueError('Invalid Figure Five ranking file')
    selected=[]
    for monkey in ['Faure','Nielsen']:
        valid=[r for r in rows if r['monkey']==monkey and r['status'].startswith('passed') and np.isfinite(float(r['cv_mahalanobis']))]
        if len({r['session'] for r in valid})!=len(valid):raise ValueError('Duplicate session in ranking')
        valid.sort(key=lambda r:-float(r['cv_mahalanobis'])) # stable ties: same input order as MATLAB
        if len(valid)<2:raise ValueError(f'Fewer than two eligible sessions for {monkey}')
        selected.extend(dict(r,rank_within_monkey=i+1) for i,r in enumerate(valid[:2]))
    return selected

def savecsv(path,rows):
    if not rows:path.write_text('');return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def behavior(path):
    with h5py.File(path,'r') as f:
        g=f['save_all_data']; ids=np.array(g['trial_indices_all']).ravel()
        if not np.isfinite(ids).all() or not (ids==ids.astype(int)).all():raise ValueError('Invalid trial IDs')
        unique,first,inv=np.unique(ids.astype(int),return_index=True,return_inverse=True)
        result={}
        for field in FIELDS:
            a=np.array(g[field]).ravel()
            if len(a)!=len(ids):raise ValueError((path,field,'shape mismatch'))
            vals=a[first]; same=(a==vals[inv])|(np.isnan(a)&np.isnan(vals[inv]))
            if not same.all():raise ValueError((path,field,'neuron-row mismatch',np.flatnonzero(~same)[:10].tolist()))
            result[field]=vals
    return unique,result

def rawcell(f,refs,tid):
    a=np.array(f[refs[tid-1]],dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=2:raise ValueError(('Unexpected eye shape',tid,a.shape))
    return a

def interval_rows(q,key,source,epoch_end):
    out=[]; t=q['t']
    for name,bit in FLAGS.items():
        for r in runs((q['flags']&bit)!=0):
            out.append(dict(key,source_file=source,reason=name,field='Eye_Data/eye_x (anchor); matched Y/pupil indices in samples table',
                start_index_0based=int(r[0]),end_index_0based=int(r[-1]),start_ms=float(t[r[0]]),end_ms=float(t[r[-1]]),
                intersects_epoch=bool(np.any(np.isfinite(t[r])&(t[r]>=0)&(t[r]<=epoch_end)))))
    dt=np.diff(t)
    for i in np.flatnonzero(~np.isfinite(dt)|(dt<=0)|(dt>q['gap_ms'])):
        out.append(dict(key,source_file=source,reason='timestamp_discontinuity',field='Eye_Data/eye_x',
            start_index_0based=int(i),end_index_0based=int(i+1),start_ms=float(t[i]),end_ms=float(t[i+1]),
            intersects_epoch=bool(np.isfinite(dt[i]) and t[i]<epoch_end and t[i+1]>0),threshold_ms=float(q['gap_ms'])))
    return out

def run_session(row,root,out,cfg):
    m,s=row['monkey'],row['session']; dest=out/f'{m}_{s}';dest.mkdir()
    paths={'behavior':root/'Behavioral_Data'/m/f'{s}_good_trials_concat.mat',
           'eye':root/'Eye_Data'/m/f'Eye_Data_{s}.mat',
           'range':root/'Single_Trial_List'/m/f'single_trial_list_{s}.mat'}
    provenance={k:dict(path=str(p),sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()}
    ids,b=behavior(paths['behavior'])
    with h5py.File(paths['range'],'r') as f:lo=int(np.array(f['min_value']).squeeze());hi=int(np.array(f['max_value']).squeeze())
    ledger=[];trials=[];events=[];intervals=[];segs=[];align=[]
    keys=['monkey','session','trial_id','source_index_0based','t_ms','x_deg','y_deg','pupil','y_source_index_0based',
          'pupil_source_index_0based','y_time_residual_ms','pupil_time_residual_ms','quality_flags','full_trial_segment','epoch_segment',
          'velocity_x_deg_s','velocity_y_deg_s','unit_x','unit_y']
    sample_count=0;example_clean=None;example_gap=None;maps=[]
    with h5py.File(paths['eye'],'r') as f,h5py.File(dest/'samples.h5','w') as hf,gzip.open(dest/'sample_quality.csv.gz','wt',newline='') as sf:
        hf.attrs['source_eye_file']=str(paths['eye']);hf.attrs['timestamp_unit']='milliseconds from geo_present'
        writer=csv.writer(sf);writer.writerow(keys)
        refs={k:np.array(f['Eye_Data'][k]).ravel() for k in ['eye_x','eye_y','pupil_size']}
        for j,tid in enumerate(ids):
            key=dict(monkey=m,session=s,trial_id=int(tid))
            reason=[]
            if not lo<=tid<=hi:reason.append('outside_published_range')
            if not np.isfinite(b['photodiode_qc_bad'][j]) or b['photodiode_qc_bad'][j]!=0:reason.append('photodiode_qc')
            if not np.isfinite(b['trial_fade'][j]) or b['trial_fade'][j]!=0:reason.append('faded_or_unknown')
            if not np.isfinite(b['path_type'][j]) or b['path_type'][j]==-99 or b['geo_type'][j] not in range(1,7):reason.append('noncanonical')
            er={name:float((b[name][j]-b['geo_present'][j])*1000) for name in ['fix_start','flash_one','flash_two','flash_three']}
            if not np.isfinite(list(er.values())).all() or not 0<er['fix_start']<=er['flash_one']<er['flash_two']<er['flash_three']:reason.append('invalid_event_order')
            if any(tid>len(v) for v in refs.values()):reason.append('missing_eye_cell')
            h=[b[f'h{k}'][j] for k in range(1,7)]
            if not np.isfinite(h).all() or min(h)<=0:reason.append('invalid_geometry')
            ledger.append(dict(key,selected=not reason,reason=';'.join(reason) or 'selected',maze=b['geo_type'][j]))
            if reason:continue
            raws={name:rawcell(f,v,tid) for name,v in refs.items()}
            q=quality(raws['eye_x'],raws['eye_y'],raws['pupil_size'],cfg)
            d=detect(q,er['fix_start'],h,cfg); win=d['in_epoch']; ix=np.flatnonzero(win)
            if not len(ix):
                trials.append(dict(key,maze=int(b['geo_type'][j]),status='no_epoch_samples',epoch_ms=er['fix_start'],usable_ms=0.));continue
            for name,time in dict(maze_onset=0.,**er).items():
                ft=np.flatnonzero(np.isfinite(q['t']))
                closest=int(ft[np.argmin(np.abs(q['t'][ft]-time))]) if len(ft) else -1
                align.append(dict(key,event=name,event_time_ms=time,nearest_source_index_0based=closest,
                    nearest_sample_ms=float(q['t'][closest]) if closest>=0 else np.nan,
                    residual_ms=float(q['t'][closest]-time) if closest>=0 else np.nan,
                    sample_usable=bool(q['valid'][closest]) if closest>=0 else False))
            usable_ms=0.
            for sid in np.unique(d['segment'][d['segment']>=0]):
                ii=np.flatnonzero(d['segment']==sid); dur=float(q['t'][ii[-1]]-q['t'][ii[0]]);usable_ms+=dur
                segs.append(dict(key,segment=int(sid),start_index_0based=int(ii[0]),end_index_0based=int(ii[-1]),
                    onset_ms=float(q['t'][ii[0]]),offset_ms=float(q['t'][ii[-1]]),duration_ms=dur,n_samples=len(ii)))
            iirows=interval_rows(q,key,str(paths['eye']),er['fix_start']); intervals.extend(iirows)
            these=[dict(key,maze=int(b['geo_type'][j]),detector_status=d['status'],**ev) for ev in d['events']];events.extend(these)
            fix=[e for e in these if e['kind']=='fixation']; assigned=[e for e in fix if e['roi_code']>=0]
            rowtrial=dict(key,maze=int(b['geo_type'][j]),status=d['status'],epoch_ms=er['fix_start'],n_epoch_samples=len(ix),
                usable_samples=int((win&q['valid']).sum()),usable_ms=usable_ms,observed_fraction=usable_ms/er['fix_start'],
                gap_threshold_ms=q['gap_ms'],n_segments=len(np.unique(d['segment'][d['segment']>=0])),
                gaps_in_epoch=sum(r['reason']=='timestamp_discontinuity' and r['intersects_epoch'] for r in iirows),
                n_fixations=len(fix),n_saccades=sum(e['kind']=='saccade' for e in these),
                assigned_fixation_ms=sum(e['duration_ms'] for e in assigned),
                velocity_threshold_x=float(d['threshold'][0]),velocity_threshold_y=float(d['threshold'][1]))
            trials.append(rowtrial)
            for k,name in enumerate(ROI_NAMES):rowtrial[f'dwell_{name}_ms']=sum(e['duration_ms'] for e in assigned if e['roi_code']==k)
            group=hf.create_group(str(tid));group.attrs['maze']=int(b['geo_type'][j]);group.attrs['gap_threshold_ms']=q['gap_ms']
            group.attrs['epoch_end_ms']=er['fix_start'];group.attrs['detector_status']=d['status']
            datasets={'source_index_0based':ix,**{name:q[name][ix] for name in ['t','x','y','pupil','y_index','pupil_index','y_residual_ms','pupil_residual_ms','flags','segment']},
                      'epoch_segment':d['segment'][ix],'velocity':d['velocity'][ix],'unit_xy':d['normalized'][ix]}
            for name,values in datasets.items():group.create_dataset(name,data=values,compression='gzip')
            # Original pupil samples are stored on THEIR timestamps; mapped pupil is an observed match, never interpolation.
            pp=raws['pupil_size']; group.create_dataset('native_pupil_epoch',data=pp[(pp[:,1]>=0)&(pp[:,1]*1000<=er['fix_start'])],compression='gzip')
            for i in ix:
                writer.writerow([m,s,tid,i,q['t'][i],q['x'][i],q['y'][i],q['pupil'][i],q['y_index'][i],q['pupil_index'][i],
                    q['y_residual_ms'][i],q['pupil_residual_ms'][i],q['flags'][i],q['segment'][i],d['segment'][i],
                    *d['velocity'][i],*d['normalized'][i]])
            sample_count+=len(ix)
            for ev in fix:maps.append((int(b['geo_type'][j]),ev['centroid_unit_x'],ev['centroid_unit_y'],ev['roi_code'],ev['duration_ms']))
            pack=dict(key=key,q=q,d=d,h=h,event_times=er,maze=int(b['geo_type'][j]),summary=rowtrial)
            if example_clean is None or rowtrial['observed_fraction']>example_clean['summary']['observed_fraction']:example_clean=pack
            if rowtrial['gaps_in_epoch'] and (example_gap is None or rowtrial['usable_ms']>example_gap['summary']['usable_ms']):example_gap=pack
            if len(trials)%50==0:print(f'{m} {s}: {len(trials)} selected trials processed',flush=True)
    for name,data in [('trial_selection',ledger),('trials',trials),('events',events),('segments',segs),('quality_intervals',intervals),('event_alignment',align)]:savecsv(dest/f'{name}.csv',data)
    for name,p in paths.items():
        if p.stat().st_size!=provenance[name]['size'] or p.stat().st_mtime_ns!=provenance[name]['mtime_ns']:raise RuntimeError(f'Source changed: {p}')
    summary=dict(monkey=m,session=s,selected_trials=len(trials),epoch_samples=sample_count,
        detector_statuses={k:sum(r['status']==k for r in trials) for k in sorted({r['status'] for r in trials})},
        fixations=sum(e['kind']=='fixation' for e in events),saccades=sum(e['kind']=='saccade' for e in events),
        mean_observed_fraction=float(np.mean([r.get('observed_fraction',0.) for r in trials])),provenance=provenance)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    from visual_qc import session_map,trial_plot,movie
    session_map(m,s,maps,cfg,dest/'normalized_H_ROIs.png')
    movie_info=[]
    for name,example in [('clean',example_clean),('interrupted',example_gap)]:
        if example is not None:
            trial_plot(example,cfg,dest/f'{name}_trial_{example["key"]["trial_id"]}.png')
    # One illustrative movie per ranked session, alternating clean/interrupted.
    example=example_clean if row['rank_within_monkey']==1 else (example_gap or example_clean)
    if example is not None:
        movie_info=movie(example,cfg,dest/f'QC_trial_{example["key"]["trial_id"]}.mp4')
    (dest/'movie_metadata.json').write_text(json.dumps(movie_info,indent=2))
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--paper-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gap-median-factor',type=float,default=5.);p.add_argument('--gap-p90-factor',type=float,default=3.)
    a=p.parse_args(); paper=a.paper_root.resolve();out=a.output.resolve();raw=paper/'Data/Physiology'
    if raw==out or raw in out.parents:raise ValueError('Output inside raw data')
    cfg=Config(gap_median_factor=a.gap_median_factor,gap_p90_factor=a.gap_p90_factor)
    if cfg.gap_median_factor<=0 or cfg.gap_p90_factor<=0:raise ValueError('Gap factors must be positive')
    rank=paper/'Figure_Five/01_Session_Ranking/Outputs/all_session_mahalanobis_results.csv';selected=ranking(rank)
    out.mkdir(parents=True,exist_ok=False);savecsv(out/'selected_sessions.csv',selected)
    manifest=dict(paper_root=str(paper),ranking_path=str(rank),ranking_sha256=sha(rank),config=asdict(cfg),quality_flag_bits=FLAGS,
        python=sys.version,packages={name:importlib.metadata.version(name) for name in ['numpy','h5py','matplotlib','imageio-ffmpeg']},
        script_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},
        clock_assumption='Eye timestamps are seconds relative to geo_present on the task/photodiode-aligned clock, per user; no clock refit.',
        window='maze onset (0) through fixation onset; flash_one/two/three measured photodiode event times retained for QC',
        no_interpolation=True,neural_data_loaded=False,sessions=[])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for row in selected:
        print(f'Selected {row["monkey"]} {row["session"]} rank {row["rank_within_monkey"]}',flush=True)
        manifest['sessions'].append(run_session(row,raw,out,cfg));(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Completed all dynamically ranked sessions',flush=True)

if __name__=='__main__':main()
