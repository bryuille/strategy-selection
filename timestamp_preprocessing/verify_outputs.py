"""Validate saved results against original observations and continuous segments."""
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np
import imageio_ffmpeg

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    manifest=json.loads((a.output/'manifest.json').read_text());results=[]
    assert len(manifest['sessions'])==4,'Run incomplete'
    for s in manifest['sessions']:
        path=a.output/f'{s["monkey"]}_{s["session"]}'
        events=list(csv.DictReader((path/'events.csv').open()));bytrial={}
        for e in events:bytrial.setdefault(int(e['trial_id']),[]).append(e)
        checked=0;events_checked=0
        with h5py.File(s['provenance']['eye']['path'],'r') as raw,h5py.File(path/'samples.h5','r') as f:
            refs={k:np.array(raw['Eye_Data'][k]).ravel() for k in ['eye_x','eye_y','pupil_size']}
            for trial in f:
                tid=int(trial);g=f[trial];idx=g['source_index_0based'][()]; t=g['t'][()];flags=g['flags'][()];seg=g['epoch_segment'][()]
                xraw=np.array(raw[refs['eye_x'][tid-1]]);yraw=np.array(raw[refs['eye_y'][tid-1]]);praw=np.array(raw[refs['pupil_size'][tid-1]])
                np.testing.assert_array_equal(g['x'][()],xraw[idx,0]);np.testing.assert_array_equal(t,xraw[idx,1]*1000)
                for name,native,indexname in [('y',yraw,'y_index'),('pupil',praw,'pupil_index')]:
                    match=g[indexname][()];values=g[name][()];ok=match>=0
                    np.testing.assert_array_equal(values[ok],native[match[ok],0]);assert np.isnan(values[~ok]).all()
                    assert np.max(np.abs(t[ok]-native[match[ok],1]*1000),initial=0)<=manifest['config']['match_tolerance_ms']
                assert ((t>=0)&(t<=g.attrs['epoch_end_ms'])).all(); assert (flags[seg>=0]==0).all()
                for sid in np.unique(seg[seg>=0]):
                    tt=t[seg==sid];dt=np.diff(tt)
                    assert ((dt>0)&(dt<=g.attrs['gap_threshold_ms'])).all()
                lookup={int(source):j for j,source in enumerate(idx)}
                for e in bytrial.get(tid,[]):
                    start,end=lookup[int(e['start_index'])],lookup[int(e['end_index'])]
                    assert np.all(seg[start:end+1]==int(e['segment'])) and int(e['segment'])>=0
                    assert np.isfinite(g['x'][start:end+1]).all() and np.isfinite(g['y'][start:end+1]).all()
                    assert float(e['duration_ms'])==t[end]-t[start]
                    minimum=manifest['config']['saccade_min_ms'] if e['kind']=='saccade' else manifest['config']['fixation_min_ms']
                    assert float(e['duration_ms'])>=minimum
                    events_checked+=1
                checked+=len(idx)
        movies=[]
        for movie in path.glob('*.mp4'):
            frames,secs=imageio_ffmpeg.count_frames_and_secs(str(movie))
            meta=json.loads((path/'movie_metadata.json').read_text())
            assert frames==meta['frames'] and abs(secs-frames/meta['fps'])<.1
            movies.append(dict(file=movie.name,frames=frames,duration_seconds=secs))
        assert len(movies)==1
        results.append(dict(monkey=s['monkey'],session=s['session'],samples_exactly_matched=checked,
                            events_verified_not_crossing_invalid_segments=events_checked,movies=movies))
    result=dict(status='passed',sessions=results)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
