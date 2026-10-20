"""Validate current looks against saved paper outputs; no results distributed."""
import argparse,csv,json
from pathlib import Path
import h5py,numpy as np
from current import qualified_looks
p=argparse.ArgumentParser();p.add_argument('--paper-root',type=Path,required=True);p.add_argument('--canonical-samples',type=Path,required=True);a=p.parse_args();E=a.paper_root/'Eye_Strategy_Analysis'
def read(p):
 with p.open() as f:return list(csv.DictReader(f))
rows=read(E/'outputs/canonical_trial_features.csv')+read(E/'random_timestamp_outputs/random_trial_features.csv')
expected={}
for r in read(E/'raw_roi_normalized_fixations/looks.csv'):expected.setdefault((r['monkey'],r['session'],r['geometry_kind'],int(r['trial_id'])),[]).append(r)
count=0;trials=0;maxerr=0
for kind in ['canonical','random']:
 for m,s in sorted({(r['monkey'],r['session']) for r in rows}):
  folder=(a.canonical_samples if kind=='canonical' else E/'random_timestamp_outputs')/f'{m}_{s}';events={}
  for e in read(folder/'events.csv'):events.setdefault(int(e['trial_id']),[]).append(e)
  with h5py.File(folder/'samples.h5') as f:
   for r in rows:
    if (r['geometry_kind'],r['monkey'],r['session'])!=(kind,m,s):continue
    tid=int(r['trial_id']);g=f[str(tid)];seg=g['epoch_segment'][:];q=dict(t=g['t'][:],x=g['x'][:],y=g['y'][:],valid=seg>=0,segment=seg)
    looks=qualified_looks(q,[float(r[f'h{k}']) for k in range(1,7)],float(g.attrs['epoch_end_ms']),events.get(tid,[]));old=expected.get((m,s,kind,tid),[])
    assert len(looks)==len(old),(m,s,kind,tid,'count')
    for x,y in zip(looks,old):
     assert x['roi']==y['roi'] and x['island_roi_disagreement']==(y['island_roi_disagreement']=='True'),(m,s,tid,'ROI')
     assert x['stable_intervals']==json.loads(y['stable_intervals'])
     for k in ['raw_x','raw_y','normalized_x','normalized_y','stable_duration_ms','onset_ms','offset_ms']:
      err=abs(x[k]-float(y[k]));maxerr=max(maxerr,err);assert err<1e-7,(m,s,tid,k,err)
    count+=len(looks);trials+=1
  print(kind,m,s,'verified',flush=True)
print(json.dumps(dict(trials=trials,looks=count,max_absolute_error=maxerr)))
