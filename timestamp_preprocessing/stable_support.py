"""Experimental 75-ms stable support using exact maximum pairwise distance <=2 deg.
One look per usable interval between saved detected saccades. No interpolation.
"""
import numpy as np
from scipy.spatial.distance import cdist
CENTERS=np.array([[0.,0.],[-1.,1.],[-1.,-1.],[1.,1.],[1.,-1.]])
NAMES=['origin','LU','LD','RU','RD']
def runs(mask):
 ix=np.flatnonzero(mask)
 return np.split(ix,np.flatnonzero(np.diff(ix)>1)+1) if len(ix) else []
def ends_for_diameter(xy,diameter=2.):
 n=len(xy);d2=cdist(xy,xy,'sqeuclidean');ends=np.empty(n,int);left=0
 for right in range(n):
  bad=np.flatnonzero(d2[left:right,right]>diameter**2)
  if len(bad):
   newleft=left+int(bad[-1])+1;ends[left:newleft]=right-1;left=newleft
 ends[left:]=n-1
 return ends,d2

def detect_looks(q,normalized,epoch_end,events):
 t=q['t'];pos=np.c_[q['x'],q['y']];sac=np.zeros(len(t),bool)
 for e in events:
  if e['kind']=='saccade':sac|=(t>=float(e['onset_ms']))&(t<=float(e['offset_ms']))
 eligible=q['valid']&np.isfinite(t)&(t>=0)&(t<=epoch_end)&~sac;output=[]
 for sid in np.unique(q['segment'][eligible]):
  for ix in runs(eligible&(q['segment']==sid)):
   assert np.isfinite(pos[ix]).all() and np.all(np.diff(t[ix])>0)
   if t[ix[-1]]-t[ix[0]]<75:continue
   ends,d2=ends_for_diameter(pos[ix]);good=np.flatnonzero(t[ix[ends]]-t[ix]>=75)
   if not len(good):continue
   change=np.zeros(len(ix)+1,int);np.add.at(change,good,1);np.add.at(change,ends[good]+1,-1);support=np.cumsum(change[:-1])>0;islands=runs(support);weights=np.zeros(len(ix));intervals=[];islandrois=[]
   for z in islands:
    dt=np.diff(t[ix[z]]);weights[z[:-1]]+=dt/2;weights[z[1:]]+=dt/2
    intervals.append([float(t[ix[z[0]]]),float(t[ix[z[-1]]])])
   center=np.average(normalized[ix],axis=0,weights=weights);distance=np.linalg.norm(CENTERS-center,axis=1);roi=NAMES[int(distance.argmin())] if distance.min()<=1 else 'unassigned'
   for z in islands:
    c=np.average(normalized[ix[z]],axis=0,weights=weights[z]);d=np.linalg.norm(CENTERS-c,axis=1);islandrois.append(NAMES[int(d.argmin())] if d.min()<=1 else 'unassigned')
   ss=ix[support];assert not sac[ix].any()
   # Independently validate a witness window satisfying the actual pairwise criterion.
   a=int(good[np.argmax(t[ix[ends[good]]]-t[ix[good]])]);b=int(ends[a]);diam=float(np.sqrt(d2[a:b+1,a:b+1].max()));assert diam<=2 and t[ix[b]]-t[ix[a]]>=75
   output.append(dict(kind='qualified_look',onset_ms=float(t[ss[0]]),offset_ms=float(t[ss[-1]]),duration_ms=float(weights.sum()),centroid_unit_x=float(center[0]),centroid_unit_y=float(center[1]),roi=roi,segment=int(sid),candidate_start_ms=float(t[ix[0]]),candidate_end_ms=float(t[ix[-1]]),start_index=int(ss[0]),end_index=int(ss[-1]),stable_intervals=intervals,stable_island_rois=islandrois,ambiguous_roi=len(set(islandrois))>1,witness_start_ms=float(t[ix[a]]),witness_end_ms=float(t[ix[b]]),witness_diameter_deg=diam))
 return sorted(output,key=lambda e:e['onset_ms'])
