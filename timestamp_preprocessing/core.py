"""Timestamp-native eye preprocessing. No resampling, filling, or temporal interpolation."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass(frozen=True)
class Config:
    match_tolerance_ms: float = .002
    gap_median_factor: float = 5.
    gap_p90_factor: float = 3.
    position_limit_deg: float = 30.
    blink_min_ms: float = 20.
    blink_merge_ms: float = 10.
    blink_padding_ms: float = 50.
    fixation_dispersion_deg: float = 2.
    fixation_min_ms: float = 100.
    velocity_half_window_ms: float = 4.
    velocity_min_span_ms: float = 2.
    saccade_factor: float = 8.
    saccade_min_ms: float = 12.
    saccade_merge_ms: float = 40.
    roi_radius: float = 1.

FLAGS = {'bad_timestamp':1,'nonfinite_x':2,'unmatched_y':4,'nonfinite_y':8,
         'outside_30deg':16,'unmatched_pupil':32,'invalid_pupil':64,'blink_padding':128,
         'nonincreasing_timestamp':256}
ROI_NAMES = ['origin','LU','LD','RU','RD']
ROI = np.array([[0.,0.],[-1.,1.],[-1.,-1.],[1.,1.],[1.,-1.]])

def runs(mask):
    z=np.flatnonzero(mask)
    return np.split(z,np.flatnonzero(np.diff(z)>1)+1) if z.size else []

def gap_threshold(t, cfg):
    d=np.diff(t); d=d[np.isfinite(d)&(d>0)]
    if len(d)<10:
        return np.nan
    # Central 90% reduces influence of true interruptions; no nominal Hz input.
    central=d[d<=np.quantile(d,.9)]
    return max(cfg.gap_median_factor*np.median(central),cfg.gap_p90_factor*np.quantile(central,.9))

def match(target, source, tolerance):
    """Nearest actual observation, with unique matches and explicit residuals."""
    result=np.full(len(target),-1,dtype=int); residual=np.full(len(target),np.nan)
    finite=np.flatnonzero(np.isfinite(source))
    if not len(finite): return result,residual
    order=finite[np.argsort(source[finite],kind='stable')]; st=source[order]
    k=np.searchsorted(st,target); left=np.clip(k-1,0,len(st)-1); right=np.clip(k,0,len(st)-1)
    j=np.where(np.abs(st[left]-target)<=np.abs(st[right]-target),left,right)
    chosen=order[j]; diff=source[chosen]-target
    _, inv, counts=np.unique(st,return_inverse=True,return_counts=True)
    ok=np.isfinite(target)&(np.abs(diff)<=tolerance)&(counts[inv[j]]==1)
    # If two X observations claim one Y/pupil observation, retain neither as a pair.
    claimed=chosen[ok]; unique,n=np.unique(claimed,return_counts=True)
    ok &= ~np.isin(chosen,unique[n>1])
    result[ok]=chosen[ok]; residual[ok]=diff[ok]
    return result,residual

def segments(t,valid,gap):
    ids=np.full(len(t),-1,dtype=int); sid=-1
    for i in np.flatnonzero(valid):
        if i==0 or not valid[i-1] or not np.isfinite(gap) or not (0<t[i]-t[i-1]<=gap): sid+=1
        ids[i]=sid
    return ids

def warp(x,y,h):
    """Same unit-H geometry as upstream; transform observed coordinates only."""
    h=np.asarray(h,dtype=float)
    if not np.isfinite(h).all() or (h<=0).any(): raise ValueError('Invalid arm lengths')
    h1,h2,h3,h4,h5,h6=h
    xn=np.where(x<=0,x/h1,x/h4)
    frac=np.minimum(np.abs(xn),1.)
    top=7.+frac*(np.where(xn<=0,h2,h5)-7.)
    bottom=(h3+h6)/2+frac*(np.where(xn<=0,h3,h6)-(h3+h6)/2)
    return np.column_stack([xn,np.where(y>=0,y/top,y/bottom)])

def quality(xraw,yraw,praw,cfg):
    t=xraw[:,1]*1000.; x=xraw[:,0].copy()
    yi,yr=match(t,yraw[:,1]*1000.,cfg.match_tolerance_ms)
    pi,pr=match(t,praw[:,1]*1000.,cfg.match_tolerance_ms)
    y=np.full(len(t),np.nan); pupil=np.full(len(t),np.nan)
    y[yi>=0]=yraw[yi[yi>=0],0]; pupil[pi>=0]=praw[pi[pi>=0],0]
    flag=np.zeros(len(t),dtype=np.uint16)
    def mark(mask,name): flag[mask]|=FLAGS[name]
    mark(~np.isfinite(t),'bad_timestamp');mark(~np.isfinite(x),'nonfinite_x')
    mark(yi<0,'unmatched_y');mark(~np.isfinite(y),'nonfinite_y')
    mark((np.abs(x)>cfg.position_limit_deg)|(np.abs(y)>cfg.position_limit_deg),'outside_30deg')
    mark(pi<0,'unmatched_pupil');mark(~np.isfinite(pupil)|(pupil<=0),'invalid_pupil')
    bad=np.flatnonzero(~np.isfinite(np.diff(t))|(np.diff(t)<=0))
    flag[bad]|=FLAGS['nonincreasing_timestamp'];flag[bad+1]|=FLAGS['nonincreasing_timestamp']
    threshold=gap_threshold(t,cfg)
    pt=praw[:,1]*1000.; pv=praw[:,0]; pseg=segments(pt,np.isfinite(pt),gap_threshold(pt,cfg))
    blinks=[]
    for sid in np.unique(pseg[pseg>=0]):
        ii=np.flatnonzero(pseg==sid); pp=pv[ii]; tt=pt[ii]
        loss=~np.isfinite(pp)|(pp<=0)
        d=np.abs(np.diff(pp)); good=np.isfinite(d)&~loss[:-1]&~loss[1:]
        cutoff=5*np.quantile(d[good],.95) if good.any() else np.inf
        rapid=np.zeros(len(pp),bool)
        if cutoff>0 and np.isfinite(cutoff): rapid[1:]=good&(d>cutoff)
        candidate=loss|rapid
        rr=runs(candidate)
        merged=[]
        for r in rr:
            if merged and tt[r[0]]-tt[merged[-1][-1]]<=cfg.blink_merge_ms:
                merged[-1]=np.arange(merged[-1][0],r[-1]+1)
            else: merged.append(r)
        for r in merged:
            if tt[r[-1]]-tt[r[0]]>=cfg.blink_min_ms:
                start,end=float(tt[r[0]]),float(tt[r[-1]])
                blinks.append((start,end))
                mark((t>=start-cfg.blink_padding_ms)&(t<=end+cfg.blink_padding_ms),'blink_padding')
    valid=flag==0
    return dict(t=t,x=x,y=y,pupil=pupil,y_index=yi,pupil_index=pi,y_residual_ms=yr,pupil_residual_ms=pr,
                flags=flag,valid=valid,gap_ms=threshold,segment=segments(t,valid,threshold),blinks=blinks)

def support_weights(t):
    if len(t)<2:return np.zeros(len(t))
    d=np.diff(t); w=np.empty(len(t)); w[0]=d[0]/2;w[-1]=d[-1]/2
    if len(t)>2:w[1:-1]=(d[:-1]+d[1:])/2
    return w

def velocity(t,xy,cfg):
    """Local time-weighted linear slope using observed points only. No gap crossing."""
    n=len(t); out=np.full((n,2),np.nan)
    if n<3:return out
    z=(t-t[0])/1000.; w=support_weights(t)
    left=np.searchsorted(t,t-cfg.velocity_half_window_ms);right=np.searchsorted(t,t+cfg.velocity_half_window_ms,side='right')
    cols=np.column_stack([w,w*z,w*z*z,w*xy[:,0],w*xy[:,1],w*z*xy[:,0],w*z*xy[:,1]])
    cs=np.vstack([np.zeros(7),np.cumsum(cols,axis=0)]); ss=cs[right]-cs[left]
    sw,st,stt,sx,sy,stx,sty=ss.T; denom=sw*stt-st*st
    ok=(right-left>=3)&(t[np.maximum(right-1,0)]-t[left]>=cfg.velocity_min_span_ms)&(denom>1e-18)
    out[ok,0]=(sw[ok]*stx[ok]-st[ok]*sx[ok])/denom[ok]
    out[ok,1]=(sw[ok]*sty[ok]-st[ok]*sy[ok])/denom[ok]
    return out

def idt(t,xy,cfg):
    """Greedy I-DT, elapsed-time minimum duration, on one valid continuous segment."""
    result=[]; i=0;n=len(t)
    while i<n:
        j=int(np.searchsorted(t,t[i]+cfg.fixation_min_ms,side='left'))
        if j>=n:break
        block=xy[i:j+1];lo=block.min(axis=0);hi=block.max(axis=0)
        if (hi-lo).sum()>cfg.fixation_dispersion_deg:i+=1;continue
        while j+1<n:
            nl=np.minimum(lo,xy[j+1]);nh=np.maximum(hi,xy[j+1])
            if (nh-nl).sum()>cfg.fixation_dispersion_deg:break
            j+=1;lo,hi=nl,nh
        result.append((i,j));i=j+1
    return result

def detect(q,epoch_end,h,cfg):
    inwin=np.isfinite(q['t'])&(q['t']>=0)&(q['t']<=epoch_end)
    seg=segments(q['t'],q['valid']&inwin,q['gap_ms'])
    vel=np.full((len(seg),2),np.nan); xy=np.column_stack([q['x'],q['y']]); normalized=warp(q['x'],q['y'],h)
    for sid in np.unique(seg[seg>=0]):
        ii=np.flatnonzero(seg==sid); vel[ii]=velocity(q['t'][ii],xy[ii],cfg)
    finite=np.isfinite(vel).all(axis=1); vv=vel[finite]
    if len(vv)>=10:
        scale=np.sqrt(np.median((vv-np.median(vv,axis=0))**2,axis=0))
        threshold=cfg.saccade_factor*scale
        status='ok' if np.isfinite(threshold).all() and (threshold>1e-10).all() else 'degenerate_velocity_threshold'
    else:threshold=np.array([np.nan,np.nan]);status='insufficient_velocity_samples'
    sac=np.zeros(len(seg),bool);events=[]
    def event(kind,ii,sid):
        t=q['t'][ii]; w=support_weights(t); center=np.average(normalized[ii],axis=0,weights=w) if w.sum()>0 else normalized[ii[0]]
        dist=np.linalg.norm(ROI-center,axis=1); roi=int(np.argmin(dist)) if dist.min()<=cfg.roi_radius else -1
        members=np.flatnonzero(seg==sid)
        events.append(dict(kind=kind,segment=int(sid),start_index=int(ii[0]),end_index=int(ii[-1]),
          onset_ms=float(t[0]),offset_ms=float(t[-1]),duration_ms=float(t[-1]-t[0]),
          touches_segment_start=bool(ii[0]==members[0]),touches_segment_end=bool(ii[-1]==members[-1]),
          centroid_unit_x=float(center[0]),centroid_unit_y=float(center[1]),roi=ROI_NAMES[roi] if roi>=0 else 'unassigned',
          roi_code=roi,peak_speed_deg_s=float(np.max(np.linalg.norm(vel[ii][np.isfinite(vel[ii]).all(axis=1)],axis=1))) if np.isfinite(vel[ii]).all(axis=1).any() else np.nan))
    for sid in np.unique(seg[seg>=0]):
        ii=np.flatnonzero(seg==sid); t=q['t'][ii]
        if status=='ok':
            score=np.sum((vel[ii]/threshold)**2,axis=1)
            hit=np.isfinite(vel[ii]).all(axis=1)&(score>1)
            rr=[r for r in runs(hit) if t[r[-1]]-t[r[0]]>=cfg.saccade_min_ms]
            merged=[]
            for r in rr:
                if merged and t[r[0]]-t[merged[-1][-1]]<=cfg.saccade_merge_ms and np.isfinite(vel[ii[merged[-1][-1]:r[0]+1]]).all():
                    merged[-1]=np.arange(merged[-1][0],r[-1]+1)
                else:merged.append(r)
            for r in merged:sac[ii[r]]=True;event('saccade',ii[r],sid)
        for start,end in idt(t,xy[ii],cfg):
            # Remove saccade overlap and re-check minimum duration on each fragment.
            local=np.arange(start,end+1)
            for r in runs(~sac[ii[local]]):
                jj=ii[local[r]]
                if q['t'][jj[-1]]-q['t'][jj[0]]>=cfg.fixation_min_ms:
                    event('fixation' if status=='ok' else 'fixation_unconfirmed',jj,sid)
    return dict(segment=seg,velocity=vel,normalized=normalized,events=events,status=status,threshold=threshold,in_epoch=inwin)
