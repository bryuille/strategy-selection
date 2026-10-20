"""QC plots and movies from recorded samples; no trajectory interpolation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from core import ROI,ROI_NAMES,warp
COLORS=['#7b5ab3','#267caf','#21a386','#d48624','#ca4e61']

def maze(ax,h=None,cfg=None):
    if h is None:
        left,right=-1.,1.;upL,downL,upR,downR,stem=1.,-1.,1.,-1.,1.
    else:
        h1,h2,h3,h4,h5,h6=h;left,right=-h1,h4;upL,downL,upR,downR,stem=h2,-h3,h5,-h6,7.
    for xx,yy in [([left,right],[0,0]),([left,left],[downL,upL]),([right,right],[downR,upR]),([0,0],[0,stem])]:
        ax.plot(xx,yy,color='#253541',lw=2,zorder=2)
    ax.scatter([0],[stem],marker='+',color='#253541',s=75,zorder=3)
    ax.set_aspect('equal');ax.grid(alpha=.13)
    if h is None:
        for (x,y),name,c in zip(ROI,ROI_NAMES,COLORS):
            ax.add_patch(Circle((x,y),cfg.roi_radius,fc=c,ec=c,alpha=.09,zorder=0))
            ax.text(x,y+.1,name,color=c,fontsize=9,ha='center',weight='bold')
        ax.set_xlim(-2.3,2.3);ax.set_ylim(-2.3,2.3);ax.set_xlabel('Unit-H x');ax.set_ylabel('Unit-H y')
    else:
        lim=max(abs(left),right,abs(downL),upL,abs(downR),upR,7)+4
        ax.set_xlim(-lim,lim);ax.set_ylim(-lim,lim);ax.set_xlabel('Horizontal gaze (degrees)');ax.set_ylabel('Vertical gaze (degrees)')

def session_map(m,s,points,cfg,path):
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    arr=np.asarray(points,float).reshape(-1,5)
    for number,ax in enumerate(axs.flat,1):
        maze(ax,cfg=cfg);z=arr[arr[:,0]==number]
        for roi in range(-1,5):
            v=z[z[:,3]==roi]
            if len(v):ax.scatter(v[:,1],v[:,2],s=np.clip(v[:,4]/15,5,40),alpha=.4,c='gray' if roi<0 else COLORS[roi],edgecolors='none')
        ax.set_title(f'Maze {number}: {len(z)} fixation centroids',fontsize=11)
    fig.suptitle(f'{m} / {s} — normalized H-maze and five ROIs\nObserved fixation centroids; marker size follows duration; gray = unassigned',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(path,dpi=150);plt.close(fig)

def trial_plot(pack,cfg,path):
    q,d,key=pack['q'],pack['d'],pack['key']; end=pack['event_times']['fix_start']; w=d['in_epoch']
    fig=plt.figure(figsize=(13,8));gs=fig.add_gridspec(3,2,width_ratios=[1.7,1])
    axes=[fig.add_subplot(gs[k,0]) for k in range(3)];am=fig.add_subplot(gs[:,1]);maze(am,cfg=cfg)
    for ax,field,label in zip(axes[:2],['x','y'],['X (deg)','Y (deg)']):
        ax.scatter(q['t'][w],q[field][w],s=2,c='#b8bec2')
        for sid in np.unique(d['segment'][d['segment']>=0]):
            ii=np.flatnonzero(d['segment']==sid);ax.plot(q['t'][ii],q[field][ii],lw=.7,color='#267caf')
        ax.set_ylabel(label)
    speed=np.linalg.norm(d['velocity'],axis=1)
    for sid in np.unique(d['segment'][d['segment']>=0]):
        ii=np.flatnonzero(d['segment']==sid);axes[2].plot(q['t'][ii],speed[ii],lw=.7,color='#267caf')
        am.plot(d['normalized'][ii,0],d['normalized'][ii,1],lw=.6,alpha=.5,color='#267caf')
    for event in d['events']:
        if event['kind']=='fixation':
            c=COLORS[event['roi_code']] if event['roi_code']>=0 else 'gray'
            am.scatter(event['centroid_unit_x'],event['centroid_unit_y'],s=50,color=c,zorder=5)
        if event['kind']=='saccade':
            for ax in axes:ax.axvspan(event['onset_ms'],event['offset_ms'],color='#ca4e61',alpha=.13)
    tt=q['t']; gaps=np.flatnonzero(np.diff(tt)>q['gap_ms'])
    for i in gaps:
        if tt[i]<end and tt[i+1]>0:
            for ax in axes:ax.axvspan(max(0,tt[i]),min(end,tt[i+1]),color='#e5a84b',alpha=.3)
    for ax in axes:ax.set_xlim(0,end);ax.grid(alpha=.15)
    axes[2].set_ylabel('Speed (deg/s)');axes[2].set_xlabel('Time from maze onset (ms)')
    am.set_title('Normalized observed trajectories\nLines stop at every invalid segment')
    fig.suptitle(f'{key["monkey"]} / {key["session"]} / trial {key["trial_id"]}\nBlue = usable; gray = recorded excluded; orange = time gap; red = detected saccade',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(path,dpi=150);plt.close(fig)

def movie(pack,cfg,path):
    import imageio_ffmpeg
    q,d,key=pack['q'],pack['d'],pack['key'];et=pack['event_times'];t=q['t']
    finite=np.isfinite(t);end=min(float(np.max(t[finite])),et['flash_three']+300.)
    frame_ms=20.;fps=50;times=np.arange(0,end+1e-8,frame_ms)
    fig,(a,b)=plt.subplots(1,2,figsize=(10.8,5.6),dpi=100);maze(a,pack['h'],cfg);maze(b,cfg=cfg)
    a.set_title('Recorded gaze in degrees');b.set_title('Same recorded gaze in unit H')
    raw_dot,=a.plot([],[],'o',color='#c54343',ms=7);norm_dot,=b.plot([],[],'o',color='#c54343',ms=7)
    raw_trail,=a.plot([],[],color='#c54343',lw=1);norm_trail,=b.plot([],[],color='#c54343',lw=1)
    status=fig.text(.5,.04,'',ha='center',fontsize=10)
    title=fig.suptitle('',fontsize=13);fig.tight_layout(rect=(0,.12,1,.9))
    # Bound each segment: current markers cannot cross a no-observation interval.
    bounds=[]
    for sid in np.unique(q['segment'][q['segment']>=0]):
        ii=np.flatnonzero(q['segment']==sid)
        bounds.append((t[ii[0]],t[ii[-1]],int(sid),ii))
    normalized=warp(q['x'],q['y'],pack['h'])
    fig.canvas.draw();width,height=fig.canvas.get_width_height()
    gen=imageio_ffmpeg.write_frames(str(path),(width,height),fps=fps,codec='libx264',pix_fmt_in='rgb24',
        pix_fmt_out='yuv420p',quality=7,macro_block_size=2,ffmpeg_log_level='error')
    gen.send(None)
    for now in times:
        available=next((r for r in bounds if r[0]<=now<=r[1]),None)
        text='NO USABLE OBSERVATION — marker hidden'
        raw_dot.set_data([],[]);norm_dot.set_data([],[]);raw_trail.set_data([],[]);norm_trail.set_data([],[])
        if available is not None:
            ii=available[3];j=int(ii[np.argmin(abs(t[ii]-now))])
            if abs(t[j]-now)<=q['gap_ms']/2:
                raw_dot.set_data([q['x'][j]],[q['y'][j]]);norm_dot.set_data([normalized[j,0]],[normalized[j,1]])
                trail=ii[(t[ii]>=now-150)&(t[ii]<=now)]
                raw_trail.set_data(q['x'][trail],q['y'][trail]);norm_trail.set_data(normalized[trail,0],normalized[trail,1])
                text=f'Observed sample {j} at {t[j]:.3f} ms | frame residual {t[j]-now:+.3f} ms'
        phase='sampling epoch' if now<=et['fix_start'] else 'outside preprocessing epoch'
        flashes=[name for name in ['flash_one','flash_two','flash_three'] if 0<=now-et[name]<100]
        if flashes:text+=' | '+', '.join(flashes)+' (100 ms display annotation)'
        status.set_text(text+'\nNo interpolated gaze; trails never cross invalid segments. Playback 1×.')
        title.set_text(f'{key["monkey"]} / {key["session"]} / trial {key["trial_id"]} / Maze {pack["maze"]}\n{now:.0f} ms from maze onset — {phase}')
        fig.canvas.draw();rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy();gen.send(rgb)
    gen.close();plt.close(fig)
    return dict(trial_id=key['trial_id'],frames=len(times),fps=fps,frame_step_ms=frame_ms,
                playback_speed=1,source_start_ms=0,source_end_ms=end,event_times_ms=et,
                selection='largest observed fraction for rank 1; interrupted trial with largest observed duration for rank 2',
                marker='nearest original usable sample, only within a continuous segment and within half the adaptive gap threshold',
                no_gaze_interpolation=True)
