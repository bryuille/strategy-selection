"""Current 75-ms stable looks and raw center/side ROI assignment.

quality/detect preserve timestamp-native saccades; legacy I-DT events are not
used to define current looks. Missing data and gaps remain boundaries.
"""
import numpy as np
try:
    from .core import Config, quality, detect, warp, support_weights
    from .stable_support import detect_looks
except ImportError:
    from core import Config, quality, detect, warp, support_weights
    from stable_support import detect_looks


def raw_roi(x, y, h):
    def distance(a, b):
        a, b = np.asarray(a), np.asarray(b)
        v = b - a
        u = np.clip(np.dot(np.array([x, y]) - a, v) / np.dot(v, v), 0, 1)
        return np.linalg.norm(np.array([x, y]) - (a + u * v))
    if np.hypot(x, y) <= 2.5:
        return 'center'
    a, b, c, d, e, f = h
    if x < 0 and min(distance((0, 0), (-a, 0)), distance((-a, -c), (-a, b))) <= 2.5:
        return 'left'
    if x > 0 and min(distance((0, 0), (d, 0)), distance((d, -f), (d, e))) <= 2.5:
        return 'right'
    return 'unassigned'


def qualified_looks(q, h, epoch_end_ms, saccades):
    """Return current raw-ROI looks from observed samples and detected saccades."""
    h = np.asarray(h, dtype=float)
    if h.shape != (6,) or not np.isfinite(h).all() or (h <= 0).any():
        raise ValueError('Geometry must contain six finite positive arm lengths')
    if not np.isfinite(epoch_end_ms) or epoch_end_ms <= 0:
        raise ValueError('Epoch end must be finite and positive (milliseconds)')
    t = q['t']; seg = q['segment']
    looks = detect_looks(q, warp(q['x'], q['y'], h), epoch_end_ms, saccades)
    output = []
    for e in looks:
        w = np.zeros(len(t)); island_labels = []
        for a, b in e['stable_intervals']:
            ix = np.flatnonzero((t >= a) & (t <= b) & (seg == e['segment']))
            assert len(ix) >= 2 and np.all(np.diff(t[ix]) > 0) and q['valid'][ix].all()
            iw = support_weights(t[ix]); w[ix] += iw
            island_labels.append(raw_roi(np.average(q['x'][ix], weights=iw),
                                         np.average(q['y'][ix], weights=iw), h))
        ix = w > 0
        assert np.isfinite(q['x'][ix]).all() and np.isfinite(q['y'][ix]).all()
        x = float(np.average(q['x'][ix], weights=w[ix])); y = float(np.average(q['y'][ix], weights=w[ix]))
        assert abs(w.sum() - e['duration_ms']) < 1e-7
        xy = warp(np.array([x]), np.array([y]), h)[0]
        output.append(dict(kind='qualified_look', raw_x=x, raw_y=y,
                           normalized_x=float(xy[0]), normalized_y=float(xy[1]),
                           roi=raw_roi(x, y, h), stable_duration_ms=e['duration_ms'],
                           onset_ms=e['onset_ms'], offset_ms=e['offset_ms'],
                           segment=e['segment'], stable_intervals=e['stable_intervals'],
                           island_roi_disagreement=len(set(island_labels)) > 1,
                           candidate_start_ms=e['candidate_start_ms'], candidate_end_ms=e['candidate_end_ms'],
                           witness_diameter_deg=e['witness_diameter_deg']))
    return output


def preprocess_trial(xraw, yraw, pupilraw, h, epoch_end_ms, cfg=None):
    """Arrays are N×2 [value, seconds relative to maze onset]; epoch in ms.

    Returns observed-sample QC, saccades and current qualified looks. Does not
    read neural labels or create/interpolate any eye observation.
    """
    cfg = Config() if cfg is None else cfg
    for a in [xraw, yraw, pupilraw]:
        if np.asarray(a).ndim != 2 or np.asarray(a).shape[1] != 2:
            raise ValueError('Expected N×2 [value, seconds] eye arrays')
    # Validate geometry/epoch before the legacy saccade engine uses them.
    h = np.asarray(h, dtype=float)
    if h.shape != (6,) or not np.isfinite(h).all() or (h <= 0).any() or not np.isfinite(epoch_end_ms) or epoch_end_ms <= 0:
        raise ValueError('Invalid geometry or epoch')
    q = quality(np.asarray(xraw), np.asarray(yraw), np.asarray(pupilraw), cfg)
    d = detect(q, epoch_end_ms, h, cfg)
    saccades = [e for e in d['events'] if e['kind'] == 'saccade']
    epoch_q = dict(q, segment=d['segment'], valid=d['segment'] >= 0)
    looks = qualified_looks(epoch_q, h, epoch_end_ms, saccades)
    for e in looks:
        e['saccade_status'] = d['status']
    return dict(quality=q, epoch_segment=d['segment'], velocity=d['velocity'],
                saccade_status=d['status'], saccade_threshold=d['threshold'],
                saccades=saccades, looks=looks)
