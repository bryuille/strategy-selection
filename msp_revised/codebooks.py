"""Codebooks: rules that assign each fixation centroid to one named state.

A codebook is a function of the fixation table (`features.load`): per
fixation it sees the centroid in unit H (``xy_u``), in degrees (``xy_d``),
and the trial's arm lengths ``h`` (so degree-frame landmarks can be placed
per maze). It returns a state index in ``0..k-1`` or -1 for "unassigned".

Two coordinate frames, on purpose. Unit H puts the landmarks at fixed
coordinates but stretches x on the right side by ``10 / h4``: up to 2x
between mazes 1-2 and 5-6, where the H/S mix also flips. Degree-frame
codebooks place the same landmarks per maze from ``h`` and so cannot
inherit that stretch. A result that holds in both frames is not a
scaling artifact.

Families (``key`` → states):

balls_u<r>      unit H: origin ball + four exit balls, radius r          (5)
balls6_u<r>     ... plus the point (0, 1) above the origin ("top")        (6)
balls_d<r>      degrees: origin + four exits, radius r deg                (5)
balls6_d<r>     ... plus (0, STEM_DEG) deg ("top")                        (6)
quads_u<r>      origin ball + maze quadrants inside the rounded hull      (5)
quads_out_u<r>  ... plus an explicit "outside the maze" state             (6)
halves_u<r>     origin ball + left / right inside the hull                (3)
halves_out_u<r> ... plus outside                                          (4)
segments_d      origin ball (2 deg) + nearest maze segment in degrees within
                SEG_TOL_DEG: crossbar-L, crossbar-R, bar-LU, bar-LD, bar-RU,
                bar-RD (the H: crossbar through the fixation point)         (7)
segments4_d     the same merged: origin, crossbar, left bar, right bar    (4)
sectors<k>_d    origin ball (2 deg) + k angular sectors from the fixation
                point, degree frame (angles are not warp-invariant)       (k+1)
rings_d         eccentricity from the fixation point in degrees:
                < 2, 2-6, 6-10, > 10                                      (4)
rings_side_d    origin (< 2 deg) + three rings x left / right             (7)
inout_u<r>      origin ball / inside the hull / outside                   (3)
ybands_d        origin (2 deg) + vertical bands in degrees: below (y < -2),
                crossbar level (-2..5), upper arms (5..9), above (> 9)    (5)
capsules_u<r>   origin ball + four straight capsules origin -> exit       (5)
capsules4_u<w>  four capsules origin -> exit with NO origin ball: the centre
                is split between the four directions (nearest capsule axis,
                half width w unit H)                                      (4)
capsules4_d<w>  the same in degrees (half width w deg)                    (4)
oquads_u<r>     the origin ball (r unit H) split into its four quadrants,
                plus one "outside the ball" state                         (5)
oquads_d<r>     the same in degrees (r deg)                               (5)
ohalves_u<r>    origin ball split left / right, plus outside              (3)
ohalves_d<r>    the same in degrees                                       (3)
oquads_balls_u<r> origin quadrants + the four exit balls (r unit H)       (8)
oquads_balls_d<r> the same in degrees                                     (8)

`REGISTRY` maps every key to its `Codebook`; `SEARCH_KEYS` is the set the
search runs. Two states would make the Pearson correlation of two group
means always +-1, so every codebook here has >= 3 states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from msp_revised.config import STEM_DEG

# ---- shared geometry ---------------------------------------------------------

UNIT_EXITS = {"LU": (-1.0, 1.0), "LD": (-1.0, -1.0), "RU": (1.0, 1.0), "RD": (1.0, -1.0)}
EXIT_HALF_WIDTH = 1.0  # exits at (+-1, +-1)
PERIMETER_RADIUS = 1.0  # hull = r1 exit balls joined by tangents (msp_trends)
SEG_TOL_DEG = 2.5
ORIGIN_DEG = 2.0
RING_EDGES_DEG = (2.0, 6.0, 10.0)
YBAND_EDGES_DEG = (-2.0, 5.0, 9.0)
CAPSULE_HALF_WIDTH = 0.35  # unit H


def deg_landmarks(h):
    """``(6, 2)`` degree-frame landmark positions per fixation from ``h`` (n, 6):
    origin, LU, LD, RU, RD, top (0, STEM_DEG)."""
    h = np.asarray(h, dtype=float)
    h1, h2, h3, h4, h5, h6 = (h[:, k] for k in range(6))
    z = np.zeros_like(h1)
    return np.stack(
        [
            np.stack([z, z], 1),
            np.stack([-h1, h2], 1),
            np.stack([-h1, -h3], 1),
            np.stack([h4, h5], 1),
            np.stack([h4, -h6], 1),
            np.stack([z, np.full_like(z, STEM_DEG)], 1),
        ],
        axis=1,
    )


UNIT_LANDMARKS = np.array(
    [[0.0, 0.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, 1.0], [1.0, -1.0], [0.0, 1.0]]
)
LANDMARK_NAMES = ("origin", "LU", "LD", "RU", "RD", "top")


def nearest_within(xy, centres, radius):
    """Nearest centre within `radius`, else -1. `centres` is ``(k, 2)`` or
    ``(n, k, 2)`` (per-fixation centres)."""
    xy = np.asarray(xy, dtype=float)
    centres = np.asarray(centres, dtype=float)
    if xy.shape[0] == 0:
        return np.empty(0, dtype=np.int16)
    if centres.ndim == 2:
        centres = np.broadcast_to(centres[None], (xy.shape[0],) + centres.shape)
    dist = np.linalg.norm(xy[:, None, :] - centres, axis=-1)
    eligible = dist <= float(radius)
    state = np.full(xy.shape[0], -1, dtype=np.int16)
    has = eligible.any(axis=1)
    pick = np.where(eligible, dist, np.inf).argmin(axis=1)
    state[has] = pick[has]
    return state


def inside_hull(xy_u):
    """Within `PERIMETER_RADIUS` of the square the exits span (unit H)."""
    xy = np.asarray(xy_u, dtype=float)
    excess = np.clip(np.abs(xy) - EXIT_HALF_WIDTH, 0.0, None)
    return np.linalg.norm(excess, axis=-1) <= PERIMETER_RADIUS


def point_segment_distance(p, a, b):
    """Distance from points `p` (n, 2) to segments a->b (n, 2 each)."""
    ab = b - a
    ap = p - a
    denom = (ab**2).sum(axis=1)
    t = np.divide((ap * ab).sum(axis=1), denom, out=np.zeros(len(p)), where=denom > 0)
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return np.linalg.norm(p - proj, axis=1)


# ---- the codebook object -----------------------------------------------------


@dataclass(frozen=True)
class Codebook:
    key: str
    names: tuple
    frame: str  # "unit" or "deg"
    label: str
    _assign: callable
    radius: float | None = None  # drawn in the codebook figure when set

    @property
    def k(self):
        return len(self.names)

    def assign(self, fix):
        """State per fixation; `fix` has ``xy_u``, ``xy_d``, ``h`` (per fixation)."""
        out = np.asarray(self._assign(fix), dtype=np.int16)
        assert out.shape == (len(fix["xy_u"]),)
        assert out.max(initial=-1) < self.k
        return out


def _balls_u(r, junction):
    k = 6 if junction else 5
    return lambda fix: nearest_within(fix["xy_u"], UNIT_LANDMARKS[:k], r)


def _balls_d(r, junction):
    k = 6 if junction else 5
    return lambda fix: nearest_within(fix["xy_d"], deg_landmarks(fix["h"])[:, :k], r)


def _quadrants(xy):
    right, up = xy[:, 0] >= 0, xy[:, 1] >= 0
    # order: LU, LD, RU, RD  -> indices 1..4 (origin is 0)
    return np.where(right, np.where(up, 3, 4), np.where(up, 1, 2))


def _quads_u(r, outside, halves):
    def f(fix):
        xy = np.asarray(fix["xy_u"], dtype=float)
        state = np.full(len(xy), -1, dtype=np.int16)
        if not len(xy):
            return state
        origin = np.linalg.norm(xy, axis=1) <= r
        inside = inside_hull(xy) & ~origin
        if halves:
            region = np.where(xy[:, 0] >= 0, 2, 1)
            k_in = 3
        else:
            region = _quadrants(xy)
            k_in = 5
        state[inside] = region[inside]
        state[origin] = 0
        if outside:
            state[~inside & ~origin] = k_in
        return state

    return f


def _segments(merged):
    def f(fix):
        p = np.asarray(fix["xy_d"], dtype=float)
        n = len(p)
        if not n:
            return np.full(0, -1, dtype=np.int16)
        lm = deg_landmarks(fix["h"])  # origin, LU, LD, RU, RD, top
        o, lu, ld, ru, rd = (lm[:, i] for i in range(5))
        cl = np.stack([lu[:, 0], np.zeros(n)], 1)  # left end of the crossbar
        cr = np.stack([ru[:, 0], np.zeros(n)], 1)  # right end of the crossbar
        segs = [
            (cl, o),  # crossbar-L
            (o, cr),  # crossbar-R
            (cl, lu),  # bar-LU
            (cl, ld),  # bar-LD
            (cr, ru),  # bar-RU
            (cr, rd),  # bar-RD
        ]
        dist = np.stack([point_segment_distance(p, a, b) for a, b in segs], 1)
        best = dist.argmin(axis=1)
        state = np.where(dist.min(axis=1) <= SEG_TOL_DEG, 1 + best, -1).astype(np.int16)
        if merged:
            merge = np.array([0, 1, 1, 2, 2, 3, 3], dtype=np.int16)
            ok = state >= 0
            state[ok] = merge[state[ok]]
        state[np.linalg.norm(p, axis=1) <= ORIGIN_DEG] = 0
        return state

    return f


def _sectors(k):
    def f(fix):
        p = np.asarray(fix["xy_d"], dtype=float)
        state = np.full(len(p), -1, dtype=np.int16)
        if not len(p):
            return state
        rad = np.linalg.norm(p, axis=1)
        ang = np.arctan2(p[:, 1], p[:, 0])  # 0 = right, pi/2 = up
        # sector 0 is centred on "right"; sectors go anticlockwise
        width = 2 * np.pi / k
        sec = np.floor(((ang + width / 2) % (2 * np.pi)) / width).astype(int)
        state[:] = 1 + sec
        state[rad <= ORIGIN_DEG] = 0
        return state

    return f


def _rings(side):
    def f(fix):
        p = np.asarray(fix["xy_d"], dtype=float)
        state = np.full(len(p), -1, dtype=np.int16)
        if not len(p):
            return state
        rad = np.linalg.norm(p, axis=1)
        ring = np.searchsorted(np.asarray(RING_EDGES_DEG), rad, side="right")  # 0..3
        if not side:
            return ring.astype(np.int16)
        right = p[:, 0] >= 0
        out = np.where(ring == 0, 0, 1 + (ring - 1) * 2 + right.astype(int))
        return out.astype(np.int16)

    return f


def _inout_u(r):
    def f(fix):
        xy = np.asarray(fix["xy_u"], dtype=float)
        state = np.full(len(xy), -1, dtype=np.int16)
        if not len(xy):
            return state
        origin = np.linalg.norm(xy, axis=1) <= r
        inside = inside_hull(xy) & ~origin
        state[:] = 2
        state[inside] = 1
        state[origin] = 0
        return state

    return f


def _ybands(fix):
    p = np.asarray(fix["xy_d"], dtype=float)
    state = np.full(len(p), -1, dtype=np.int16)
    if not len(p):
        return state
    band = np.searchsorted(np.asarray(YBAND_EDGES_DEG), p[:, 1], side="right")  # 0..3
    state[:] = 1 + band
    state[np.linalg.norm(p, axis=1) <= ORIGIN_DEG] = 0
    return state


def _capsules_u(r):
    def f(fix):
        p = np.asarray(fix["xy_u"], dtype=float)
        n = len(p)
        state = np.full(n, -1, dtype=np.int16)
        if not n:
            return state
        o = np.zeros((n, 2))
        dist = np.stack(
            [point_segment_distance(p, o, np.broadcast_to(np.asarray(e), (n, 2)))
             for e in UNIT_EXITS.values()], 1
        )
        best = dist.argmin(axis=1)
        state = np.where(dist.min(axis=1) <= CAPSULE_HALF_WIDTH, 1 + best, -1).astype(np.int16)
        state[np.linalg.norm(p, axis=1) <= r] = 0
        return state

    return f


def _capsules4(frame, w):
    """Nearest origin->exit axis within half width `w`; no origin ball."""

    def f(fix):
        p = np.asarray(fix["xy_u" if frame == "unit" else "xy_d"], dtype=float)
        n = len(p)
        if not n:
            return np.full(0, -1, dtype=np.int16)
        o = np.zeros((n, 2))
        if frame == "unit":
            exits = [np.broadcast_to(np.asarray(e), (n, 2)) for e in UNIT_EXITS.values()]
        else:
            lm = deg_landmarks(fix["h"])
            exits = [lm[:, i] for i in (1, 2, 3, 4)]
        dist = np.stack([point_segment_distance(p, o, e) for e in exits], 1)
        best = dist.argmin(axis=1)
        return np.where(dist.min(axis=1) <= w, best, -1).astype(np.int16)

    return f


def _origin_split(frame, r, halves, exits):
    """Quadrants (or halves) of the origin ball; `exits` adds the exit balls,
    otherwise everything outside the ball is one state."""

    def f(fix):
        p = np.asarray(fix["xy_u" if frame == "unit" else "xy_d"], dtype=float)
        n = len(p)
        state = np.full(n, -1, dtype=np.int16)
        if not n:
            return state
        inside = np.linalg.norm(p, axis=1) <= r
        if halves:
            part = np.where(p[:, 0] >= 0, 1, 0)
            k_in = 2
        else:
            part = _quadrants(p) - 1  # 0..3 in LU, LD, RU, RD order
            k_in = 4
        state[inside] = part[inside]
        if exits:
            if frame == "unit":
                centres = UNIT_LANDMARKS[1:5]
            else:
                centres = deg_landmarks(fix["h"])[:, 1:5]
            ball = nearest_within(p, centres, r)
            hit = ~inside & (ball >= 0)
            state[hit] = k_in + ball[hit]
        else:
            state[~inside] = k_in
        return state

    return f


EXITS = ("LU", "LD", "RU", "RD")
QUAD_NAMES = ("origin",) + EXITS


def _build_registry():
    reg = {}

    def add(cb):
        assert cb.key not in reg and cb.k >= 3, cb.key
        reg[cb.key] = cb

    for r in (0.5, 0.75, 1.0):
        add(Codebook(f"balls_u{r:g}", QUAD_NAMES, "unit",
                     f"origin + four exit balls, r = {r:g} unit H", _balls_u(r, False), r))
        add(Codebook(f"balls6_u{r:g}", QUAD_NAMES + ("top",), "unit",
                     f"origin + four exits + top (0, 1), r = {r:g} unit H", _balls_u(r, True), r))
    for r in (2.0, 3.0):
        add(Codebook(f"balls_d{r:g}", QUAD_NAMES, "deg",
                     f"origin + four exit balls, r = {r:g} deg", _balls_d(r, False), r))
        add(Codebook(f"balls6_d{r:g}", QUAD_NAMES + ("top",), "deg",
                     f"origin + four exits + top (0, {STEM_DEG:g}), r = {r:g} deg", _balls_d(r, True), r))
    for r in (0.5,):
        add(Codebook(f"quads_u{r:g}", QUAD_NAMES, "unit",
                     f"origin ball (r = {r:g}) + maze quadrants", _quads_u(r, False, False), r))
        add(Codebook(f"quads_out_u{r:g}", QUAD_NAMES + ("outside",), "unit",
                     f"origin ball (r = {r:g}) + quadrants + outside", _quads_u(r, True, False), r))
        add(Codebook(f"halves_u{r:g}", ("origin", "left", "right"), "unit",
                     f"origin ball (r = {r:g}) + maze halves", _quads_u(r, False, True), r))
        add(Codebook(f"halves_out_u{r:g}", ("origin", "left", "right", "outside"), "unit",
                     f"origin ball (r = {r:g}) + halves + outside", _quads_u(r, True, True), r))
        add(Codebook(f"inout_u{r:g}", ("origin", "inside", "outside"), "unit",
                     f"origin ball (r = {r:g}) / inside the maze / outside", _inout_u(r), r))
        add(Codebook(f"capsules_u{r:g}", QUAD_NAMES, "unit",
                     f"origin ball (r = {r:g}) + capsules origin → exit", _capsules_u(r), r))
    add(Codebook("segments_d",
                 ("origin", "cross-L", "cross-R", "bar-LU", "bar-LD", "bar-RU", "bar-RD"), "deg",
                 f"origin ({ORIGIN_DEG:g} deg) + nearest H segment within {SEG_TOL_DEG:g} deg",
                 _segments(False), ORIGIN_DEG))
    add(Codebook("segments4_d", ("origin", "crossbar", "left bar", "right bar"), "deg",
                 f"origin ({ORIGIN_DEG:g} deg) + nearest H part within {SEG_TOL_DEG:g} deg",
                 _segments(True), ORIGIN_DEG))
    add(Codebook("sectors4_d", ("origin", "right", "up", "left", "down"), "deg",
                 f"origin ({ORIGIN_DEG:g} deg) + four direction sectors", _sectors(4), ORIGIN_DEG))
    add(Codebook("sectors8_d", ("origin", "R", "UR", "U", "UL", "L", "DL", "D", "DR"), "deg",
                 f"origin ({ORIGIN_DEG:g} deg) + eight direction sectors", _sectors(8), ORIGIN_DEG))
    add(Codebook("rings_d", ("< 2°", "2–6°", "6–10°", "> 10°"), "deg",
                 "eccentricity from the fixation point (deg)", _rings(False)))
    add(Codebook("rings_side_d",
                 ("origin", "2–6° L", "2–6° R", "6–10° L", "6–10° R", "> 10° L", "> 10° R"), "deg",
                 "eccentricity rings × side (deg)", _rings(True)))
    add(Codebook("ybands_d", ("origin", "below", "crossbar level", "upper arms", "above"), "deg",
                 "origin + vertical bands (deg)", _ybands, ORIGIN_DEG))
    for w in (0.35, 0.5):
        add(Codebook(f"capsules4_u{w:g}", EXITS, "unit",
                     f"four capsules origin → exit, half width {w:g} unit H, no origin ball",
                     _capsules4("unit", w)))
    for w in (2.0, 3.0):
        add(Codebook(f"capsules4_d{w:g}", EXITS, "deg",
                     f"four capsules origin → exit, half width {w:g} deg, no origin ball",
                     _capsules4("deg", w)))
    OQ = ("o-LU", "o-LD", "o-RU", "o-RD")
    for r in (0.5, 0.75):
        add(Codebook(f"oquads_u{r:g}", OQ + ("outside",), "unit",
                     f"origin ball (r = {r:g} unit H) split into quadrants + outside",
                     _origin_split("unit", r, False, False), r))
        add(Codebook(f"ohalves_u{r:g}", ("o-left", "o-right", "outside"), "unit",
                     f"origin ball (r = {r:g} unit H) split left / right + outside",
                     _origin_split("unit", r, True, False), r))
        add(Codebook(f"oquads_balls_u{r:g}", OQ + EXITS, "unit",
                     f"origin quadrants + four exit balls, r = {r:g} unit H",
                     _origin_split("unit", r, False, True), r))
    for r in (2.0, 3.0, 5.0):
        add(Codebook(f"oquads_d{r:g}", OQ + ("outside",), "deg",
                     f"origin ball (r = {r:g} deg) split into quadrants + outside",
                     _origin_split("deg", r, False, False), r))
        add(Codebook(f"ohalves_d{r:g}", ("o-left", "o-right", "outside"), "deg",
                     f"origin ball (r = {r:g} deg) split left / right + outside",
                     _origin_split("deg", r, True, False), r))
        add(Codebook(f"oquads_balls_d{r:g}", OQ + EXITS, "deg",
                     f"origin quadrants + four exit balls, r = {r:g} deg",
                     _origin_split("deg", r, False, True), r))
    return reg


REGISTRY = _build_registry()
SEARCH_KEYS = tuple(REGISTRY)
DEFAULT_KEY = "balls_u0.5"


def get(key):
    try:
        return REGISTRY[key]
    except KeyError:
        raise KeyError(f"unknown codebook {key!r}; known: {', '.join(REGISTRY)}") from None
