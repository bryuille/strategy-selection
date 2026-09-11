from pathlib import Path

import numpy as np

from data.builder import (
    CLEAN_EYE_DATA_KEYS,
    EYE_BEHAVIORAL_FIELDS,
    EYE_BEHAVIORAL_OPTIONAL_FIELDS,
    build_eye_behavioral_data,
    build_eye_data,
    build_post_flash_metadata,
    build_post_flash_timebins,
    build_pre_flash_metadata,
    build_pre_flash_timebins,
    build_trial_metadata,
    build_trial_timebins,
    clean_eye_data,
)
from data.config import (
    SESSION,
    behavioral_npz_path,
    monkey_for_session,
    processed_npz,
    single_trial_npz_path,
)
from data.convert import load_npz, load_session_data
from data.labeler import (
    build_lr_choices,
    build_strategy_choices,
    build_svm_choices,
)


def savez_atomic(path, **data):
    """`np.savez` to `path` via a hidden sibling and an atomic rename.

    Same reasoning as `data.convert.write_npz`, and it matters for two distinct
    hazards. A `np.savez` straight onto `path` leaves a truncated-but-existing
    npz if the process dies mid-write (a Slurm TIMEOUT, a scancel, a full
    disk), and the next run sees the file, believes the cache is warm, and
    carries the corruption downstream. It also means two jobs building the same
    cache concurrently can have one read what the other is halfway through
    writing -- which is exactly what happens when the pre-flash sweeps run in
    parallel rather than being serialised behind a Slurm dependency.

    `Path.replace` is atomic within a filesystem, so a reader sees either the
    previous complete file or the new complete file, never a partial one. The
    temp name is passed as an open handle because `np.savez` appends ".npz" to
    a *path* lacking the suffix, which would defeat the rename.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **data)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path


def ensure_npz(path, build):
    if path.exists():
        return load_npz(path)
    data = build()
    savez_atomic(path, **data)
    return data


def resolve_session(session=None):
    return SESSION if session is None else session


def session_data(session=None):
    return load_session_data(resolve_session(session))


def load_trial_timebins(session=None):
    session = resolve_session(session)
    return ensure_npz(
        processed_npz(f"{session}_trial_timebins"),
        lambda: {"trial_timebins": build_trial_timebins(session_data(session))},
    )["trial_timebins"]


def load_trial_metadata(session=None):
    session = resolve_session(session)
    loaded = ensure_npz(
        processed_npz(f"{session}_trial_metadata"),
        lambda: dict(
            zip(
                ("path_type", "flash2_ms", "flash3_ms", "trial_mask"),
                build_trial_metadata(session_data(session)),
            )
        ),
    )
    return loaded["path_type"], loaded["flash2_ms"], loaded["flash3_ms"], loaded["trial_mask"]


def load_pre_flash_timebins(session=None):
    session = resolve_session(session)
    return ensure_npz(
        processed_npz(f"{session}_pre_flash_timebins"),
        lambda: {
            "pre_flash_timebins": build_pre_flash_timebins(session_data(session))
        },
    )["pre_flash_timebins"]


def load_pre_flash_metadata(session=None):
    session = resolve_session(session)
    loaded = ensure_npz(
        processed_npz(f"{session}_pre_flash_metadata"),
        lambda: dict(
            zip(
                ("path_type", "flash1_ms", "trial_mask"),
                build_pre_flash_metadata(session_data(session)),
            )
        ),
    )
    return loaded["path_type"], loaded["flash1_ms"], loaded["trial_mask"]


def load_post_flash_timebins(session=None):
    session = resolve_session(session)
    return ensure_npz(
        processed_npz(f"{session}_post_flash_timebins"),
        lambda: {
            "post_flash_timebins": build_post_flash_timebins(session_data(session))
        },
    )["post_flash_timebins"]


def load_post_flash_metadata(session=None):
    session = resolve_session(session)
    loaded = ensure_npz(
        processed_npz(f"{session}_post_flash_metadata"),
        lambda: dict(
            zip(
                ("path_type", "trial_mask"),
                build_post_flash_metadata(session_data(session)),
            )
        ),
    )
    return loaded["path_type"], loaded["trial_mask"]


########## Labels


def load_lr_choices(session=None):
    session = resolve_session(session)
    monkey = monkey_for_session(session)
    return ensure_npz(
        processed_npz(f"{session}_lr_choices"),
        lambda: {
            "lr_choices": build_lr_choices(
                load_npz(behavioral_npz_path(monkey, session))
            )
        },
    )["lr_choices"]


def load_single_trial_range(session=None):
    """`min_value`, `max_value`: the session's published single-trial window."""
    session = resolve_session(session)
    listed = load_npz(single_trial_npz_path(monkey_for_session(session), session))
    return int(listed["min_value"]), int(listed["max_value"])


def load_strategy_choices(session=None):
    session = resolve_session(session)
    monkey = monkey_for_session(session)

    return ensure_npz(
        processed_npz(f"{session}_strategy_choices"),
        lambda: {
            "strategy_choices": build_strategy_choices(
                load_trial_timebins(session),
                load_npz(behavioral_npz_path(monkey, session)),
                *load_single_trial_range(session),
            )
        },
    )["strategy_choices"]


def load_svm_choices(session=None):
    """Maze-1-vs-6 SVM strategy labels and CV diagnostics for `session`.

    Mirrors `load_strategy_choices`'s cache convention (same key,
    ``strategy_choices``, so `eye_pre_flash.classifier.labels` reads either
    with only a stem change) but returns the full dict -- `svm_auc` and the
    rest of `build_svm_choices`'s diagnostics are cached alongside it, not
    thrown away.
    """
    session = resolve_session(session)
    monkey = monkey_for_session(session)

    return ensure_npz(
        processed_npz(f"{session}_strategy_svm"),
        lambda: build_svm_choices(
            load_trial_timebins(session),
            load_npz(behavioral_npz_path(monkey, session)),
            *load_single_trial_range(session),
        ),
    )


########## Eye data (all sessions)


def load_eye_data(monkey="Faure"):
    return ensure_npz(
        processed_npz(f"{monkey}_eye_data"),
        lambda: build_eye_data(monkey),
    )


def load_eye_behavioral_data(monkey="Faure"):
    """Trial-level behavioral fields for every session with eye data.

    A cache written before `EYE_BEHAVIORAL_OPTIONAL_FIELDS` existed is rebuilt
    in place. Optional fields are NaN for sessions whose behavioral npz predate
    their addition to `data.convert.BEHAVIORAL_FIELDS` (fix by re-converting
    behavioral data where `data/mat/` lives, then deleting this cache).
    """
    path = processed_npz(f"{monkey}_eye_behavioral")
    wanted = set(EYE_BEHAVIORAL_FIELDS) | set(EYE_BEHAVIORAL_OPTIONAL_FIELDS)
    if path.exists():
        loaded = load_npz(path)
        if wanted <= set(loaded):
            return loaded
    built = build_eye_behavioral_data(monkey)
    savez_atomic(path, **built)
    return built


def load_clean_eye_data(monkey="Faure", *, start_ms=None, end_ms=None):
    """Saccade/fixation events for `monkey`, cached per detection window.

    `start_ms` clips each trial to ``[fix_start - start_ms, fix_start - end_ms]``
    *before* the detectors run, and the cache stem carries the window so the
    windowed and whole-trial event tables never overwrite each other. The two
    are different analyses, not the same answer computed twice -- I-DT is a
    greedy sequential scan, so clipping before vs after detection agrees on
    only 7.8% of trials (measured).

    Pre-fixation callers pass the analysis window and get a ~5x cheaper pass
    (1466 ms against a median 7322 ms trial), which is every caller today.
    `start_ms=None` keeps the whole trial and is left in for an analysis whose
    events sit outside the pre-fixation window.
    """
    if start_ms is None:
        stem = f"{monkey}_clean_eye_data"
    else:
        stem = f"{monkey}_clean_eye_data_s{int(start_ms)}_e{int(end_ms or 0)}"
    path = processed_npz(stem)
    if path.exists():
        loaded = load_npz(path)
        if set(CLEAN_EYE_DATA_KEYS) <= set(loaded):
            return loaded
    cleaned = clean_eye_data(
        load_eye_data(monkey), monkey=monkey, start_ms=start_ms, end_ms=end_ms
    )
    savez_atomic(path, **cleaned)
    return cleaned


def load_attractor_eye_data(monkey="Faure"):
    """Warped positions, validity and timebase. K-independent by construction --
    the codebook lives in `eye_pre_flash.classifier.features`, not here."""
    from data.attractor import ATTRACTOR_EYE_DATA_KEYS, produce_attractor_eye_data

    path = processed_npz(f"{monkey}_attractor_eye_data")
    if path.exists():
        loaded = load_npz(path)
        if set(ATTRACTOR_EYE_DATA_KEYS) <= set(loaded):
            return loaded
    data = produce_attractor_eye_data(load_eye_data(monkey), monkey=monkey)
    savez_atomic(path, **data)
    return data
