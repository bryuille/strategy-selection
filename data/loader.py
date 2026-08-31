import numpy as np

from data.builder import (
    CLEAN_EYE_DATA_KEYS,
    build_eye_data,
    build_eye_behavioral_data,
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
)


def ensure_npz(path, build):
    if path.exists():
        return load_npz(path)
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
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


########## Eye data (all sessions)


def load_eye_data(monkey="Faure"):
    return ensure_npz(
        processed_npz(f"{monkey}_eye_data"),
        lambda: build_eye_data(monkey),
    )


def load_eye_behavioral_data(monkey="Faure"):
    return ensure_npz(
        processed_npz(f"{monkey}_eye_behavioral"),
        lambda: build_eye_behavioral_data(monkey),
    )


def load_clean_eye_data(monkey="Faure"):
    path = processed_npz(f"{monkey}_clean_eye_data")
    if path.exists():
        loaded = load_npz(path)
        if set(CLEAN_EYE_DATA_KEYS) <= set(loaded):
            return loaded
    cleaned = clean_eye_data(load_eye_data(monkey), monkey=monkey)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **cleaned)
    return cleaned


def load_attractor_eye_data(monkey="Faure", k=None):
    from data.attractor import ATTRACTOR_EYE_DATA_KEYS, DEFAULT_K, produce_attractor_eye_data

    k = DEFAULT_K if k is None else int(k)
    specific = processed_npz(f"{monkey}_attractor_k{k}_eye_data")
    default = processed_npz(f"{monkey}_attractor_eye_data")
    for path in (specific, default):
        if not path.exists():
            continue
        loaded = load_npz(path)
        if set(ATTRACTOR_EYE_DATA_KEYS) <= set(loaded) and int(loaded["codebook_k"]) == k:
            return loaded
    data = produce_attractor_eye_data(load_eye_data(monkey), monkey=monkey, k=k)
    specific.parent.mkdir(parents=True, exist_ok=True)
    np.savez(specific, **data)
    if k == DEFAULT_K:
        np.savez(default, **data)
    return data
