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
from data.convert import load_data, load_npz
from data.config import SESSION, processed_npz
from data.labeler import (
    build_lr_choices,
    build_strategy_choices,
)


def _ensure_npz(path, build):
    if path.exists():
        return load_npz(path)
    data = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def load_trial_timebins():
    return _ensure_npz(
        processed_npz(f"{SESSION}_trial_timebins"),
        lambda: {"trial_timebins": build_trial_timebins(load_data())},
    )["trial_timebins"]


def load_trial_metadata():
    loaded = _ensure_npz(
        processed_npz(f"{SESSION}_trial_metadata"),
        lambda: dict(
            zip(
                ("path_type", "flash2_ms", "flash3_ms", "trial_mask"),
                build_trial_metadata(load_data()),
            )
        ),
    )
    return loaded["path_type"], loaded["flash2_ms"], loaded["flash3_ms"], loaded["trial_mask"]


def load_pre_flash_timebins():
    return _ensure_npz(
        processed_npz(f"{SESSION}_pre_flash_timebins"),
        lambda: {"pre_flash_timebins": build_pre_flash_timebins(load_data())},
    )["pre_flash_timebins"]


def load_pre_flash_metadata():
    loaded = _ensure_npz(
        processed_npz(f"{SESSION}_pre_flash_metadata"),
        lambda: dict(
            zip(
                ("path_type", "flash1_ms", "trial_mask"),
                build_pre_flash_metadata(load_data()),
            )
        ),
    )
    return loaded["path_type"], loaded["flash1_ms"], loaded["trial_mask"]


def load_post_flash_timebins():
    return _ensure_npz(
        processed_npz(f"{SESSION}_post_flash_timebins"),
        lambda: {"post_flash_timebins": build_post_flash_timebins(load_data())},
    )["post_flash_timebins"]


def load_post_flash_metadata():
    loaded = _ensure_npz(
        processed_npz(f"{SESSION}_post_flash_metadata"),
        lambda: dict(
            zip(("path_type", "trial_mask"), build_post_flash_metadata(load_data()))
        ),
    )
    return loaded["path_type"], loaded["trial_mask"]


########## Labels


def load_lr_choices():
    return _ensure_npz(
        processed_npz(f"{SESSION}_lr_choices"),
        lambda: {"lr_choices": build_lr_choices(load_data())},
    )["lr_choices"]


def load_strategy_choices():
    return _ensure_npz(
        processed_npz(f"{SESSION}_strategy_choices"),
        lambda: {
            "strategy_choices": build_strategy_choices(
                load_trial_timebins(), load_trial_metadata()[0]
            )
        },
    )["strategy_choices"]


def load_all_session_strategy_choices(monkey="Faure"):
    from data.labeler import build_all_session_strategy_choices
    path = processed_npz(f"{monkey}_strategy_choices_all_sessions")
    if path.exists():
        return load_npz(path)
    data = build_all_session_strategy_choices(monkey)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
    return data


def load_session_strategy_choices(session, monkey="Faure"):
    from data.labeler import build_session_strategy_choices
    path = processed_npz(f"{monkey}_{session}_strategy_choices")
    if path.exists():
        return load_npz(path)["strategy_choices"]
    choices = build_session_strategy_choices(session, monkey=monkey)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strategy_choices=choices)
    return choices


########## Eye data (all sessions)


def load_eye_data(monkey="Faure"):
    return _ensure_npz(
        processed_npz(f"{monkey}_eye_data"),
        lambda: build_eye_data(monkey),
    )


def load_eye_behavioral_data(monkey="Faure"):
    return _ensure_npz(
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


def load_attractor_strategy_choices(monkey="Faure"):
    from data.labeler import build_attractor_strategy_choices
    path = processed_npz(f"{monkey}_attractor_strategy_choices")
    if path.exists():
        return load_npz(path)["strategy_choices"]
    attractor = load_attractor_eye_data(monkey)
    behavioral = load_eye_behavioral_data(monkey)
    choices = build_attractor_strategy_choices(attractor, behavioral)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, strategy_choices=choices)
    return choices

