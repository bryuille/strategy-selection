import numpy as np


def sigmoid(dv):
    return 1.0 / (1.0 + np.exp(-dv))


def path_type_for(maze, exit_idx):
    return (maze - 1) * 4 + exit_idx


def maze_for(path_type):
    return int((path_type - 1) // 4 + 1)


def strategy_label_maze_summary():
    """Print % hierarchical (0) / sequential (1) strategy labels by maze group."""
    from data.loader import load_strategy_choices, load_trial_metadata

    y = load_strategy_choices()
    pt, _, _, mask = load_trial_metadata()

    valid = mask & ~np.isnan(y)
    mazes = np.array([maze_for(p) for p in pt[valid]])
    labels = y[valid]
    
    print("\nPer-maze breakdown:")
    for m in range(1, 7):
        sel = mazes == m
        n = sel.sum()
        n1 = (labels[sel] == 1).sum()
        n0 = (labels[sel] == 0).sum()
        print(
            f"  maze {m}: n={n} hier(0)={n0} ({100 * n0 / n:.1f}%)"
            f"seq(1)={n1} ({100 * n1 / n:.1f}%)"
        )