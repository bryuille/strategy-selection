import numpy as np


def sigmoid_dv(dv):
    return 1.0 / (1.0 + np.exp(-dv))


def path_type_for(maze, exit_idx):
    return (maze - 1) * 4 + exit_idx


def maze_for(path_type):
    return int((path_type - 1) // 4 + 1)
