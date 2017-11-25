"""
Learning rate schedules.

"""
import math
import numpy as np


# LearningRate Schedule for Snapshot Ensemble
def cosine_anneal_schedule(initial_lr, num_epoch, num_snapshots):
    def schedule(t):
        cos_inner = np.pi * (t % (num_epoch // num_snapshots))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= num_epoch // num_snapshots
        cos_out = np.cos(cos_inner) + 1
        return float(initial_lr / 2 * cos_out)
    return schedule


# learning rate schedule (exponential decay)
def exp_decay_schedule(initial_lr, decay_value, decay_epoch):
    def schedule(t):
        lrate = initial_lr * math.pow(decay_value, math.floor((1 + t) / decay_epoch))
        print("lr:", lrate)
        return lrate
    return schedule
