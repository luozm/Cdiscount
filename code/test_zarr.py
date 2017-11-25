"""
test zarr for data storage and retrieval.

"""
import numpy as np
import zarr
from utils import utils

utils_dir = utils.utils_dir

a = np.load(utils_dir+'bottleneck_features_train_level1.npy')
z = zarr.array(
    a,
    store=utils_dir+'bottleneck_features_train_level1.zarr',
)

#z1 = zarr.open(utils_dir+'bottleneck_features_train_level1.zarr', mode='r')

#z = z1[:]
print()