import jax.numpy as jnp
import jax
from jax import random, scipy
import network.pvrnn as pvrnn
from misc.tools import JsonDict
import matplotlib.pyplot as plt
import sklearn.cluster as cl
import h5py
import numpy as np

mse_KL_random = np.loadtxt('train_dir/train_random_vision/KL.txt')
mse_KL_proj = np.loadtxt('train_dir/train_vision_proj/KL.txt')
mse_KL_corr_proj = np.loadtxt('train_dir_softplus/train_vision_proj_KL_Corrected/KL.txt')

plt.rc('font', size=24)

p1, = plt.semilogy(mse_KL_random[:3000])
p2, = plt.semilogy(mse_KL_proj[:3000])
p3, = plt.semilogy(mse_KL_corr_proj[:3000])

plt.legend([p1, p2, p3], ['Random', 'Random Projection', 'Random Projection with small sigma'])

plt.show()
