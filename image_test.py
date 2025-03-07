import jax.numpy as jnp
from jax import lax, vmap
from jax import random
import h5py
import jax.nn as nn

from jax import scipy

key = random.PRNGKey(0)
o = random.normal(key, (100, 30))

o = nn.standardize(o)
print(len(o.std(axis=-1)))

exit(0)

hf = h5py.File('dataset/dataset_dof.h5', 'r')

motor_pointer = hf.get('motor')
perception_pointer = hf.get('perception')

images = jnp.array(perception_pointer[0])

random_images = jnp.ones((100, 3, 4, 256))

kernel_1 = jnp.ones((8, 10, 256, 128))
kernel_2 = jnp.ones((2, 2, 128, 64))
kernel_3 = jnp.ones((2, 2, 64, 3))

convolved_1 = lax.conv_transpose(random_images, kernel_1, (2, 2), 'VALID')
convolved_2 = lax.conv_transpose(convolved_1, kernel_2, (2, 2), 'VALID')
convolved_3 = lax.conv_transpose(convolved_2, kernel_3, (2, 2), 'VALID')

print(convolved_1.shape)
print(convolved_2.shape)
print(convolved_3.shape)
print(images.shape)

