import jax.numpy as jnp
import jax
from jax import random, scipy
import network.pvrnn as pvrnn
from misc.tools import JsonDict
import matplotlib.pyplot as plt
import sklearn.cluster as cl
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d
import cv2
from pathlib import Path

#jax.config.update('jax_platform_name', 'cpu')

seq_num = 4

config = JsonDict('config/config.json')

key = random.PRNGKey(0)

hf = h5py.File('dataset/dataset_long.h5', 'r')
motor_pointer = jnp.load('dataset/motor_train.npy')#jnp.array(hf.get('motor'))

print(motor_pointer.shape)

key, key_init = random.split(key)
network = pvrnn.PVRNN(config, key_init)

network.load_state('train_dir_abs/train_vision_proj_KL_Corrected')

dataset_size, T = motor_pointer.shape[0], motor_pointer.shape[1]

key, key_gen = random.split(key)

outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, hs = pvrnn.forward_posterior(network.params_tree,
                                                                                 network.latent_vars,
                                                                                 config,
                                                                                 key_gen,
                                                                                 jnp.array(range(dataset_size)),
                                                                                 T,
                                                                                 dataset_size)

plt.plot(outputs[:, seq_num, 0])
plt.plot(motor_pointer[seq_num, :, 0], '--')
plt.show()

plt.plot(outputs[:, seq_num, 4])
plt.plot(motor_pointer[seq_num, :, 4], '--')
plt.show()

plt.plot(outputs[:, seq_num, 8])
plt.plot(motor_pointer[seq_num, :, 8], '--')
plt.show()

plt.plot(sigma_qs[0][:, seq_num])
plt.show()

plt.plot(sigma_ps[0][:, seq_num])
plt.show()



outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, hs = pvrnn.forward_prior(network.params_tree,
                                                                       network.latent_vars,
                                                                       config,
                                                                       key_gen,
                                                                       jnp.array(range(dataset_size)),
                                                                       T,
                                                                       dataset_size)

print(outputs.shape)

video_dir = 'videos/video_KL_Corrected_abs'

Path(video_dir).mkdir(exist_ok=True)

width = 64
hieght = 64

fps = 30

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i in range(img_outputs.shape[1]):

    video = cv2.VideoWriter(f'{video_dir}/prior_{i}.mp4', fourcc, float(fps), (width, hieght))

    for img in img_outputs[:, i, :, :, 0]:
        frame = np.array(img * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)

    video.release()

outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, hs = pvrnn.forward_posterior(network.params_tree,
                                                                                 network.latent_vars,
                                                                                 config,
                                                                                 key_gen,
                                                                                 jnp.array(range(dataset_size)),
                                                                                 T,
                                                                                 dataset_size)

print(outputs.shape)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i in range(img_outputs.shape[1]):

    video = cv2.VideoWriter(f'{video_dir}/posterior_{i}.mp4', fourcc, float(fps), (width, hieght))

    for img in img_outputs[:, i, :, :, 0]:
        frame = np.array(img * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)

    video.release()

perception_pointer = jnp.load('dataset/vision_train.npy')[:,:,:,:,None].transpose((1, 0, 2, 3, 4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

for i in range(perception_pointer.shape[1]):

    video = cv2.VideoWriter(f'{video_dir}/truth_{i}.mp4', fourcc, float(fps), (width, hieght))

    for img in perception_pointer[:, i, :, :, 0]:
        frame = np.array(img * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        video.write(frame)

    video.release()

#exit(0)


#plt.plot(outputs[:, 0, 0], outputs[:, 0, 1])
#plt.plot(motor_pointer[0, :, 0], motor_pointer[0, :, 1])
#plt.plot(sigma_qs[2][:, 0])
#plt.show()
#exit()

#Z = network.params_tree['latents']['Z_mu']



print(hs[-1].shape)

h = jnp.concatenate(mu_qs, axis=-1)
#h_mu = jnp.concatenate(mu_qs, axis=-1)
#h = jnp.concatenate((h, h_mu), axis=-1)
Z = h.transpose((1, 0, 2)).reshape((h.shape[0] * h.shape[1], -1))#[400 * seq_num: 400 * (seq_num + 1)]
affine_ones = jnp.ones((Z.shape[0], 1))
#Z = jnp.concatenate((Z, affine_ones), axis=1)

print(Z.shape)

[u, s, v] = jnp.linalg.svd(Z.transpose(), full_matrices=False)

print(s)

#diff = torch.tensor(null_space_dynamics(v[:10].t().cpu()[seq_num * 400: (seq_num + 1) * 400]))

tau = 1.

p_s = jnp.diag(jnp.maximum(jnp.zeros(s.shape), (jnp.ones(s.shape[0]) - 1. / (tau * s * s))))

self_expression = v.transpose() @ p_s @ v

symmetrized = jnp.abs(self_expression)

#threshold = jnp.quantile(symmetrized, 0.975)
#symmetrized = symmetrized.at[symmetrized > threshold].set(threshold)

sqrt_norm = (jnp.sqrt(symmetrized).sum() / (self_expression.shape[0] * self_expression.shape[1]))**2
print(f"C 0.5 norm = {sqrt_norm}")
#self_expression = v.t() @ v

print(f"C 1 norm = {symmetrized.sum() / self_expression.shape[0] / self_expression.shape[1]}")

#threshold = jnp.quantile(symmetrized, 0.975)
#symmetrized[symmetrized > threshold] = threshold
plt.imshow(symmetrized[seq_num*400 : (seq_num + 1)*400, seq_num*400 : (seq_num + 1)*400])
plt.show()

diagonal = jnp.diag(jnp.einsum("ij -> i", symmetrized))
inv_sqrt_diagonal = jnp.diag(1. / jnp.sqrt(jnp.einsum("ij -> i", symmetrized)))
laplacian = jnp.diag(jnp.ones(diagonal.shape[0])) - inv_sqrt_diagonal @ symmetrized @ inv_sqrt_diagonal

(sl, ul) = jnp.linalg.eigh(laplacian)

print(f'Affinity matrix eigens: {sl[:10]}')

plt.plot(sl[:50])
plt.show()

#plt.imshow(self_expression[seq_num*400:(seq_num+1)*400, seq_num*400:(seq_num+1)*400].cpu().detach().numpy())
#plt.show()
print(f'Covariance matrix eigens: {s}')


def error_function(vectors, self_ex):
    reconstr = jnp.einsum("ki, kj -> ji", vectors, self_ex)

    error = ((reconstr - vectors)**2).mean()
    r2 = 1. - error / (jnp.var(vectors))

    return r2


error = error_function(Z, self_expression)
#error = error_function(Z[sorted_train_idx, :-1], self_expression)
print(f'Error = {error}')


def smooth_kmeans(X, clusters_num):

    kmeans = cl.KMeans(clusters_num).fit(X)
    centers = jnp.array(kmeans.cluster_centers_)
    distances = jnp.sqrt(((X[:, None, :] - centers)**2).sum(axis=-1))#torch.sqrt(((ul[:, None, :] - centers)**2).sum(dim=-1))


    probs = jax.nn.softmax(-distances * 100., axis=1)
    probs = jnp.array(gaussian_filter1d(probs, 6., axis=0))

    labels_all = jnp.argmax(probs, axis=-1)#kmeans.labels_[seq_num * 400: (seq_num + 1) * 400]

    return labels_all


clusters_num = 3
#QQT_flat = all_projection_matrices(X).detach().requires_grad_(False)
ul = ul[:, :clusters_num]
all_labels = smooth_kmeans(ul, clusters_num)


def get_subspaces(Z, labels, num_clusters: int, sub_dim: int):
    subspaces = jnp.zeros((num_clusters, sub_dim, Z.shape[1]))
    Z -= Z.mean(dim=0)
    for i in range(num_clusters):
        [u, s, v] = jnp.linalg.svd(Z[labels == i].t(), full_matrices=False)
        #pca = PCA(sub_dim).fit(Z[labels == i].numpy())
        print("Explained variance:")
        print(s)
        subspaces[i] = jnp.array(u.transpose()[:sub_dim])

    grassmann_similarity_matrix = (jnp.einsum("npd, mqd -> nmpq", subspaces, subspaces)**2).sum(dim=(2, 3))
    print("Grassmann similarity matrix:")
    print(grassmann_similarity_matrix)

    return subspaces


#subspaces = get_subspaces(Z.cpu().detach(), all_labels, clusters_num, 6)


def assign_subspace_proximity_based_labels(Z, subspaces):
    Z -= Z.mean(axis=0)
    norm_Z = Z#torch.einsum("sd, s -> sd", Z, 1. / Z.norm(dim=-1))
    projected_Z = torch.einsum("sd, npd, npo -> sno", norm_Z, subspaces, subspaces)
    norms = projected_Z.norm(dim=-1)
    labels_all = torch.argmax(norms, dim=-1)
    return labels_all


#all_labels = assign_subspace_proximity_based_labels(Z.cpu(), subspaces)

labels = all_labels[seq_num * 400: (seq_num + 1) * 400]

print(labels)
samples_num = len(labels)

cmap = plt.cm.get_cmap('Accent', clusters_num)

base_colors = jnp.array([cmap(i) for i in list(range(clusters_num))])
#colors = torch.einsum("sp, pc -> sc", probs, base_colors)[:, :3]

colour_labels = np.ones((200, samples_num, 3))
for cluster in range(clusters_num):
    colour_labels[:, labels == cluster] = np.array(cmap(cluster))[:3]#colors[cluster]

#colour_probs = colors[None, :, :].expand(200, -1, -1)[:, seq_num * 400: (seq_num + 1) * 400].detach().numpy()

fig, axes = plt.subplots(nrows=2, ncols=1,figsize=(6,3))
plt.rc('font', size=24)

axes[0].plot(motor_pointer[seq_num, :].mean(axis=-1))
#axes[0].plot(target[seq_num, :][:,9])
axes[0].set_xlim([0, 400])
axes[0].set_title("Average joint angles dynamics")

#ticks = list(range(0, samples_num + 1, samples_num // 4))
#axes[1].xticks(ticks, list(range(5)))
#axes[1].yticks([])

#axes[1].imshow(colour_probs, aspect='auto')

#ticks = list(range(0, samples_num + 1, samples_num // 4))
#axes[2].xticks(ticks, list(range(5)))
#axes[2].yticks([])

axes[1].imshow(colour_labels, aspect='auto')
axes[1].set_title("Primitive labels")
#axes[2].plot(sq_diffs[seq_num])
#axes[2].set_xlim([0, 400])
#plt.subplot(2, 1, 2)
#plt.plot(diff)
#plt.plot(proj_sigma[0][0].detach().numpy())

#axes[3].plot(sigma_top_c[seq_num * 400: (seq_num + 1) * 400])
#axes[3].set_xlim([0, 400])

plt.tight_layout()
plt.show()
