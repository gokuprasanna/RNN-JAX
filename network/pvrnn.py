import numpy as np
import jax.numpy as jnp
import jax.lax as lax
from jax import random, grad, jit, value_and_grad, nn, vmap, tree_util, scipy
from functools import partial
from misc.tools import JsonDict, random_split_like_tree
import pickle

from optax import hvp


class PVRNN:

    def __init__(self, config: dict, key):
        self.config = config
        self.params_tree = {"weights": {}, "duals": {}}
        self.opt_labels = {"weights": "weights", "duals": "duals"}

        N_above = 1#config['latent_space_dim']
        self.params_tree["weights"]["layers"] = []
        for (i, layer) in enumerate(config['layers']):
            N = layer['N']
            N_Z = layer['N_Z']

            key, n_key = random.split(key)
            self_mu = random.uniform(n_key, (N, N_Z), minval=-1. / jnp.sqrt(N), maxval=1. / jnp.sqrt(N))
            self_mu_b = jnp.zeros((N_Z,))

            key, n_key = random.split(key)
            self_sigma = random.uniform(n_key, (N, N_Z), minval=-1. / jnp.sqrt(N), maxval=1. / jnp.sqrt(N))
            self_sigma_b = jnp.zeros((N_Z,))

            if layer['type'] == 'GRU':

                key, r_key = random.split(key)
                r_self = random.uniform(r_key, (N, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))
                r_self_b = jnp.zeros((N,))

                key, r_key = random.split(key)
                r_above = random.uniform(r_key, (N_above, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, r_key = random.split(key)
                r_z = random.uniform(r_key, (N_Z, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, z_key = random.split(key)
                z_self = random.uniform(z_key, (N, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))
                z_self_b = jnp.zeros((N,))

                key, z_key = random.split(key)
                z_above = random.uniform(z_key, (N_above, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, z_key = random.split(key)
                z_z = random.uniform(z_key, (N_Z, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, n_key = random.split(key)
                n_self = random.uniform(n_key, (N, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))
                n_self_b = jnp.zeros((N,))

                key, n_key = random.split(key)
                n_above = random.uniform(n_key, (N_above, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, n_key = random.split(key)
                n_z = random.uniform(n_key, (N_Z, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))
                n_z_b = jnp.zeros((N,))


                self.params_tree['weights']['layers'].append({
                                                          'r_self': r_self,
                                                          'r_self_b': r_self_b,
                                                          'r_above': r_above,
                                                          'r_z': r_z,
                                                          'z_self': z_self,
                                                          'z_self_b': z_self_b,
                                                          'z_above': z_above,
                                                          'z_z': z_z,
                                                          'n_self': n_self,
                                                          'n_self_b': n_self_b,
                                                          'n_above': n_above,
                                                          'n_z': n_z,
                                                          'n_z_b': n_z_b,
                                                          'self_mu': self_mu,
                                                          'self_mu_b': self_mu_b,
                                                          'self_sigma': self_sigma,
                                                          'self_sigma_b': self_sigma_b
                                                          })

            elif layer['type'] == 'PVRNN':

                key, r_key = random.split(key)
                r_self = random.uniform(r_key, (N, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))
                r_self_b = jnp.zeros((N,))

                key, z_key = random.split(key)
                z_self = random.uniform(z_key, (N_Z, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                key, u_key = random.split(key)
                u_self = random.uniform(u_key, (N_above, N), minval=-1./jnp.sqrt(N), maxval=1./jnp.sqrt(N))

                self.params_tree['weights']['layers'].append({
                    'r_self': r_self,
                    'r_self_b': r_self_b,
                    'z_self': z_self,
                    'u_self': u_self,
                    'self_mu': self_mu,
                    'self_mu_b': self_mu_b,
                    'self_sigma': self_sigma,
                    'self_sigma_b': self_sigma_b,
                })

            N_above = N

        N = config['output_dim']

        key, w_key = random.split(key)
        w_output = random.uniform(w_key, (N_above, N), minval=-1. / jnp.sqrt(N), maxval=1. / jnp.sqrt(N))
        b_output = jnp.zeros((N,))
        self.params_tree['weights']["w_output"] = w_output
        self.params_tree['weights']["b_output"] = b_output

        key, w_key = random.split(key)
        img_c, img_w, img_h = config['encoded_image_shape']
        total_dim = img_c * img_w * img_h
        w_image = random.uniform(w_key, (N_above, total_dim), minval=-1. / jnp.sqrt(total_dim), maxval=1. / jnp.sqrt(total_dim))
        b_image = jnp.zeros((total_dim,))
        self.params_tree['weights']["w_image"] = w_image
        self.params_tree['weights']["b_image"] = b_image

        self.params_tree['weights']['deconv_layers'] = []
        for (i, deconv_layer) in enumerate(config['deconvolution_layers']):
            k_shape = deconv_layer['kernel']
            out_dim = k_shape[0] * k_shape[1] * k_shape[3]
            key, k_key = random.split(key)
            kernel = random.uniform(k_key, k_shape, minval=-1. / jnp.sqrt(out_dim), maxval=1. / jnp.sqrt(out_dim))

            self.params_tree['weights']['deconv_layers'].append(kernel)

        self.params_tree['duals']["dual_motor_loss"] = jnp.ones((1,))
        self.params_tree['duals']["dual_image_loss"] = jnp.ones((1,))

    def init_Z(self, dataset, img_dataset, key, projection=False):
        T, batch_dim = dataset.shape[0], dataset.shape[1]
        motor_dim = dataset.shape[2]
        img_dim1, img_dim2, img_dim3 = img_dataset.shape[2], img_dataset.shape[3], img_dataset.shape[4]
        self.latent_vars = []

        num_layers = len(self.config['layers']) * 3

        for (ln, conf_layer) in enumerate(self.config['layers']):

            tau = conf_layer['tau']
            delay = int(tau * 5)
            N_Z = conf_layer['N_Z']

            mus = jnp.zeros((T, batch_dim, N_Z))
            log_sigmas = jnp.zeros((T, batch_dim, N_Z))
            if projection:

                key, key_mu, key_s = random.split(key, num=3)
                mu_proj_matr = random.normal(key_mu, ((N_Z, delay, motor_dim))) / N_Z
                sigma_proj_matr = random.normal(key_s, ((N_Z, delay, motor_dim))) / N_Z

                key, key_mu, key_s = random.split(key, num=3)
                mu_img_proj_matr = random.normal(key_mu, ((N_Z, delay, img_dim1, img_dim2, img_dim3))) / N_Z
                sigma_img_proj_matr = random.normal(key_s, ((N_Z, delay, img_dim1, img_dim2, img_dim3))) / N_Z

                p_mu_low_pass = jnp.zeros((batch_dim, N_Z))
                p_sigma_low_pass = jnp.zeros((batch_dim, N_Z))

                p_mu = jnp.zeros((batch_dim, N_Z))
                p_sigma = jnp.zeros((batch_dim, N_Z))

                mask = jnp.zeros((delay,))
                mask = mask.at[int(((ln + 2.) / num_layers) * delay): int(((ln + 2.) / num_layers + 1.) * delay)].set(1.)

                for t in range(T):
                    l_idx = min(t, T - delay)

                    dataset_idx = dataset[l_idx: l_idx + delay]#jnp.einsum("btd, t -> btd", scipy.fft.dct(dataset[:, l_idx: l_idx + delay], norm='ortho'), mask)
                    img_dataset_idx = img_dataset[l_idx: l_idx + delay]

                    p_mu_new = jnp.einsum('ztm, tbm -> bz', mu_proj_matr, dataset_idx)
                    p_img_mu_new = jnp.einsum('ztwhc, tbwhc -> bz', mu_img_proj_matr, img_dataset_idx)
                    p_sigma_new = jnp.einsum('ztm, tbm -> bz', sigma_proj_matr, dataset_idx)
                    p_img_sigma_new = jnp.einsum('ztwhc, tbwhc -> bz', sigma_img_proj_matr, img_dataset_idx)

                    p_mu_new = p_mu_new / jnp.linalg.norm(p_mu_new, axis=-1)[:, None]
                    p_sigma_new = p_sigma_new / jnp.linalg.norm(p_sigma_new, axis=-1)[:, None]

                    p_img_mu_new = p_img_mu_new / jnp.linalg.norm(p_img_mu_new, axis=-1)[:, None]
                    p_img_sigma_new = p_img_sigma_new / jnp.linalg.norm(p_img_sigma_new, axis=-1)[:, None]

                    #p_mu = p_mu - p_mu_low_pass
                    #p_sigma = p_sigma - p_sigma_low_pass

                    #p_mu_low_pass = (1. - 1. / tau) * p_mu_low_pass + p_mu_new / tau
                    #p_sigma_low_pass = (1. - 1. / tau) * p_sigma_low_pass + p_sigma_new / tau

                    p_mu = (1. - 1. / tau) * p_mu + p_mu_new / tau
                    p_sigma = jnp.zeros_like(p_img_sigma_new)#jnp.ones_like(p_img_sigma_new)*(-5.)#(p_img_sigma_new) #(1. - 1. / tau) * (p_sigma + p_sigma_low_pass)

                    mus = mus.at[t].set(p_mu)
                    log_sigmas = log_sigmas.at[t].set(p_sigma)

                    #print(f'Mean mu = {mus.mean()}, Std mu = {mus.std()}')

            self.latent_vars.append({'mus': mus, 'log_sigmas': log_sigmas})

    def init_Z_with(self, mu_q, sigma_q):
        self.latent_vars = []
        for i in range(len(mu_q)):
            log_sigmas = jnp.log(sigma_q[i])
            self.latent_vars.append({'mus': mu_q[i], 'log_sigmas': log_sigmas})

    def save_state(self, path_to_save, epoch=None):
        if epoch is None:
            with open(path_to_save + '/weights.pickle', 'wb') as file:
                pickle.dump(self.params_tree, file)
            with open(path_to_save + '/latents.pickle', 'wb') as file:
                pickle.dump(self.latent_vars, file)
        else:
            with open(path_to_save + f'/weights_{epoch}.pickle', 'wb') as file:
                pickle.dump(self.params_tree, file)
            with open(path_to_save + f'/latents_{epoch}.pickle', 'wb') as file:
                pickle.dump(self.latent_vars, file)


    def load_state(self, path_to_load, epoch=None):
        if epoch is None:
            with open(path_to_load + '/weights.pickle', 'rb') as file:
                self.params_tree = pickle.load(file)
            with open(path_to_load + '/latents.pickle', 'rb') as file:
                self.latent_vars = pickle.load(file)
        else:
            with open(path_to_load + f'/weights_{epoch}.pickle', 'rb') as file:
                self.params_tree = pickle.load(file)
            with open(path_to_load + f'/latents_{epoch}.pickle', 'rb') as file:
                self.latent_vars = pickle.load(file)
        #T = len(self.latent_vars[0]['mus'])
        #self.latent_vars = tree_util.tree_map(lambda a: jnp.array(a), self.latent_vars, is_leaf=lambda a: len(a) == T)


def to_scan(h, latents_q, posterior: bool, params, config, batch_size):
    out_h = []
    h_above = jnp.zeros((batch_size, 1))
    mu_ps = []
    mu_qs = []
    sigma_ps = []
    sigma_qs = []
    for i in range(len(config['layers'])):
        #mu_q, sigma_q, eps = mu_sigma_eps_q
        layer = params['weights']['layers'][i]

        log_sigma_p = jnp.einsum('ij, bi -> bj', layer['self_sigma'], h[i]) + layer['self_sigma_b']
        sigma_p = jnp.abs(log_sigma_p)#nn.softplus(log_sigma_p)#jnp.exp(log_sigma_p)
        mu_p = jnp.einsum('ij, bi -> bj', layer['self_mu'], h[i]) + layer['self_mu_b']

        mu_ps.append(mu_p)
        sigma_ps.append(sigma_p)

        eps = latents_q[i]['eps']
        if posterior:
            log_sigma_q = latents_q[i]['log_sigmas']
            sigma_q = jnp.abs(log_sigma_q)#nn.softplus(log_sigma_q)#jnp.exp(log_sigma_q)
            mu_q = latents_q[i]['mus']
            z = mu_q + sigma_q * eps
            mu_qs.append(mu_q)
            sigma_qs.append(sigma_q)

        else:
            z = mu_p + sigma_p * eps

        if config['layers'][i]['type'] == 'GRU':

            gru_r = jnp.einsum('ij, bi -> bj', layer['r_self'], h[i]) + jnp.einsum('ij, bi -> bj', layer['r_above'], h_above) + \
                    jnp.einsum('ij, bi -> bj', layer['r_z'], z) + layer['r_self_b']
            gru_r = nn.sigmoid(gru_r)
            gru_z = jnp.einsum('ij, bi -> bj', layer['z_self'], h[i]) + jnp.einsum('ij, bi -> bj', layer['z_above'], h_above) + \
                    jnp.einsum('ij, bi -> bj', layer['z_z'], z) + layer['z_self_b']
            gru_z = nn.sigmoid(gru_z)

            gru_n = gru_r * (jnp.einsum('ij, bi -> bj', layer['n_self'], h[i]) + layer['n_self_b']) + jnp.einsum('ij, bi -> bj', layer['n_above'], h_above) + \
                    jnp.einsum('ij, bi -> bj', layer['n_z'], z) + layer['n_z_b']

            gru_n = nn.tanh(gru_n)

            out_h.append((1 - gru_z) * gru_n + gru_z * h[i])

        elif config['layers'][i]['type'] == 'PVRNN':

            tau = config['layers'][i]['tau']

            dh = jnp.einsum('ij, bi -> bj', layer['r_self'], h[i]) + jnp.einsum('ij, bi -> bj', layer['z_self'], z) + \
                 jnp.einsum('ij, bi -> bj', layer['u_self'], h_above) + layer['r_self_b']

            dh = nn.tanh(dh)

            out_h.append((1 - 1. / tau) * h[i] + dh / tau)

        h_above = out_h[-1]

    return (out_h, (out_h, mu_ps, sigma_ps, mu_qs, sigma_qs))

def init_latent_inputs(latents: [dict], config: JsonDict, key, indices: jnp.array, T: int, batch_size: int):
    latent_inputs = []
    for i in range(len(config['layers'])):
        key, eps_key = random.split(key)
        mus = latents[i]['mus'][:, indices]
        log_sigmas = latents[i]['log_sigmas'][:, indices]
        eps = random.normal(eps_key, (T, batch_size, latents[i]['mus'].shape[2]))
        latent_inputs.append({'mus': mus, 'log_sigmas': log_sigmas, 'eps': eps})

    return latent_inputs



@partial(jit, static_argnames=['config', 'T', 'batch_size'])
def forward_posterior(params: dict, latents: [dict], config: JsonDict, key, indices: jnp.array,  T: int, batch_size: int):
    latent_inputs = init_latent_inputs(latents, config, key, indices, T, batch_size)

    h = []
    for layer in config['layers']:
        h.append(jnp.zeros((batch_size, layer['N'])))

    _, (hs, mu_ps, sigma_ps, mu_qs, sigma_qs) = lax.scan(partial(to_scan,
                                                posterior=True,
                                                params=params,
                                                config=config,
                                                batch_size=batch_size),
                                        h, latent_inputs, length=T)

    @vmap
    def motor_out(h_above):
        out = jnp.einsum("ij, bi -> bj", params['weights']['w_output'], h_above)
        return out

    im_h, im_w, im_c = config["encoded_image_shape"]
    @vmap
    def image_out(h_above):
        encoded_im_flat = jnp.einsum("ij, bi -> bj", params['weights']['w_image'], h_above) + params['weights']['b_image']
        encoded_im = encoded_im_flat.reshape((-1, im_h, im_w, im_c))

        for i in range(len(config['deconvolution_layers'])):
            deconv_kernel = params['weights']['deconv_layers'][i]
            encoded_im_tanh = jnp.tanh(encoded_im)
            encoded_im = lax.conv_transpose(encoded_im_tanh, deconv_kernel, (2, 2), 'VALID')

        img_out = nn.sigmoid(encoded_im)

        return img_out

    outputs = motor_out(hs[-1])
    img_outputs = image_out(hs[-1])

    return outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, hs


@partial(jit, static_argnames=['config', 'T', 'batch_size'])
def forward_prior(params: dict, latents: [dict], config: JsonDict, key, indices: jnp.array,  T: int, batch_size: int):
    latent_inputs = init_latent_inputs(latents, config, key, indices, T, batch_size)

    h = []
    for layer in config['layers']:
        h.append(jnp.zeros((batch_size, layer['N'])))

    _, (hs, mu_ps, sigma_ps, mu_qs, sigma_qs) = lax.scan(partial(to_scan,
                                                    posterior=False,
                                                    params=params,
                                                    config=config,
                                                    batch_size=batch_size),
                                            h, latent_inputs, length=T)

    @vmap
    def motor_out(h_above):
        out = jnp.einsum("ij, bi -> bj", params['weights']['w_output'], h_above)
        return out

    im_h, im_w, im_c = config["encoded_image_shape"]
    @vmap
    def image_out(h_above):
        encoded_im_flat = jnp.einsum("ij, bi -> bj", params['weights']['w_image'], h_above) + params['weights']['b_image']
        encoded_im = encoded_im_flat.reshape((-1, im_h, im_w, im_c))

        for i in range(len(config['deconvolution_layers'])):
            deconv_kernel = params['weights']['deconv_layers'][i]
            encoded_im_tanh = jnp.tanh(encoded_im)
            encoded_im = lax.conv_transpose(encoded_im_tanh, deconv_kernel, (2, 2), 'VALID')

        img_out = nn.sigmoid(encoded_im)

        return img_out

    outputs = motor_out(hs[-1])
    img_outputs = image_out(hs[-1])

    return outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, hs


def loss_posterior(params: dict, latents: [[dict]], config: JsonDict, key, indices: jnp.array, targets: jnp.array, im_targets: jnp.array):

    T, bs = targets.shape[0], targets.shape[1]
    outputs, img_outputs, mu_ps, sigma_ps, mu_qs, sigma_qs, _ = forward_posterior(params, latents, config, key, indices, T, bs)

    def Kl_Div(mu_p, sigma_p, mu_q, sigma_q):
        return (jnp.log((sigma_p / sigma_q) + 1e-20) +
               0.5 * ((mu_p - mu_q)**2 + sigma_q**2) / sigma_p**2 - 0.5).sum() / (T * bs)

    def Kl_Div_corr(mu_p, sigma_p, mu_q, sigma_q):
        return (jnp.log(((sigma_p**2 + 1.) / (sigma_q**2 + 1.)) + 1e-20) +
                ((mu_p - mu_q)**2 + sigma_q**2 + 1.) / (sigma_p**2 + 1.) - 1.).sum() / (T * bs)

    def Wasserstein(mu_p, sigma_p, mu_q, sigma_q):
        return (((mu_p - mu_q)**2).sum() + (sigma_p**2 + sigma_q**2 - 2 * jnp.sqrt(sigma_p**2 * sigma_q**2)).sum()) / (T * bs)

    #KL = ((-jnp.log(sigma) + (sigma ** 2 + (mu ** 2)) / 2. - 0.5).sum(axis=-1)).mean()

    mse = ((targets - outputs)**2).mean()

    if config['probability_distance'] == 'Wasserstein':
        print('Distance is Wasserstein!')
        probability_distance = Wasserstein
    elif config['probability_distance'] == 'KL_Corrected':
        print('Distance is KL Corrected!')
        probability_distance = Kl_Div_corr
    else:
        probability_distance = Kl_Div

    KL = 0
    for i in range(len(config['layers'])):
        KL += probability_distance(mu_ps[i], sigma_ps[i], mu_qs[i], sigma_qs[i])

    #vision_error = jnp.log(jnp.exp((img_outputs - im_targets)**2).mean(axis=(2,3,4))).mean()

    output_mu = img_outputs.mean(axis=(2, 3, 4))
    target_mu = im_targets.mean(axis=(2, 3, 4))
    output_var = img_outputs.var(axis=(2, 3, 4))
    target_var = im_targets.var(axis=(2, 3, 4))

    cov = ((img_outputs - output_mu[:, :, None, None, None]) * (im_targets - target_mu[:, :, None, None, None])).mean(
        axis=(2, 3, 4))

    ssim = (2 * output_mu * target_mu + 0.0001) * (2 * cov + 0.0003) / (
                (output_mu ** 2 + target_mu ** 2 + 0.0001) * (output_var + target_var + 0.0003))
    dssim = (1 - ssim) / 2
    vision_error = dssim.mean()

    dual_motor = params['duals']['dual_motor_loss']
    dual_image = params['duals']['dual_image_loss']

    loss = (mse - config['loss']['mse_eps']) * dual_motor[0] + \
           (vision_error - config['loss']['ssim_eps']) * dual_image[0] + \
           KL

    print('Traced!')

    return loss, (mse, vision_error, KL, outputs, img_outputs)


if __name__ == "__main__":
    network_cfg = {
        'layers': [{'tau': 2., 'N': 36},
                   {'tau': 4., 'N': 12}],

        'deconvolution_layers': [
            {'kernel': (17, 21, 12, 12)},
            {'kernel': (13, 17, 12, 8)},
            {'kernel': (5, 9, 8, 6)},
            {'kernel': (5, 5, 6, 3)}
        ],

        'encoded_image_shape': (12, 16, 12),

        'image_resolution': (48, 64),
        'output_dim': 12,

        'latent_space_dim': 40,

        'loss': {"mse_eps": 0.005,
                 "w": 0.001
                 },

        'log_eps': 1e-10
    }

    key = random.PRNGKey(0)

    dataset = np.array(random.normal(key, (50, 400, 12)))

    key, key_init = random.split(key)
    pvrnn = PVRNN(network_cfg, key_init)

    pvrnn.init_Z(dataset)
    indices = jnp.array([0,1,2])

    loss = loss_indices(pvrnn.params_tree, pvrnn.config, key, indices, dataset)

    to_jit = lambda params, key, indices: grad(loss_indices, 0)(params, pvrnn.config, key, indices, dataset)

    g = jit(to_jit)(pvrnn.params_tree, key, indices)

    print(g)
