import jax.numpy as jnp
from jax import random, jit, value_and_grad
import jax
from misc.tools import JsonDict, IndexedAdam
import network.pvrnn as pvrnn
import time
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import optax

from functools import partial



def train(dir, proj=False, prob_distance='KL'):

#dir = 'train_dir/train_random_vision'

    Path(dir).mkdir(exist_ok=True)

    #jax.config.update('jax_platform_name', 'cpu')
    #jax.default_device(jax.devices('gpu')[1])

    config = JsonDict('config/config.json')

    key = random.PRNGKey(0)

    #hf = h5py.File('dataset/dataset_long.h5', 'r')

    motor_pointer = jnp.load('dataset/motor_train.npy').transpose((1, 0, 2))#jnp.array(hf.get('motor'))
    batch_size = min(30, motor_pointer.shape[1])

    data_var = jnp.var(motor_pointer)
    config['loss']['mse_eps'] = data_var * 0.01
    config['probability_distance'] = prob_distance

    print(f"target mse loss = {config['loss']['mse_eps']}")

    print(motor_pointer.device_buffer.device())

    perception_pointer = jnp.load('dataset/vision_train.npy')[:,:,:,:,None].transpose((1, 0, 2, 3, 4))#jnp.array(hf.get('perception'))

    dataset_size = motor_pointer.shape[1]

    key, key_init = random.split(key)
    network = pvrnn.PVRNN(config, key_init)

    key, key_proj = random.split(key)
    network.init_Z(motor_pointer, perception_pointer, key_proj, projection=proj)

    #network.init_Z_with(mu_ps, sigma_ps)

    #pvrnn_grad = pvrnn.get_value_and_grad(config)



    pvrnn_grad = value_and_grad(pvrnn.loss_posterior, (0, 1), has_aux=True)
    opt = optax.multi_transform(
        {'weights': optax.adam(1e-3),
        # 'latents': optax.sgd(10.),
        # 'latents': optax.set_to_zero(),
        'duals': optax.chain(optax.adam(-1.), optax.keep_params_nonnegative())},

        network.opt_labels
    )

    opt_latent = IndexedAdam(1e-3)


    @partial(jit, static_argnames=['config'])
    def update(params, latents, config, opt_state, opt_latent_state, indices, targets, im_targets, key):
        key, key_g = random.split(key)

        (loss, (mse, vision_error, KL, outputs, img_outputs)), (grads, latent_grads) = \
            pvrnn_grad(params, latents, config, key_g, indices, targets, im_targets)

        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        updates, opt_latent_state = opt_latent.update(latent_grads, opt_latent_state, indices)
        latents = opt_latent.apply_updates(latents, updates, indices)
        return params, latents, opt_state, opt_latent_state, loss, mse, vision_error, KL, outputs, img_outputs


    opt_state = opt.init(network.params_tree)
    opt_latent_state = opt_latent.init(network.latent_vars)

    mses = []
    dssims = []
    KLs = []
    duals_motor = []
    duals_img = []
    total_loss = []

    for epoch in range(20000):

        start_time = time.time()
        print(f'Epoch {epoch}')

        key, key_ind = random.split(key)
        indices = np.array(random.choice(key_ind, dataset_size, (batch_size,), replace=False).sort())

        targets = motor_pointer[:, indices]
        im_targets = perception_pointer[:, indices]

        key, key_upd = random.split(key)
        start_upd_time = time.time()

        #with jax.profiler.trace(log_dir='/home/vsevolod/tmp/jax-trace'):

        network.params_tree, \
        network.latent_vars, \
        opt_state, \
        opt_latent_state, \
        loss, \
        mse, \
        vision_error, \
        KL, \
        outputs, \
        img_outputs \
            = update(network.params_tree,
                    network.latent_vars,
                    config,
                    opt_state,
                    opt_latent_state,
                    indices,
                    targets,
                    im_targets,
                    key_upd)


        loss.block_until_ready()
        img_outputs.block_until_ready()

        mses.append(mse)
        KLs.append(KL)
        dssims.append(vision_error)
        duals_motor.append(network.params_tree["duals"]["dual_motor_loss"][0])
        duals_img.append(network.params_tree["duals"]["dual_image_loss"][0])
        total_loss.append(loss)

        cur_time = time.time()

        print(f'Loss: {loss:.5f}, MSE: {mse:.5f}, DSSIM: {vision_error:.5f}, KL: {KL:.5f}')
        print(f'Dual Motor Var: {network.params_tree["duals"]["dual_motor_loss"][0]}')
        print(f'Dual Image Var: {network.params_tree["duals"]["dual_image_loss"][0]}')
        print(f'Total time: {cur_time - start_time:.4f}, Update time: {cur_time - start_upd_time}')

        if epoch % 1000 == 0:
            im = Image.fromarray((np.array(im_targets[15, 0, :, :, 0]) * 255).astype(np.uint8))
            im.save("images/true_perception.png")

            im = Image.fromarray((np.array(img_outputs[15, 0, :, :, 0]) * 255).astype(np.uint8))
            im.save("images/predicted_perception.png")

            network.save_state(f'{dir}', epoch=epoch)

    np.savetxt(f'{dir}/mse.txt', mses)
    np.savetxt(f'{dir}/dssim.txt', dssims)
    np.savetxt(f'{dir}/KL.txt', KLs)
    np.savetxt(f'{dir}/dual_motor.txt', duals_motor)
    np.savetxt(f'{dir}/dual_img.txt', duals_img)
    np.savetxt(f'{dir}/total_loss.txt', total_loss)

    network.save_state(dir)

if __name__ == "__main__":
    #train('train_dir/train_random_vision', proj=False)
    train('train_dir_PVRNN_abs/train_vision_proj_KL_Corrected', proj=True, prob_distance='KL_Corrected')
