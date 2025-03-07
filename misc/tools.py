import json
from jax.tree_util import tree_map
import jax
import jax.numpy as jnp
from functools import partial


def random_split_like_tree(rng_key, target=None, is_leaf=None):
    treedef = jax.tree_util.tree_structure(target, is_leaf=is_leaf)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_util.tree_unflatten(treedef, keys)


class JsonDict(dict):
    def __init__(self, path):
        with open(path) as f:
            self.json_file = path
            super().__init__(json.load(f))

    def __hash__(self):
        return hash(self.json_file)

    def __eq__(self, other):
        return hash(self) == hash(other)


class IndexedAdam:
    def __init__(self, lr, b_1 = 0.9, b_2 = 0.999):
        self.lr = lr
        self.b_1 = b_1
        self.b_2 = b_2
        self.epsilon = 1e-8

    def init(self, params):
        m = tree_map(lambda p: jnp.zeros(p.shape), params)
        v = tree_map(lambda p: jnp.zeros(p.shape), params)

        return {'m': m, 'v': v}

    def _update_m(self, grads, old_m, indices):
        m = old_m.at[:, indices].multiply(self.b_1)
        m = m.at[:, indices].add((1 - self.b_1) * grads[:, indices])
        return m

    def _update_v(self, grads, old_v, indices):
        v = old_v.at[:, indices].multiply(self.b_2)
        v = v.at[:, indices].add((1 - self.b_2) * grads[:, indices]**2)
        return v

    def update(self, grads, opt_state, indices):
        m = tree_map(partial(self._update_m, indices=indices), grads, opt_state['m'])
        v = tree_map(partial(self._update_v, indices=indices), grads, opt_state['v'])

        updates = tree_map(lambda m, v: -self.lr * m[:, indices]/(jnp.sqrt(v[:, indices]) + self.epsilon), m, v)
        return updates, {'m': m, 'v': v}

    def apply_updates(self, params, updates, indices):
        return tree_map(
            lambda p, u: jnp.asarray(p.at[:, indices].add(u)).astype(jnp.asarray(p).dtype),
            params, updates)
