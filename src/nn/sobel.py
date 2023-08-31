from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Tuple
from jaxtyping import Array


conv2d = partial(jax.scipy.signal.convolve2d, mode='same')


class SobelFilter(eqx.Module):
    kernel_size: Tuple[int, int]

    def __init__(self, kernel_size: int = 3):
        super().__init__()

        if kernel_size != 3:
            raise NotImplementedError

        self.kernel_size = kernel_size, kernel_size

    def __call__(self, inputs: Array):
        kernel_x = jnp.array([
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0]
        ]) / 8.0

        kernel_y = jnp.array([
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0]
        ]) / 8.0

        x_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, kernel_x)
        y_conv = jax.vmap(conv2d, in_axes=(0, None))(inputs, kernel_y)

        return jnp.concatenate([x_conv, y_conv], axis=0)
