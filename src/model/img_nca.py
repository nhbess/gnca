from typing import Callable, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.lax as lax
import equinox as eqx
import equinox.nn as nn
from jaxtyping import Array, Int, Float


TARGET_CLASS = Int[Array, "1"]
IMAGE = Float[Array, "4 H W"]
GEN_STEPS = Float[Array, "G S H W"]  # where G is number of generation steps and S is state size


class State(NamedTuple):
    iter: int
    cell_states: jax.Array
    rng_key: jr.PRNGKeyArray


class ImageNCA(eqx.Module):
    """
    Neural Cellular Automata for image generation based on Mordvintsev et al., (2021).

    It grows images by using cellular automatas describing the color channels in each cell with
    the update rules being instantiated using parameterized neural networks. To generate different
    images, the target class can be fed as input to the model as in Sudhakaran et al., (2022).

    Args:
        img_size: size of the target image.
        filter: module that determines how information is shared amongst units.
        target_encoderr: used to produce the encoding of the target class.
        update_rule: function to perform state updates.
        update_prob: probability of performing an an update.
        alive_threshold: value of alive value beyond which a unit is considered part of the target
                         image or left empty.
        generation_steps: the number of steps used for generation. If a range, a value will be
                          sampled within its limits.

    """
    img_size: Tuple[int, int]
    state_size: int
    filter: Callable
    target_encoder: Callable
    update_rule: Callable
    max_pool: eqx.nn.MaxPool2d
    generation_steps: Tuple[int, int] = (45, 96)
    update_prob: float = 0.5
    alive_threshold: float = 0.1
    training: bool = True

    def __init__(
        self,
        img_size,
        state_size,
        filter,
        target_encoder,
        update_rule,
        update_prob,
        alive_threshold,
        generation_steps
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = img_size, img_size

        if isinstance(generation_steps, int):
            generation_steps = generation_steps, generation_steps

        self.img_size = img_size
        self.state_size = state_size + 3 + 1
        self.filter = filter
        self.target_encoder = target_encoder
        self.update_rule = update_rule
        self.update_prob = update_prob
        self.alive_threshold = alive_threshold
        self.generation_steps = generation_steps
        # self._sample_steps = isinstance(self.generation_steps, tuple)
        self.max_pool = nn.MaxPool2d(filter.kernel_size, stride=1, padding=1)
        self.training = True

    def __call__(
        self,
        inputs: TARGET_CLASS,
        rng_key: jr.PRNGKeyArray
    ) -> Tuple[IMAGE, GEN_STEPS, State]:
        # TODO: inputs are not used as this would only be relevant for the goal-directed version
        steps_key, state_key = jr.split(rng_key, 2)

        g_steps = self.sample_generation_steps(steps_key)
        init_state = self.init_state(state_key)
        target_emb = self.target_encoder(inputs)[..., jnp.newaxis, jnp.newaxis]  # tiling

        def f(carry: State, _):
            iter, cell_states, key = carry
            c_key, s_key = jr.split(key)

            # only update the states if we have not surpassed the generation_steps
            cell_states = lax.cond(
                iter < g_steps,
                self.update_cell_states,
                lambda cg, *_: cg,
                cell_states, target_emb, s_key
            )

            return State(iter + 1, cell_states, c_key), cell_states[:4]

        final_state, outputs = lax.scan(f, init_state, None, self.generation_steps[1])

        # image at target generation step, full output history and final state
        return outputs[g_steps], outputs, final_state

    def sample_generation_steps(self, key: jr.PRNGKeyArray):
        return lax.cond(
            self.training,
            lambda k: jr.randint(k, (1,), *self.generation_steps).squeeze(),  # reduce to a scalar
            lambda _: self.generation_steps[1],
            key
        )

    # def get_target_embedding(self, inputs):
    #     mask = jnp.repeat(jnp.asarray([0, 1]), jnp.asarray([4, self.state_size - 4]))
    #     emb = self.target_encoder(inputs) * mask
    #     return emb[..., jnp.newaxis, jnp.newaxis]

    def init_state(self, key) -> State:
        H, W = self.img_size
        cell_states = jnp.zeros((self.state_size, *self.img_size)).at[3:, H // 2, W // 2].set(1.0)
        return State(0, cell_states, key)

    def update_cell_states(self, cell_states, target_emb, s_key):
        # NOTE: This pre alive mask is not described in the article. I imagine it helps with
        # stability. Notice that because the mask is computed using pooling this does not
        # prevent the NCA from growing correctly.
        pre_alive_mask =  self.alive_mask(cell_states)

        perception_vector = self.perceieve(cell_states + target_emb * pre_alive_mask)
        updates = self.update_rule(perception_vector)
        new_states = cell_states + updates * self.stochastic_update_mask(s_key)

        alive_mask = (self.alive_mask(new_states) & pre_alive_mask).astype(jnp.float32)

        return new_states * alive_mask

    def perceieve(self, cell_states):
        filter_grid = self.filter(cell_states)
        return jnp.concatenate([cell_states, filter_grid], axis=0)

    def stochastic_update_mask(self, key: jr.PRNGKeyArray):
        return jr.bernoulli(key, self.update_prob, self.img_size)[jnp.newaxis].astype(jnp.float32)

    def alive_mask(self, cell_states):
        # Take the alpha channel as the measure of how alive a cell is.
        return self.max_pool(cell_states[3:4]) > self.alive_threshold

    def train(self, mode=True):
        return eqx.tree_at(lambda x: x.training, self, mode)

    def eval(self):
        return self.train(False)
