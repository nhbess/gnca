import os
import sys

wspath = 'C:/Users/nbes/Documents/GitHub/gnca'
#change this to the path of the gnca folder on your computer
sys.path.append(wspath)

import os.path as osp
import time
from argparse import ArgumentParser
from typing import Callable

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import equinox.nn as nn
import optax
import matplotlib.pyplot as plt
from jax.nn import relu
from jax import config
from jaxtyping import Array, Float, Int, PyTree
from matplotlib.animation import FuncAnimation, PillowWriter

from src.dataset.emojis import SingleEmojiDataset
from src.dataset.dataloader import JaxLoader
from src.model.img_nca import ImageNCA
from src.nn.sobel import SobelFilter


def seed_everything(seed=None):
    np_rng = np.random.default_rng(seed)
    jax_key = jax.random.PRNGKey(np_rng.integers(0, 2 ** 32 - 1))
    return np_rng, jax_key


def create_model(img_size: int, hidden_state_size: int, key: jr.PRNGKeyArray):
    key1, key2 = jax.random.split(key, 2)

    state_size = hidden_state_size + 3 + 1

    filter = SobelFilter()
    goal_encoder = None
    update_rule = nn.Sequential([
        nn.Conv2d(state_size + 2 * state_size, 128, kernel_size=1, key=key1),
        nn.Lambda(relu),
        nn.Conv2d(128, state_size, kernel_size=1, key=key2),
    ])

    return ImageNCA(
        (img_size, img_size),
        hidden_state_size,
        filter,
        goal_encoder,  # currently not being used
        update_rule,
        update_prob=0.5,
        alive_threshold=0.1,
        generation_steps=(64, 96),
    )


def create_dataest(target_size: int, img_pad: int, batch_size: int):
    dataset = SingleEmojiDataset("salamander", target_size, img_pad, batch_size)

    # emoji = dataset.get_emoji()[1][0]
    # plt.imshow(np.transpose(emoji, (1, 2, 0)))
    # plt.show()

    return JaxLoader(dataset, None, collate_fn=lambda x: x)


def infinite_trainloader(loader: JaxLoader):
    while True:
        yield from loader


def compute_loss(model: Callable, loss: Callable, x: Array, y: Array, key: jr.PRNGKeyArray):
    batch_key = jr.split(key, x.shape[0])
    preds, _, _ = jax.vmap(model)(x, batch_key)
    return jnp.sum(loss(preds, y)) / len(y)


def model_params(model: eqx.Module):
    return eqx.partition(model, eqx.is_array)[0]


def train(
    model: ImageNCA,
    train_loader: JaxLoader,
    testloader: JaxLoader,
    lr: float,
    grad_accum: int,
    train_iters: int,
    eval_iters: int,
    eval_freq: int,
    print_every: int,
    rng: np.random.Generator,
) -> ImageNCA:

    loss = optax.l2_loss
    # adam = optax.adam(optax.piecewise_constant_schedule(lr, {2000: 0.1}))
    adam = optax.adam(lr)

    optim = optax.chain(
        optax.clip_by_global_norm(1.0),
        adam,
    )

    if grad_accum > 1:
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum)

    jax_key = jr.PRNGKey(rng.integers(0, 2 ** 32 -1))
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def make_step(
        model: ImageNCA,
        key: jr.PRNGKeyArray,
        opt_state: PyTree,
        x: Float[Array, "B 1"],
        y: Int[Array, "B 4 H W"],
    ):
        loss_value, grads = eqx.filter_value_and_grad(compute_loss)(model, loss, x, y, key)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value

    best_loss, best_model = np.inf, model_params(model)

    for step, (x, y) in zip(range(train_iters), infinite_trainloader(train_loader)):
        jax_key, step_key = jr.split(jax_key, 2)

        model = model.train()
        model, opt_state, train_loss = make_step(model, step_key, opt_state, x, y)

        if (step % eval_freq) == 0:
            eval_loss = evaluate(model, testloader, loss, eval_iters, rng)
            if eval_loss < best_loss:
                best_model = model_params(model)
        else:
            eval_loss = None

        if (step % print_every) == 0:
            print_str = f"{step=}, train_loss={train_loss}"

            if eval_loss is not None:
                print_str = print_str + f", eval_loss={eval_loss}"

            print(print_str)


    return best_model


def evaluate(model: ImageNCA, loader: JaxLoader, loss: Callable, iters: int, rng: np.random.Generator):
    model = model.eval()
    total_loss, total_examples = 0.0, 0

    jax_key = jr.PRNGKey(rng.integers(0, 2 ** 32 - 1))

    for _, (x, y) in zip(range(iters), infinite_trainloader(loader)):
        jax_key, step_key = jr.split(jax_key, 2)
        total_loss += compute_loss(model, loss, x, y, step_key) * len(y)
        total_examples += len(y)

    return total_loss / total_examples


def save_model(model: eqx.Module, save_folder: str):
    save_file = osp.join(save_folder, "best_model.eqx")
    eqx.tree_serialise_leaves(save_file, model)


def load_model(model: eqx.Module, save_folder: str):
    # def deserialise_filter_spec(f, x):
    #     if isinstance(x, jax.dtypes.bfloat16):
    #         return jax.dtypes.bfloat16(jnp.load(f).item())
    #     else:
    #         return eqx.default_deserialise_filter_spec(f, x)

    save_file = osp.join(save_folder, "best_model.eqx")
    return eqx.tree_deserialise_leaves(save_file, model)


def to_viz(frames, scale=2):
    frames = np.transpose(np.asarray(frames), (0, 2, 3, 1))
    frames = np.repeat(frames, scale, 1)
    frames = np.repeat(frames, scale, 2)
    return frames


def generate_growth_gif(
    model: ImageNCA,
    loader: JaxLoader,
    save_folder: str,
    rng: np.random.Generator
):
    key = jr.PRNGKey(rng.integers(0, 2 ** 32 - 1))
    instance = next(iter(loader))[0][0]

    _, gen_steps, _ = model.eval()(instance, key)

    def to_alpha(x):
        return jnp.clip(x[3:4], 0.0, 1.0)

    def to_rgb(x):
        # assume rgb premultiplied by alpha
        rgb, a = x[:3], to_alpha(x)
        return jnp.clip(1.0 - a + rgb, 0.0, 1.0)

    frames = jax.device_put(jax.vmap(to_rgb)(gen_steps), jax.devices("cpu")[0])
    frames = to_viz(frames)

    fig = plt.figure()
    ax = plt.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    im = plt.imshow(frames[0], vmin=0, vmax=1)
    def animate(i):
        ax.set_title(f"Growth step: {i}")
        im.set_array(frames[i])
        return im,

    ani = FuncAnimation(fig, animate, interval=200, blit=True, repeat=True, frames=len(frames))
    ani.save(osp.join(save_folder, "salamander-growth.gif"), dpi=150, writer=PillowWriter(fps=8))


def main(
    train_iters = 10000,
    eval_iters = 1,
    eval_freq = 100,
    batch_size = 8,
    lr = 2e-3,
    grad_accum = 1,
    img_size = 40,
    img_pad = 16,
    print_every = 1,
    save_folder = None,
    seed = None,
    debug = False,
):
    np_rng, key = seed_everything(seed)

    if save_folder is None:
        save_folder = osp.join("data", "logs", "salamander", f"{time.strftime('%Y-%m-%d_%H-%M')}")

    loader = create_dataest(img_size, img_pad, batch_size)
    model = create_model(img_size + 2 * img_pad, 12, key)

    if osp.exists(save_folder) and not debug:
        trained_model = load_model(model, save_folder)

    else:
        os.makedirs(save_folder, exist_ok=debug)

        best_params = train(
            model, loader, loader, lr, grad_accum, train_iters, eval_iters, eval_freq, print_every, np_rng
        )

        trained_model = eqx.combine(best_params, model)
        save_model(trained_model, save_folder)


    eval_loss = evaluate(trained_model, loader, optax.l2_loss, 14, np_rng)
    print(f"Trained model loss: {eval_loss}")

    generate_growth_gif(trained_model, loader, save_folder, np_rng)


if __name__ == "__main__":
    parser = ArgumentParser("salamander-gen")

    parser.add_argument("--train_iters", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--eval_iters", type=int, default=10)
    parser.add_argument("--eval_freq", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=40)
    parser.add_argument("--img_pad", type=int, default=16)
    parser.add_argument("--save_folder", type=str, default=None)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--disable_jit", action="store_true", default=False)

    args = parser.parse_args()

    if args.debug:
        args.train_iters = 1
        args.eval_iters = 1
        args.eval_freq = 1
        args.print_every = 1
        args.save_folder = osp.join("data", "logs", "salamander", "debug")

    disable_jit = args.__dict__.pop("diseble_jit")
    if disable_jit:
        config.update('jax_disable_jit', True)

    main(**vars(args))
