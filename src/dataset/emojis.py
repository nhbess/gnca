import io
import requests
from PIL import Image
from enum import Enum
from typing import Union

import numpy as np
from torch.utils.data import IterableDataset


class Emoji(Enum):
    SALAMANDER = "🦎"


class SingleEmojiDataset(IterableDataset):
    def __init__(
        self,
        emoji: Union[str, Emoji],
        target_size: int = 40,
        pad: int = 16,
        batch_size: int = 64
    ) -> None:
        super().__init__()
        if isinstance(emoji, str):
            emoji = Emoji[emoji.upper()]

        emoji_image = load_emoji(emoji.value, target_size)

        self.emoji = np.pad(emoji_image, ((pad, pad), (pad, pad), (0, 0)), "constant")
        self.emoji_name = emoji.name
        self.target_size = target_size
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            yield self.get_emoji()

    def get_emoji(self):
        # Only one emoji, so input is fixed
        inputs = np.repeat(np.expand_dims(np.array([0]), axis=0), self.batch_size, axis=0)
        targets = np.repeat(np.expand_dims(self.emoji, axis=0), self.batch_size, axis=0)
        return inputs, np.transpose(targets, [0, 3, 1, 2])  # NCHW


# Code from https://colab.research.google.com/github/google-research/self-organising-systems

def load_emoji(emoji, max_size):
    code = hex(ord(emoji))[2:].lower()
    url = f"https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u{code}.png?raw=true"
    return load_image(url, max_size)


def load_image(url, max_size=40):
    r = requests.get(url)
    img = Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    img = np.asarray(img, dtype=np.float32) / 255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img


def make_circle_masks(n, h, w, r=None):
    x = np.linspace(-1.0, 1.0, w)[None, None, :]
    y = np.linspace(-1.0, 1.0, h)[None, :, None]
    center = np.random.uniform(-0.5, 0.5, size=[2, n, 1, 1])
    if r is None:
        r = np.random.uniform(0.1, 0.4, size=[n, 1, 1])
    x, y = (x - center[0]) / r, (y - center[1]) / r
    mask = x * x + y * y < 1.0
    return mask.astype(float)
