#!/bin/bash

# train a model for each emoji
python -m scripts.emojis --target bang
python -m scripts.emojis --target butterfly
python -m scripts.emojis --target eye
python -m scripts.emojis --target fish
python -m scripts.emojis --target ladybug
python -m scripts.emojis --target pretzel
python -m scripts.emojis --target salamander
python -m scripts.emojis --target smiley
python -m scripts.emojis --target tree
python -m scripts.emojis --target web


# train model on all of them simultanously
python -m scripts.emojis --target all --train_iters 20000 --lr 0.001 --batch_size 16
