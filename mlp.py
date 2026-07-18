import torch
from helpers import find_mappings


def train_mlp():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    itos, stoi = find_mappings(chars, ".")

    # Build training and validation datasets
    block_size = 3
    training, validation = [], []

    for word in words[:5]:
        context = [0] * block_size

        for ch in word + ".":
            ch_val = stoi[ch]
            training.append(context)
            validation.append(ch_val)
            context = context[1:] + [ch_val]

    training = torch.tensor(training)
    validation = torch.tensor(validation)
    embedding_space = torch.randn(
        27, 2
    )  # Randomly initialize the embeddings, learned via gradient optimization
    cur_embeddings = embedding_space[training]
