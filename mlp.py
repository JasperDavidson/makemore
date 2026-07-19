import torch
from helpers import find_mappings

block_size = 3
embedding_dims = 2
l1_size = 100
l_out_size = 27  # Matches number of character options


def forward_pass(embeddings: torch.Tensor) -> torch.Tensor:
    # Run the MLP
    # Layer One
    w1 = torch.randn(block_size * embedding_dims, l1_size)
    b1 = torch.randn(l1_size)

    # Perform layer one calculation
    # - Note that we resize to (-1, 6) in order to pass all block_size * embedding_dims vals to the first layer
    # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
    l1_out = torch.tanh(embeddings.view(-1, 6) @ w1 + b1)

    # Output layer
    w_out = torch.randn(l1_size, l_out_size)
    b_out = torch.randn(l_out_size)

    # Compute the output and apply softmax
    mlp_out = l1_out @ w_out + b_out
    mlp_exp = mlp_out.exp()
    prob = mlp_exp / (mlp_exp.sum(1, keepdim=True))

    return prob


def train_mlp():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    itos, stoi = find_mappings(chars, ".")

    # Build training and validation datasets
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
        27, embedding_dims
    )  # Randomly initialize the embeddings, learned via gradient optimization
    cur_embeddings = embedding_space[training]

    forward_pass(cur_embeddings)
