import torch
from helpers import find_mappings

# Hyperparameters
lr = 0.1
block_size = 3
embedding_dims = 2
num_training_iter = 10000
reg_loss_param = 0.1

l1_size = 100
l_out_size = 27  # Matches number of character options


def forward_pass(embeddings: torch.Tensor, validation: torch.Tensor, train_iter: int):
    # Layer One
    w1 = torch.randn(block_size * embedding_dims, l1_size, requires_grad=True)
    b1 = torch.randn(l1_size, requires_grad=True)

    # Output layer
    w_out = torch.randn(l1_size, l_out_size, requires_grad=True)
    b_out = torch.randn(l_out_size, requires_grad=True)

    for cur_iter in range(train_iter):
        # Perform layer one calculation
        # - Note that we resize to (-1, 6) in order to pass all block_size * embedding_dims vals to the first layer
        # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
        l1_out = torch.tanh(embeddings.view(-1, 6) @ w1 + b1)

        # Compute the output and apply softmax
        mlp_out = l1_out @ w_out + b_out
        mlp_exp = mlp_out.exp()
        prob = mlp_exp / (mlp_exp.sum(1, keepdim=True))

        data_loss = -prob[torch.arange(len(validation)), validation].log().mean()
        reg_loss = (w1**2).sum().mean() + (w_out**2).sum().mean()
        loss = data_loss + reg_loss_param * reg_loss
        print(f"Data loss @ {cur_iter} = {data_loss}")
        print(f"Reg loss @ {cur_iter} = {reg_loss}")
        print(f"Loss @ {cur_iter} = {loss}")
        w1.grad = None
        b1.grad = None
        w_out.grad = None
        b_out.grad = None

        loss.backward()
        assert w1.grad is not None
        assert b1.grad is not None
        assert w_out.grad is not None
        assert b_out.grad is not None

        w1.data += -lr * w1.grad
        b1.data += -lr * b1.grad
        w_out.data += -lr * w_out.grad
        b_out.data += -lr * b_out.grad


def train_mlp():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    itos, stoi = find_mappings(chars, ".")

    # Build training and validation datasets
    training, validation = [], []

    for word in words:
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

    forward_pass(cur_embeddings, validation, num_training_iter)
