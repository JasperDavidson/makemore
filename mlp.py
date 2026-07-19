import torch
from helpers import find_mappings

# Hyperparameters
lr = 0.1
block_size = 3
embedding_dims = 2
num_training_iter = 10000
reg_loss_param = 0.1
minibatch_size = 32

l1_size = 100
l_out_size = 27  # Matches number of character options


def forward_pass(training: torch.Tensor, validation: torch.Tensor, train_iter: int):
    embedding_space = torch.randn(
        27, embedding_dims
    )  # Randomly initialize the embeddings, learned via gradient optimization

    # Layer One
    w1 = torch.randn(block_size * embedding_dims, l1_size)
    b1 = torch.randn(l1_size)

    # Output layer
    w_out = torch.randn(l1_size, l_out_size)
    b_out = torch.randn(l_out_size)

    parameters = [embedding_space, w1, b1, w_out, b_out]
    for p in parameters:
        p.requires_grad = True

    data_loss = 0
    reg_loss = 0
    loss = 0

    def compute_overall_loss() -> list[torch.Tensor]:
        cur_embeddings = embedding_space[training]

        # Perform layer one calculation
        # - Note that we resize to (-1, 6) in order to pass all block_size * embedding_dims vals to the first layer
        # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
        l1_out = torch.tanh(cur_embeddings.view(-1, 6) @ w1 + b1)

        # Compute the output and loss
        logits = l1_out @ w_out + b_out
        data_loss = torch.nn.functional.cross_entropy(logits, validation)
        reg_loss = (w1**2).mean() + (w_out**2).mean()
        loss = data_loss + reg_loss_param * reg_loss

        return [loss, data_loss, reg_loss]

    for next_train_cap in range(train_iter):
        minibatch_idx = torch.randint(0, training.shape[0], (minibatch_size,))
        cur_embeddings = embedding_space[training[minibatch_idx]]

        # Perform layer one calculation
        # - Note that we resize to (-1, 6) in order to pass all block_size * embedding_dims vals to the first layer
        # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
        l1_out = torch.tanh(cur_embeddings.view(-1, 6) @ w1 + b1)

        # Compute the output and loss
        logits = l1_out @ w_out + b_out
        data_loss = torch.nn.functional.cross_entropy(logits, validation[minibatch_idx])
        reg_loss = (w1**2).mean() + (w_out**2).mean()
        loss = data_loss + reg_loss_param * reg_loss

        for p in parameters:
            p.grad = None
        loss.backward()

        for p in parameters:
            assert p.grad is not None
            p.data += -lr * p.grad

        if next_train_cap % (train_iter / 10) == 0:
            overall_loss = compute_overall_loss()
            print(f"Loss @ {next_train_cap} = {overall_loss[0].data}")
            print(f"Data loss @ {next_train_cap} = {overall_loss[1].data}")
            print(f"Reg loss @ {next_train_cap} = {overall_loss[2].data}")


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

    forward_pass(training, validation, num_training_iter)
