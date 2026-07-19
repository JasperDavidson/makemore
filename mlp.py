import torch
from helpers import find_mappings, create_data_splits
import random

# Hyperparameters
block_size = 3
embedding_dims = 10
num_training_iter = 100000
reg_loss_param = 0.01
minibatch_size = 100

l1_size = 300
l_out_size = 27  # Matches number of character options


def train_mlp(x: torch.Tensor, y: torch.Tensor, train_iter: int) -> list[torch.Tensor]:
    embedding_space = torch.randn(
        27, embedding_dims
    )  # Randomly initialize the embeddings, learned via gradient optimization

    # Layer One
    w1 = torch.empty(block_size * embedding_dims, l1_size)
    torch.nn.init.kaiming_normal_(w1, mode="fan_in", nonlinearity="tanh")
    b1 = torch.randn(l1_size) * 0.01

    # Output layer
    # Make weights smaller as to prevent initial overconfidence
    w_out = torch.randn(l1_size, l_out_size) * 0.01
    b_out = torch.zeros(l_out_size) * 0

    parameters = [embedding_space, w1, b1, w_out, b_out]
    for p in parameters:
        p.requires_grad = True

    # Ad hoc pre-training lr optimization
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre
    lr_loss = []

    for i in range(1000):
        lr = lrs[i]
        loss = compute_minibatch_loss(
            minibatch_size, x, y, embedding_space, w1, b1, w_out, b_out
        )[0]
        lr_loss.append(loss.item())

        for p in parameters:
            p.grad = None
        loss.backward()

        for p in parameters:
            assert p.grad is not None
            p.data += -lr * p.grad

    min_loss_idx = lr_loss.index(min(lr_loss))
    optimized_lr = lrs[min_loss_idx]

    # Perform the training forward and backward passes
    for next_train_cap in range(train_iter):
        lr = optimized_lr / 100 if next_train_cap > train_iter * 0.75 else optimized_lr
        loss = compute_minibatch_loss(
            minibatch_size, x, y, embedding_space, w1, b1, w_out, b_out
        )[0]

        for p in parameters:
            p.grad = None
        loss.backward()

        for p in parameters:
            assert p.grad is not None
            p.data += -lr * p.grad

    test_loss, test_data_loss, test_reg_loss = compute_overall_loss(
        x, y, embedding_space, w1, b1, w_out, b_out
    )
    print(f"Test loss = {test_loss.item()}")
    print(f"Test data loss = {test_data_loss.item()}")
    print(f"Test reg loss = {test_reg_loss.item()}")

    return [embedding_space, w1, b1, w_out, b_out]


# Compute overall loss for validation/testing
@torch.no_grad()
def compute_overall_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    embedding_space: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> list[torch.Tensor]:
    cur_embeddings = embedding_space[x]

    # Perform layer one calculation
    # - Note that we resize to (-1, block_size * embedding_dims) in order to pass all block_size * embedding_dims vals to the first layer
    # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
    l1_out = torch.tanh(cur_embeddings.view(-1, block_size * embedding_dims) @ w1 + b1)

    # Compute the output and loss
    logits = l1_out @ w_out + b_out
    data_loss = torch.nn.functional.cross_entropy(logits, y)
    reg_loss = (w1**2).mean() + (w_out**2).mean()
    loss = data_loss + reg_loss_param * reg_loss

    return [loss, data_loss, reg_loss]


def compute_minibatch_loss(
    batch_size: int,
    x: torch.Tensor,
    y: torch.Tensor,
    embedding_space: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> list[torch.Tensor]:
    minibatch_idx = torch.randint(0, x.shape[0], (batch_size,))
    cur_embeddings = embedding_space[x[minibatch_idx]]

    # Perform layer one calculation
    # - Note that we resize to (-1, block_size * embedding_dims) in order to pass all block_size * embedding_dims vals to the first layer
    # - tanh introduces non-linearity into the system, addressing the flaw of the simple 27-neuron layer by allowing the network to learn features
    l1_out = torch.tanh(cur_embeddings.view(-1, block_size * embedding_dims) @ w1 + b1)

    # Compute the output and loss
    logits = l1_out @ w_out + b_out
    data_loss = torch.nn.functional.cross_entropy(logits, y[minibatch_idx])
    reg_loss = (w1**2).mean() + (w_out**2).mean()
    loss = data_loss + reg_loss_param * reg_loss

    return [loss, data_loss, reg_loss]


@torch.no_grad()
def infer(
    num_words: int,
    itos: dict[int, str],
    embedding_space: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w_out: torch.Tensor,
    b_out: torch.Tensor,
) -> list[str]:
    predictions = [""] * num_words
    for i in range(num_words):
        context = torch.tensor([0] * block_size)
        cur_char = 0

        while True:
            predictions[i] += itos[cur_char]

            # Compute the forward pass and find the probabilities
            cur_embeddings = embedding_space[context]

            hidden_out = torch.tanh(
                cur_embeddings.view(-1, block_size * embedding_dims) @ w1 + b1
            )

            logits = hidden_out @ w_out + b_out
            prob = torch.softmax(logits, dim=1)

            # Sample to find the next char and update context
            cur_char = int(torch.multinomial(prob, 1).item())
            if cur_char == 0:
                break

            context = torch.cat((context[1:], torch.tensor([cur_char])))

        predictions[i] = predictions[i][1:]

    return predictions


def train_call():
    # Word dataset to train off of
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    itos, stoi = find_mappings(chars, ".")

    # Build training and validation datasets
    random.shuffle(words)
    train_split = int(len(words) * 0.8)
    val_split = int(len(words) * 0.9)
    test_split = len(words)

    x_train, y_train = create_data_splits(words[0:train_split], stoi, block_size)
    x_val, y_val = create_data_splits(words[0:val_split], stoi, block_size)
    x_test, y_test = create_data_splits(words[0:test_split], stoi, block_size)

    embedding_space, w1, b1, w_out, b_out = train_mlp(
        x_train, y_train, num_training_iter
    )
    val_loss, val_data_loss, val_reg_loss = compute_overall_loss(
        x_val, y_val, embedding_space, w1, b1, w_out, b_out
    )

    print(f"\nVal loss = {val_loss.item()}")
    print(f"Val data loss = {val_data_loss.item()}")
    print(f"Val reg loss = {val_reg_loss.item()}")

    predictions = infer(10, itos, embedding_space, w1, b1, w_out, b_out)
    print("\nPredictions:")
    for prediction in predictions:
        print(prediction)
