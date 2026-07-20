import torch
import torch.nn as nn
from helpers import find_mappings, create_data_splits
import random

# Hyperparameters
block_size = 3
embedding_dims = 10
num_training_iter = 100000
reg_loss_param = 0.01
minibatch_size = 100
momentum = 0.1
bn_epsilon = 1e-5

l1_size = 300
l_out_size = 27  # Matches number of character options


class BatchNorm(nn.Module):
    def __init__(self, fan_out: int, momentum: float, epsilon: float):
        super().__init__()

        self.gamma = nn.Parameter(torch.randn(1, fan_out))
        self.beta = nn.Parameter(torch.zeros(1, fan_out))
        self.running_mean = nn.Buffer(torch.zeros(1, fan_out))
        self.running_var = nn.Buffer(torch.ones(1, fan_out))
        self.momentum = momentum
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_mean = x.mean(dim=0, keepdim=True) if self.training else self.running_mean
        cur_var = x.var(dim=0, keepdim=True) if self.training else self.running_var
        norm_preact = (
            self.gamma * ((x - cur_mean) / (cur_var + self.epsilon) ** 0.5) + self.beta
        )

        if self.training:
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * cur_mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * cur_var

        return norm_preact


class HiddenLinear(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(fan_out, fan_in))
        nn.init.kaiming_normal_(self.weights, mode="fan_in", nonlinearity="tanh")

        self.bn = BatchNorm(fan_out, momentum, bn_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weights.t()
        y_bn = self.bn(y)
        y_act = torch.tanh(y_bn)

        return y_act

    def reg_loss(self) -> torch.Tensor:
        return (self.weights**2).mean()


class OutLinear(nn.Module):
    def __init__(
        self,
        fan_in: int,
        fan_out: int,
    ):
        super().__init__()

        self.weights = nn.Parameter(torch.randn(fan_out, fan_in) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1, fan_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.t() + self.bias

    def reg_loss(self) -> torch.Tensor:
        return (self.weights**2).mean()


class MLP(nn.Module):
    def __init__(
        self,
        block_size: int,
        embedding_dims: int,
        l1_size: int,
        out_size: int,
        embedding_card: int,
    ):
        super().__init__()

        self.l1 = HiddenLinear(block_size * embedding_dims, l1_size)
        self.out = OutLinear(l1_size, out_size)
        self.embedding_space = nn.Embedding(embedding_card, embedding_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding_space(x)
        emb_flat = emb.view(emb.size(0), -1)

        h = self.l1(emb_flat)
        logits = self.out(h)

        return logits

    def reg_loss(self):
        return self.l1.reg_loss() + self.out.reg_loss()

    def backward(self, loss: torch.Tensor, lr: torch.Tensor):
        for p in self.parameters():
            p.grad = None
        loss.backward()

        with torch.no_grad():
            for p in self.parameters():
                assert p.grad is not None
                p -= lr * p.grad


def train_mlp(x: torch.Tensor, y: torch.Tensor, train_iter: int) -> MLP:
    model = MLP(block_size, embedding_dims, l1_size, l_out_size, 27)
    model.train()

    # Ad hoc pre-training lr optimization
    lre = torch.linspace(-3, 0, 1000)
    lrs = 10**lre
    lr_loss = []

    for i in range(1000):
        lr = lrs[i]
        loss = compute_minibatch_loss(minibatch_size, x, y, model)[0]
        lr_loss.append(loss.item())

        model.backward(loss, lr)

    min_loss_idx = lr_loss.index(min(lr_loss))
    optimized_lr = lrs[min_loss_idx]

    # Perform the training forward and backward passes
    for next_train_cap in range(train_iter):
        lr = optimized_lr / 100 if next_train_cap > train_iter * 0.75 else optimized_lr
        loss = compute_minibatch_loss(minibatch_size, x, y, model)[0]
        model.backward(loss, lr)

    return model


# Compute overall loss for validation/testing
@torch.no_grad()
def compute_overall_loss(
    x: torch.Tensor, y: torch.Tensor, model: MLP
) -> list[torch.Tensor]:
    logits = model(x)
    data_loss = torch.nn.functional.cross_entropy(logits, y)
    reg_loss = model.reg_loss()
    loss = data_loss + reg_loss_param * reg_loss

    return [loss, data_loss, reg_loss]


def compute_minibatch_loss(
    batch_size: int,
    x: torch.Tensor,
    y: torch.Tensor,
    model: MLP,
) -> list[torch.Tensor]:
    minibatch_idx = torch.randint(0, x.shape[0], (batch_size,))
    x_batch = x[minibatch_idx]
    y_batch = y[minibatch_idx]

    # Compute the output and loss
    logits = model(x_batch)
    data_loss = torch.nn.functional.cross_entropy(logits, y_batch)
    reg_loss = model.reg_loss()
    loss = data_loss + reg_loss_param * reg_loss

    return [loss, data_loss, reg_loss]


@torch.no_grad()
def infer(num_words: int, itos: dict[int, str], model: MLP) -> list[str]:
    predictions = [""] * num_words
    for i in range(num_words):
        context = torch.zeros(block_size, dtype=torch.long)
        cur_char = 0

        while True:
            predictions[i] += itos[cur_char]

            logits = model.forward(context.unsqueeze(0))
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

    model = train_mlp(x_train, y_train, num_training_iter)

    model.eval()
    val_loss, val_data_loss, val_reg_loss = compute_overall_loss(x_val, y_val, model)
    test_loss, test_data_loss, test_reg_loss = compute_overall_loss(
        x_test, y_test, model
    )

    print(f"\nVal loss = {val_loss.item()}")
    print(f"Val data loss = {val_data_loss.item()}")
    print(f"Val reg loss = {val_reg_loss.item()}")

    print(f"\nTest loss = {test_loss.item()}")
    print(f"Test data loss = {test_data_loss.item()}")
    print(f"Test reg loss = {test_reg_loss.item()}")

    predictions = infer(10, itos, model)
    print("\nPredictions:")
    for prediction in predictions:
        print(prediction)
