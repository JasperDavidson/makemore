from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import helpers

# --- Hyperparameters ---
block_size = 8
n_embd = 10
n_hidden = 200
vocab_size = 27
batch_size = 32
max_steps = 200000
bn_momentum = 0.1
bn_epsilon = 1e-5
seed = 42


# --- Layers ---
class Linear(nn.Module):
    def __init__(self, fan_in: int, fan_out: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(fan_in, fan_out) / fan_in**0.5)
        self.bias = nn.Parameter(torch.zeros(fan_out)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class BatchNorm1d(nn.Module):
    def __init__(self, dim: int, eps: float = bn_epsilon, momentum: float = bn_momentum):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(1, dim))
        self.beta = nn.Parameter(torch.zeros(1, dim))
        self.register_buffer("running_mean", torch.zeros(1, dim))
        self.register_buffer("running_var", torch.ones(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Part 3 BatchNorm: expects 2D (batch, features).
        # You'll revisit this when activations become 3D.
        if self.training:
            xmean = x.mean(dim=0, keepdim=True)
            xvar = x.var(dim=0, keepdim=True)
        else:
            xmean = self.running_mean
            xvar = self.running_var

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        out = self.gamma * xhat + self.beta

        if self.training:
            with torch.no_grad():
                self.running_mean = (
                    (1 - self.momentum) * self.running_mean + self.momentum * xmean
                )
                self.running_var = (
                    (1 - self.momentum) * self.running_var + self.momentum * xvar
                )

        return out


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, ix: torch.Tensor) -> torch.Tensor:
        return self.weight[ix]


class Flatten(nn.Module):
    """Flatten the full context: (B, T, C) -> (B, T*C)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)


class Sequential(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def build_mlp() -> Sequential:
    """Flat MLP: embed -> flatten all context -> hidden -> logits."""
    model = Sequential(
        [
            Embedding(vocab_size, n_embd),
            Flatten(),
            Linear(n_embd * block_size, n_hidden, bias=False),
            BatchNorm1d(n_hidden),
            Tanh(),
            Linear(n_hidden, vocab_size),
        ]
    )

    # Scale down last-layer weights so initial predictions aren't overconfident
    last = model.layers[-1]
    assert isinstance(last, Linear)
    with torch.no_grad():
        last.weight.mul_(0.1)

    return model


def train(model: Sequential, xtr: torch.Tensor, ytr: torch.Tensor) -> list[float]:
    model.train()
    parameters = list(model.parameters())
    lossi: list[float] = []

    for i in range(max_steps):
        ix = torch.randint(0, xtr.shape[0], (batch_size,))
        xb, yb = xtr[ix], ytr[ix]

        logits = model(xb)
        loss = F.cross_entropy(logits, yb)

        for p in parameters:
            p.grad = None
        loss.backward()

        lr = 0.1 if i < 150000 else 0.01
        with torch.no_grad():
            for p in parameters:
                assert p.grad is not None
                p -= lr * p.grad

        if i % 10000 == 0:
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        lossi.append(loss.log10().item())

    return lossi


@torch.no_grad()
def split_loss(
    model: Sequential,
    split: str,
    xtr: torch.Tensor,
    ytr: torch.Tensor,
    xdev: torch.Tensor,
    ydev: torch.Tensor,
    xte: torch.Tensor,
    yte: torch.Tensor,
) -> float:
    model.eval()
    x, y = {
        "train": (xtr, ytr),
        "val": (xdev, ydev),
        "test": (xte, yte),
    }[split]
    logits = model(x)
    loss = F.cross_entropy(logits, y)
    print(split, loss.item())
    return loss.item()


@torch.no_grad()
def sample(model: Sequential, itos: dict[int, str], num_words: int = 20) -> list[str]:
    model.eval()
    words: list[str] = []

    for _ in range(num_words):
        out: list[str] = []
        context = [0] * block_size
        while True:
            logits = model(torch.tensor([context]))
            probs = F.softmax(logits, dim=1)
            ix = int(torch.multinomial(probs, num_samples=1).item())
            context = context[1:] + [ix]
            if ix == 0:
                break
            out.append(itos[ix])
        words.append("".join(out))

    return words


def preview_dataset(xtr: torch.Tensor, ytr: torch.Tensor, itos: dict[int, str]) -> None:
    for x, y in zip(xtr[:20], ytr[:20]):
        print("".join(itos[int(ix.item())] for ix in x), "-->", itos[int(y.item())])


def train_call() -> None:
    torch.manual_seed(seed)

    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    itos, _ = helpers.find_mappings(chars, ".")

    print(len(words))
    print(max(len(w) for w in words))
    print(words[:8])
    print(itos)
    print(vocab_size)

    data = helpers.compute_data_sets(block_size)
    xtr, ytr = data.training.x, data.training.y
    xdev, ydev = data.validation.x, data.validation.y
    xte, yte = data.test.x, data.test.y

    preview_dataset(xtr, ytr, itos)

    model = build_mlp()
    print(sum(p.nelement() for p in model.parameters()))

    train(model, xtr, ytr)

    split_loss(model, "train", xtr, ytr, xdev, ydev, xte, yte)
    split_loss(model, "val", xtr, ytr, xdev, ydev, xte, yte)

    print("\nSamples:")
    for w in sample(model, itos):
        print(w)

    # -------------------------------------------------------------------------
    # WaveNet work starts here (lecture progression, implement yourself):
    # - FlattenConsecutive(n): fuse n consecutive embeddings, keep a time axis
    # - hierarchical stacking of FlattenConsecutive(2) + Linear blocks
    # - BatchNorm that handles 3D (batch, time, channels) activations
    # - scale n_embd / n_hidden once the architecture is in place
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    train_call()
