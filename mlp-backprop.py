"""Manual backprop exercise boilerplate for the makemore MLP.

Forward pass is decomposed into elementary ops so each step can be
backpropped by hand and checked against PyTorch autograd via `cmp`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import helpers

# --- Hyperparameters ---
block_size = 3
n_embd = 10
n_hidden = 64
vocab_size = 27
batch_size = 32
bn_epsilon = 1e-5
seed = 2147483647


@dataclass
class MLPParams:
    w_emb: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    bn_gamma: torch.Tensor
    bn_beta: torch.Tensor

    @property
    def all(self) -> list[torch.Tensor]:
        return [self.w_emb, self.w1, self.b1, self.w2, self.b2, self.bn_gamma, self.bn_beta]


@dataclass
class ForwardState:
    """Intermediate tensors from the decomposed forward pass."""

    emb: torch.Tensor
    emb_flat: torch.Tensor
    pre_bn: torch.Tensor
    bn_mean: torch.Tensor
    bn_centered: torch.Tensor
    bn_centered_sq: torch.Tensor
    bn_var: torch.Tensor
    bn_inv_std: torch.Tensor
    bn_norm: torch.Tensor
    pre_act: torch.Tensor
    h: torch.Tensor
    logits: torch.Tensor
    logit_max: torch.Tensor
    logits_shifted: torch.Tensor
    exp_logits: torch.Tensor
    exp_sum: torch.Tensor
    inv_exp_sum: torch.Tensor
    probs: torch.Tensor
    log_probs: torch.Tensor
    loss: torch.Tensor

    @property
    def intermediates(self) -> list[torch.Tensor]:
        return [
            self.log_probs,
            self.probs,
            self.exp_logits,
            self.exp_sum,
            self.inv_exp_sum,
            self.logits_shifted,
            self.logit_max,
            self.logits,
            self.h,
            self.pre_act,
            self.bn_norm,
            self.bn_inv_std,
            self.bn_var,
            self.bn_centered_sq,
            self.bn_centered,
            self.pre_bn,
            self.bn_mean,
            self.emb_flat,
            self.emb,
        ]


def init_params(
    vocab_size: int,
    n_embd: int,
    block_size: int,
    n_hidden: int,
    generator: torch.Generator,
) -> MLPParams:
    """Initialize parameters with non-standard scaling to expose backprop bugs."""
    fan_in = n_embd * block_size

    w_emb = torch.randn(vocab_size, n_embd, generator=generator)
    w1 = torch.randn(fan_in, n_hidden, generator=generator) * (5 / 3) / (fan_in**0.5)
    b1 = torch.randn(n_hidden, generator=generator) * 0.1  # useless because of BN
    w2 = torch.randn(n_hidden, vocab_size, generator=generator) * 0.1
    b2 = torch.randn(vocab_size, generator=generator) * 0.1
    bn_gamma = torch.randn(1, n_hidden) * 0.1 + 1.0
    bn_beta = torch.randn(1, n_hidden) * 0.1

    params = MLPParams(
        w_emb=w_emb,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        bn_gamma=bn_gamma,
        bn_beta=bn_beta,
    )
    for p in params.all:
        p.requires_grad = True

    return params


def sample_minibatch(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(0, x.shape[0], (batch_size,), generator=generator)
    return x[ix], y[ix]


def forward(
    xb: torch.Tensor,
    yb: torch.Tensor,
    params: MLPParams,
    *,
    n: int,
    bn_epsilon: float,
) -> ForwardState:
    """Chunked forward pass — one elementary op per line for manual backprop."""
    w_emb, w1, b1, w2, b2, gamma, beta = (
        params.w_emb,
        params.w1,
        params.b1,
        params.w2,
        params.b2,
        params.bn_gamma,
        params.bn_beta,
    )

    emb = w_emb[xb]
    emb_flat = emb.view(emb.shape[0], -1)

    pre_bn = emb_flat @ w1 + b1

    bn_mean = (1 / n) * pre_bn.sum(0, keepdim=True)
    bn_centered = pre_bn - bn_mean
    bn_centered_sq = bn_centered**2
    bn_var = (1 / (n - 1)) * bn_centered_sq.sum(0, keepdim=True)
    bn_inv_std = (bn_var + bn_epsilon) ** -0.5
    bn_norm = bn_centered * bn_inv_std
    pre_act = gamma * bn_norm + beta

    h = torch.tanh(pre_act)

    logits = h @ w2 + b2

    logit_max = logits.max(1, keepdim=True).values
    logits_shifted = logits - logit_max
    exp_logits = logits_shifted.exp()
    exp_sum = exp_logits.sum(1, keepdim=True)
    inv_exp_sum = exp_sum**-1
    probs = exp_logits * inv_exp_sum
    log_probs = probs.log()
    loss = -log_probs[torch.arange(n), yb].mean()

    return ForwardState(
        emb=emb,
        emb_flat=emb_flat,
        pre_bn=pre_bn,
        bn_mean=bn_mean,
        bn_centered=bn_centered,
        bn_centered_sq=bn_centered_sq,
        bn_var=bn_var,
        bn_inv_std=bn_inv_std,
        bn_norm=bn_norm,
        pre_act=pre_act,
        h=h,
        logits=logits,
        logit_max=logit_max,
        logits_shifted=logits_shifted,
        exp_logits=exp_logits,
        exp_sum=exp_sum,
        inv_exp_sum=inv_exp_sum,
        probs=probs,
        log_probs=log_probs,
        loss=loss,
    )


def zero_grad(params: MLPParams) -> None:
    for p in params.all:
        p.grad = None


def retain_intermediate_grads(state: ForwardState) -> None:
    for t in state.intermediates:
        t.retain_grad()


def cmp(name: str, manual_grad: torch.Tensor, tensor: torch.Tensor) -> None:
    """Compare a manually computed gradient against autograd."""
    assert tensor.grad is not None
    exact = torch.all(manual_grad == tensor.grad).item()
    approx = torch.allclose(manual_grad, tensor.grad)
    maxdiff = (manual_grad - tensor.grad).abs().max().item()
    print(
        f"{name:15s} | exact: {str(exact):5s} | "
        f"approximate: {str(approx):5s} | maxdiff: {maxdiff}"
    )


def main() -> None:
    g = torch.Generator().manual_seed(seed)

    data = helpers.compute_data_sets(block_size)
    xtr, ytr = data.training.x, data.training.y

    params = init_params(vocab_size, n_embd, block_size, n_hidden, g)
    print(f"num params: {sum(p.nelement() for p in params.all)}")

    xb, yb = sample_minibatch(xtr, ytr, batch_size, g)
    n = batch_size

    state = forward(xb, yb, params, n=n, bn_epsilon=bn_epsilon)

    zero_grad(params)
    retain_intermediate_grads(state)
    state.loss.backward()

    # --- Manual backprop goes here ---
    # Work backwards from dlog_probs, then call cmp() for each tensor.
    #
    # Example (once you've derived the gradient):
    #   dlog_probs = torch.zeros_like(state.log_probs)
    #   dlog_probs[torch.arange(n), yb] = -1.0 / n
    #   cmp("log_probs", dlog_probs, state.log_probs)


if __name__ == "__main__":
    main()
