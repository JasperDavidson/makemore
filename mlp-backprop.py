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
        return [
            self.w_emb,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.bn_gamma,
            self.bn_beta,
        ]


@dataclass
class ForwardState:
    """Intermediate tensors from the decomposed forward pass."""

    # Parameters (same tensors as MLPParams — so state.w2.grad etc. work)
    w_emb: torch.Tensor
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    bn_gamma: torch.Tensor
    bn_beta: torch.Tensor

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
        w_emb=w_emb,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        bn_gamma=gamma,
        bn_beta=beta,
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

    state = forward(xb, yb, params, n=batch_size, bn_epsilon=bn_epsilon)

    zero_grad(params)
    retain_intermediate_grads(state)
    state.loss.backward()

    # --- Manual backprop goes here ---
    # Work backwards from dlog_probs, then call cmp() for each tensor.
    #
    # Example (once you've derived the gradient):
    dlog_probs = torch.zeros_like(state.log_probs)
    dlog_probs[torch.arange(batch_size), yb] = -1.0 / batch_size
    cmp("log_probs", dlog_probs, state.log_probs)

    d_probs = dlog_probs / state.probs
    cmp("probs", d_probs, state.probs)

    d_inv_exp_sum = (d_probs * state.exp_logits).sum(1, keepdim=True)
    cmp("inv_exp_sum", d_inv_exp_sum, state.inv_exp_sum)

    d_exp_sum = -1 * state.exp_sum**-2 * d_inv_exp_sum
    cmp("exp_sum", d_exp_sum, state.exp_sum)

    d_exp_logits = d_probs * state.inv_exp_sum + d_exp_sum
    cmp("exp_logits", d_exp_logits, state.exp_logits)

    d_logits_shifted = d_exp_logits * state.exp_logits
    cmp("logits_shifted", d_logits_shifted, state.logits_shifted)

    d_logit_max = d_logits_shifted.sum(1, keepdim=True) * -1.0
    cmp("logit_max", d_logit_max, state.logit_max)

    max_selector = torch.zeros_like(state.logits)
    max_selector[torch.arange(batch_size), state.logits.argmax(1)] = 1.0
    d_logits = d_logits_shifted + d_logit_max * max_selector
    cmp("logits", d_logits, state.logits)

    ##### Equation: logits = h @ w2 + b2

    ### Non-batched
    # dlogits_j/dw2_i_j = h_i
    # d_loss/dw2_i_j = d_loss/dlogits_j * h_i

    ### Batched # dlogits_b_j/dw2_i_j = h_b_i
    # d_loss/dw2_i_j = sum(d_loss/dlogits_b_j * h_b_i) over b
    # d_loss/dlogits -> (batch size, vocab size)
    # h -> (batch_size, hidden_size)
    d_w2 = state.h.T @ d_logits
    cmp("w2", d_w2, state.w2)

    ### Non-batched
    # dlogits_j/d_h_k = w_k_j
    # d_loss/d_h_k = sum(d_loss/dlogits_j * w_k_j) over j

    ### Batched
    # dlogits_b_j/d_h_b_k = w_k_j
    # d_loss/d_h_b_k = sum(d_loss/dlogits_b_j * w_k_j) over j
    # d_loss/dlogits -> (batch size, vocab size)
    # w -> (hidden size, batch size)
    d_h = d_logits @ state.w2.T
    cmp("h", d_h, state.h)

    ### Non-batched
    # dlogits_j/d_b2 = 1
    # d_loss/d_b2 = d_loss/dlogits

    ### Batched
    # dlogits_b_j/d_b2 = 1
    # d_loss/d_b2 = sum(d_loss/dlogits_b) over b
    d_b2 = d_logits.sum(dim=0)
    cmp("b2", d_b2, state.b2)

    ##### Equation: h = torch.tanh(pre_act)

    # d_h/d_preact = 1 - tanh^2(preact)
    # d_loss/d_preact = d_loss/d_h * (1 - tanh^2(preact))
    d_pre_act = d_h * (1 - torch.tanh(state.pre_act) ** 2)
    cmp("pre_act", d_pre_act, state.pre_act)

    ##### Equation: pre_act = gamma + bn_norm + beta

    ### Non-batched
    # d_pre_act_j/d_gamma_j = bn_norm_j
    # d_loss/d_gamma_j = d_loss/d_pre_act_j * bn_norm_j

    ### Batched
    # d_pre_act_b_j/d_gamma_j = bn_norm_b_j
    # d_loss/d_gamma_j = (d_loss/d_pre_act_b_j * bn_norm_b_j) sum over b
    d_bngamma = (d_pre_act * state.bn_norm).sum(dim=0, keepdim=True)
    cmp("gamma", d_bngamma, state.bn_gamma)

    ### Non-batched
    # d_pre_act_j/d_bn_norm_j = gamma_j
    # d_loss/d_bn_norm_j = d_loss/d_pre_act_j * gamma_j

    ### Batched
    # d_pre_act_b_j/d_bn_norm_b_j = gamma_j
    # d_loss/d_bn_norm_b_j = d_loss/d_pre_act_b_j * gamma_j
    d_bn_norm = d_pre_act * state.bn_gamma
    cmp("bn_norm", d_bn_norm, state.bn_norm)

    # d_loss/d_beta_j = (d_loss/d_pre_act_b_j) sum over b
    d_bnbeta = d_pre_act.sum(dim=0)
    cmp("beta", d_bnbeta, state.bn_beta)

    ##### Equation: bn_norm = bn_centered * bn_inv_std

    # (batch, hidden) * (1, hidden) -> elementwise
    # d_bn_norm_b_j/d_bn_centered_b_j = bn_inv_std_j
    # d_loss/d_bn_centered_b_j = d_loss/d_bn_norm_b_j * bn_inv_std_j
    d_bn_centered = d_bn_norm * state.bn_inv_std

    # d_bn_norm_b_j/d_bn_inv_std_j = bn_centered_b_j
    # d_loss/d_bn_inv_std = (d_loss/d_bn_norm_b_j * bn_centered_b_j) sum over b
    d_bn_inv_std = (d_bn_norm * state.bn_centered).sum(dim=0, keepdim=True)
    cmp("bn_inv_std", d_bn_inv_std, state.bn_inv_std)

    ##### Equation: bn_inv_std = (bn_var + bn_epsilon) ** -0.5

    # d_bn_inv_std/d_bn_var = -1/2 * (bn_var + bn_epsilon) ** -1.5
    # d_loss/b_bn_var = d_loss/bn_inv_std * (-1/2 * (bn_var + bn_epsilon) ** -1.5)
    d_bn_var = d_bn_inv_std * (-0.5 * (state.bn_var + bn_epsilon) ** (-1.5))
    cmp("bn_var", d_bn_var, state.bn_var)

    # d_loss/b_bn_epsilon = 0 since it is constant

    ##### Equation: bn_var = (1 / (n - 1)) * bn_centered_sq.sum(dim=0, keepdim=True)

    # d_bn_var_j/d_bn_centered_sq_b_j = 1 / (n - 1) for all j
    # d_loss/d_bn_centered_sq_b_j = d_loss/d_bn_var_j * ((1) / (n - 1) for all j)
    d_bn_centered_sq = (
        torch.ones_like(state.bn_centered_sq) * d_bn_var * (batch_size - 1) ** -1
    )  # Note the explicit broadcasting here
    cmp("bn_centered_sq", d_bn_centered_sq, state.bn_centered_sq)

    ##### Equation: bn_centered_sq = bn_centered**2

    # d_bn_centered_sq_b_j/d_bn_centered_b_j = 2 * bn_centered_b_j
    # d_loss/d_bn_centered_b_j = d_loss/d_bn_centered_sq_b_j * 2 * bn_centered_b_j
    d_bn_centered += d_bn_centered_sq * (2 * state.bn_centered)
    cmp("bn_centered", d_bn_centered, state.bn_centered)

    ##### Equation: bn_centered = pre_bn - bn_mean -> (batch, hidden) = (batch, hidden) - (1, hidden)

    # d_bn_centered_b_j/d_pre_bn_b_j = 1
    # d_loss/d_pre_bn_b_j = d_loss/d_bn_centered_b_j
    d_pre_bn = d_bn_centered
    # TODO: cmp("pre_bn", d_pre_bn, state.pre_bn)

    # d_bn_centered_b_j/d_bn_mean_j = -1
    # d_loss/d_bn_mean_j = -1 * (d_loss/d_bn_centered_b_j) sum over b
    d_bn_mean = -1 * d_bn_centered.sum(dim=0)
    cmp("bn_mean", d_bn_mean, state.bn_mean)

    ##### Equation: bn_mean = (1 / n) * pre_bn.sum(dim=0, keepdim=True)
    # pre_bn = (batch, hidden)
    # bn_mean = (1, hidden)

    # d_bn_mean_j/d_pre_bn_b_j = 1 / n
    # d_loss/d_d_pre_bn_b_j = d_loss/d_bn_mean_j * (1 / n)
    d_pre_bn += d_bn_mean * (1 / batch_size)
    cmp("pre_bn", d_pre_bn, state.pre_bn)

    ##### Equation: pre_bn = emb_flat @ w1 + b1
    # pre_bn = (batch, hidden)
    # emb_flat = (batch, h_in)
    # w1 = (h_in, hidden)
    # b1 = (1, hidden)

    # (batch, in)
    # d_pre_bn_b_k/d_emb_flat_b_i = w1_i_k
    # d_loss/d_emb_flat_b_i = d_loss/d_pre_bn_b_k * w1_i_k
    d_emb_flat = d_pre_bn @ state.w1.T
    cmp("emb_flat", d_emb_flat, state.emb_flat)

    # (in, hidden)
    # d_pre_bn_b_k/d_w1_i_k = emb_flat_b_i
    # d_loss/d_w1_i_k = d_loss/d_pre_bn_b_k * emb_flat_b_i
    d_w1 = state.emb_flat.T @ d_pre_bn
    cmp("w1", d_w1, state.w1)

    # (1, hidden)
    # d_pre_bn_b_k/d_b1_k = 1
    # d_loss/d_b1_k = (d_loss/d_pre_bn_b_k) sum over b
    d_b1 = d_pre_bn.sum(dim=0)
    cmp("b1", d_b1, state.b1)

    ##### Equation: emb_flat = emb.view(emb.shape[0], -1)

    # Note no actual gradient, just reshaping the representation
    d_emb = d_emb_flat.view(batch_size, block_size, n_embd)

    ##### Equation: emb = w_emb[xb]
    # w_emb = (vocab_size, n_emb)
    # emb = (batch, block_size, n_emb)
    # xb = (batch, block_size)

    d_w_emb = torch.zeros_like(state.w_emb)
    for batch_idx in range(batch_size):
        for block_idx in range(block_size):
            d_w_emb[xb[batch_idx][block_idx]] += d_emb[batch_idx][block_idx]
    cmp("d_w_emb", d_w_emb, state.w_emb)


if __name__ == "__main__":
    main()
