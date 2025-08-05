"""PyTorch implementation of the Refined Lion optimizer."""

import torch
import math
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    r"""Implements the Refined Lion algorithm.

    This optimizer has been adapted from the paper 'Refined Lion: A new
    optimizer for better training of deep neural networks'.
    https://www.nature.com/articles/s41598-025-07112-4
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        alpha=1.0,
        bias=True,
    ):
        """Initialize the hyperparameters.
        Args:
          params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups.
          lr (float, optional): learning rate, denoted as eta (η) in the paper
            (default: 1e-3).
          betas (Tuple[float, float], optional): coefficients used for computing
            running averages of the gradient, denoted as beta1 (β1) and beta2 (β2)
            (default: (0.9, 0.99)).
          weight_decay (float, optional): decoupled weight decay coefficient,
            denoted as lambda (λ) in the paper (default: 0.01).
          alpha (float, optional): hyperparameter for the arctan transformation,
            denoted as alpha (α) in the paper (default: 1.0).
          bias (bool, optional): whether to use Adam-style bias correction
            (default: True).
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, alpha=alpha, bias=bias)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
          closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        Returns:
          the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1

                c_t = exp_avg.clone().mul_(beta1).add(grad, alpha=1 - beta1)

                if group["bias"]:
                    bias_correction = 1 - beta1**state["step"]
                    m_hat_t = c_t.div(bias_correction)
                else:
                    m_hat_t = c_t

                update = torch.arctan(m_hat_t * group["alpha"]) * (2 / math.pi)
                if group["weight_decay"] != 0:
                    update.add_(p, alpha=group["weight_decay"])

                p.add_(update, alpha=-group["lr"])

                exp_avg.mul_(beta2).add(grad, alpha=1 - beta2)

        return loss
