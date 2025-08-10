import torch
import math
from torch.optim.optimizer import Optimizer

class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.99),
        weight_decay=0.01,
        alpha=1.0,
        use_bias_correction=True,
    ):
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
        defaults = dict(
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            alpha=alpha,
            use_bias_correction=use_bias_correction
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            alpha = group["alpha"]
            bc = group["use_bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg = state["exp_avg"]

                c_t = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)

                if bc:
                    c_hat = c_t.div(1 - beta1**state["step"])
                else:
                    c_hat = c_t

                update = (2.0 / math.pi) * torch.atan(alpha * c_hat)
                if wd != 0:
                    update = update.add(p, alpha=wd)

                p.add_(update, alpha=-lr)

                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss
