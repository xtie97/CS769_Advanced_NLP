from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data # gradient of the loss w.r.t. the parameter p
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                if len(state)==0:
                    # Initialize state
                    state["step"] = 0
                    state["m0"] = torch.zeros_like(p.data)
                    state["v0"] = torch.zeros_like(p.data)
                
                m0, v0 = state['m0'], state['v0']
                state["step"] += 1

                # Update first and second moments of the gradients
                state["m0"] = beta1 * m0 + (1-beta1) * grad
                state["v0"] = beta2 * v0 + (1-beta2) * grad**2
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                m0_hat = state["m0"] / bias_correction1
                v0_hat = state["v0"] / bias_correction2

                # Compute the effective learning rate            

                # Add weight decay before the main gradient-based updates.
                # Please note that we are using the "efficient version" given in https://arxiv.org/abs/1412.6980
                p.data -= alpha * group['weight_decay'] * p.data # update the weight decay
                p.data -= alpha * m0_hat / ( v0_hat**0.5 + eps) # update the parameters


        return loss
