import itertools
from typing import TYPE_CHECKING, Union

from jaxtyping import Float
import torch
from torch.optim.optimizer import ParamsT


def precondition(
    grad: Float[torch.Tensor, "mean_param ..."],
    precond_factor: Float[torch.Tensor, "mean_param ... rank"],
    dampening: float = 0.0,
    normalize: bool = True,
):
    needs_squeezing = False
    if grad.ndim < 3 and precond_factor.ndim >= 3:
        needs_squeezing = True
        grad = grad.unsqueeze(-1)

    # Uniformly damped spectrum
    preconditioned_grad = precond_factor @ (precond_factor.mT @ grad) + dampening * grad

    # Normalization
    if normalize:
        preconditioned_grad = preconditioned_grad / torch.linalg.vector_norm(
            preconditioned_grad, ord=2
        )

    # TODO: Handle low rank covariances

    # Lower bounded spectrum
    # U, L, V = torch.linalg.svd(precond_factor)
    # L[L < 1.0] = self.precond_dampening

    # L, Q = torch.linalg.eigh(precond_factor @ precond_factor.mT)
    # L[L < 1.0] = self.precond_dampening
    # preconditioned_grad = Q @ torch.diag_embed(L) @ Q.mT @ grad

    if not needs_squeezing:
        return preconditioned_grad
    else:
        return preconditioned_grad.squeeze(-1)


class NGD(torch.optim.SGD):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, torch.Tensor] = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        precond_dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        normalize: bool = True,
        *,
        maximize: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
        )
        self.precond_dampening = precond_dampening
        self.normalize = normalize

        # Sort param_groups by module
        self.param_groups.sort(key=lambda x: x["module"])

    def step(self, closure=None):
        """Perform a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Iterate through parameter groups by module
        for _, module_param_groups in itertools.groupby(
            self.param_groups, key=lambda x: x["module"]
        ):
            for param_group in module_param_groups:

                # Ignore parameters which do not have a gradient
                param_group["params"] = [
                    p for p in param_group["params"] if p.grad is not None
                ]

                stacked_cov_params = None
                if not any([".cov" in name for name in param_group["param_names"]]):
                    # Mean parameters
                    param_ndim_thresh = 1
                    if (
                        "cov_params" in param_group
                        and param_group["cov_params"] is not None
                    ):
                        cov_params = [
                            p if p.ndim > 2 else p.unsqueeze(1)
                            for p in param_group["cov_params"]
                        ]
                        stacked_cov_params = torch.concat(cov_params, dim=1)
                else:
                    # Covariance parameters
                    param_ndim_thresh = 2
                    cov_params = [
                        p if p.ndim > 2 else p.unsqueeze(1)
                        for p in param_group["params"]
                    ]
                    stacked_cov_params = torch.concat(cov_params, dim=1)

                grads = [
                    p.grad if p.grad.ndim > param_ndim_thresh else p.grad.unsqueeze(1)
                    for p in param_group["params"]
                ]
                stacked_grad = torch.concat(grads, dim=1)

                # Precondition mean parameters with a covariance, and covariance parameters
                if stacked_cov_params is not None:
                    stacked_grad = precondition(
                        stacked_grad,
                        precond_factor=stacked_cov_params,
                        dampening=self.precond_dampening,
                        normalize=self.normalize,
                    )

                # Update parameters
                start_idx = 0
                momentum_buffer_list = []
                lr = param_group["lr"]

                for i, param in enumerate(param_group["params"]):

                    # Split (preconditioned) stacked gradient into individual parameter gradients
                    if param.ndim == param_ndim_thresh:
                        end_idx = start_idx + 1
                    else:
                        end_idx = start_idx + param.shape[-param_ndim_thresh]

                    grad = stacked_grad[:, start_idx:end_idx, ...].reshape(param.shape)
                    start_idx = end_idx

                    # Weight decay
                    if param_group["weight_decay"] != 0:
                        grad = grad.add(param, alpha=param_group["weight_decay"])

                    # Momentum
                    if param_group["momentum"] != 0:
                        momentum_buffer_list.append(
                            self.state[param].get("momentum_buffer")
                        )

                        if momentum_buffer_list[i] is None:
                            momentum_buffer_list[i] = grad.detach().clone()
                        else:
                            momentum_buffer_list[i].mul_(param_group["momentum"]).add_(
                                grad, alpha=1 - param_group["dampening"]
                            )

                        if param_group["nesterov"]:
                            grad = grad.add(
                                momentum_buffer_list[i], alpha=param_group["momentum"]
                            )
                        else:
                            grad = momentum_buffer_list[i]

                    # Update parameters
                    param.data.add_(
                        grad, alpha=-lr if not param_group["maximize"] else lr
                    )

                    # Update momentum_buffer in state
                    if param_group["momentum"] != 0:
                        self.state[param]["momentum_buffer"] = momentum_buffer_list[i]

        return loss
