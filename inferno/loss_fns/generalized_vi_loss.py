from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class WassersteinDistance(nn.Module):
    """Wasserstein distance between two probabilistic models.

    Computes the Wasserstein distance between the distributions over parameters of two
    probabilistic models.

    Note: Currently the distributions are assumed to be Gaussian.

    :param input:  Input model.
    :param target: Target model.
    """

    # TODO: If we have biases, then the covariances are not tracked correctly, since
    # currently by iterating through parameters separately we are assuming they are all independent.
    # Use ideas from how we implement NGD to correct for this (i.e. iterate through parameters grouped by layers).

    # TODO: Currently only works for factorized and low rank covariances and fails silently otherwise.

    def __init__(self):
        super().__init__()

    def forward(
        self,
        input: bnn.BNNMixin,
        target: bnn.BNNMixin,
    ):
        _tmp_param = next(input.parameters())
        l2_norm_means = torch.zeros(
            (), dtype=_tmp_param.dtype, device=_tmp_param.device, requires_grad=True
        )
        bures_metric = torch.zeros(
            (), dtype=_tmp_param.dtype, device=_tmp_param.device, requires_grad=True
        )

        # Iterate over parameters
        for (name_param_input, param_input), (name_param_target, param_target) in zip(
            input.named_parameters(), target.named_parameters()
        ):

            if ".cov" not in name_param_input and ".cov" not in name_param_target:
                # Compute L2 distance for mean parameters
                l2_norm_means = (
                    l2_norm_means + (param_input - param_target).pow(2).sum()
                )
            elif ".cov" in name_param_input and ".cov" in name_param_target:
                # Compute Bures metric for covariance parameters
                # NOTE: Uses assumption that covariance is factorized.

                # Variance terms
                bures_metric = (
                    bures_metric + param_input.pow(2).sum() + param_target.pow(2).sum()
                )

                # Mixed term
                cov_input_sqrt = param_input.reshape(-1, param_input.shape[-1])
                U, Ssqrt, _ = torch.linalg.svd(cov_input_sqrt, full_matrices=True)
                if len(Ssqrt) < U.shape[-1]:
                    Ssqrt = torch.concat(
                        (
                            Ssqrt,
                            torch.zeros(
                                U.shape[-1] - len(Ssqrt),
                                dtype=Ssqrt.dtype,
                                device=Ssqrt.device,
                            ),
                        )
                    )
                USsqrt = U * Ssqrt

                cov_target_sqrt = param_target.reshape(-1, param_target.shape[-1])

                Msqrt = USsqrt.mT @ cov_target_sqrt
                _, S_Msqrt, _ = torch.linalg.svd(Msqrt)

                bures_metric = bures_metric - 2 * S_Msqrt.sum()

            else:
                raise ValueError("Models do not have the same parameters.")

        return l2_norm_means + bures_metric


class GeneralizedVILoss(nn.modules.loss._Loss):
    r"""Generalized Variational Inference Loss.

    Computes the regularized loss defined by

    $$
    \bar{\ell}_R(\theta) = \mathbb{E}_{q_{\theta}(w)}\big[\sum_{n=1}^N \ell(y_n, f_w(x_n))\big] + \lambda D(q_{\theta}(w), p(w))
    $$

    where $\ell(y_n, f_w(x_n))$ is a given loss function, $D(q_{\theta}(w), p(w))$ a divergence between the variational
    distribution $q_{\theta}(w)$ and the prior $p(w)$, and $\lambda > 0$ is the regularization strength.

    :param loss:    Loss function encouraging the variational distribution
                    to be consistent with the data.
    :param divergence:  Divergence regularizing the variational distribution to remain
                        close to the prior.
    :param prior_model: Model defining the prior distribution.
    :param regularization_strength: Weight for the regularization term. Note that this weight assumes
                                    a ``sum`` reduction. If the ``loss`` uses a ``mean`` reduction, then
                                    the regularization strength is divided by the batch size to be consistent
                                    with the normalization of the loss.
    """

    def __init__(
        self,
        loss: nn.modules.loss._Loss,
        divergence: Callable[[nn.Module], nn.Module],
        prior_model: bnn.BNNMixin,
        regularization_strength: float = 1.0,
    ) -> None:
        super().__init__(reduction=loss.reduction)

        if regularization_strength is None:
            regularization_strength = 1.0

        self.loss = loss
        self.divergence = divergence
        self.prior_model = copy.deepcopy(
            prior_model
        )  # Make copy so that turning off gradients doesn't cause unexpected behavior upstream.
        # Turn off gradients for prior
        for p in self.prior_model.parameters():
            p.requires_grad_(False)
        self.regularization_strength = regularization_strength

    def forward(
        self,
        input: Float[Tensor, "*sample batch"] | Float[Tensor, "*sample batch ..."],
        target: Float[Tensor, "batch"] | Float[Tensor, "batch ..."],
        model: bnn.BNNMixin,
    ) -> Float[Tensor, ""]:

        # Expected loss
        expected_loss = self.loss(input, target)

        # Regularizer given by divergence
        divergence_term = self.divergence(model, self.prior_model)

        # Adjust regularization strength for the type of reduction
        if self.loss.reduction == "sum":
            regularization_strength = self.regularization_strength
        elif self.loss.reduction == "mean":
            regularization_strength = self.regularization_strength / target.shape[0]
        else:
            raise NotImplementedError(
                f"Reduction {self.loss.reduction} not supported. Use one of 'mean' or 'sum'."
            )

        return expected_loss + regularization_strength * divergence_term
