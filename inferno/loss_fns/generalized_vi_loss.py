from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class GeneralizedVILoss(nn.modules.loss._Loss):
    """Generalized Variational Inference Loss.

    :param loss:    Loss function encouraging the variational distribution
                    to be consistent with the data.
    :param divergence:  Divergence regularizing the variational distribution to remain
                        close to the prior.
    :param prior_model: Prior distribution.
    :param regularization_strength: Weight for the regularization term. Note that the reduction is
                                    applied at the end, meaning that the regularization strength is
                                    always relative to the loss summed over the batch.
    :param reduction:   Specifies the reduction to apply to the output over the data batch dimension:
                        ``'mean'`` | ``'sum'``. ``'mean'``: the weighted mean of the
                        batch is taken, ``'sum'``: the output will be summed over the batch.
    """

    def __init__(
        self,
        loss: nn.modules.loss._Loss,
        divergence: Callable[[nn.Module], nn.Module],
        prior_model: bnn.BNNMixin,
        regularization_strength: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__(reduction=reduction)

        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"Reduction must be one of 'mean' or 'sum' but is: {reduction}"
            )
        if regularization_strength is None:
            regularization_strength = 1.0

        self.loss = loss
        self.divergence = divergence
        self.prior_model = prior_model
        self.regularization_strength = regularization_strength

    def forward(
        self,
        input: Float[Tensor, "*sample batch prediction"],
        target: Float[Tensor, "batch target"],
        model: bnn.BNNMixin,
    ) -> Float[Tensor, ""]:
        raise NotImplementedError
