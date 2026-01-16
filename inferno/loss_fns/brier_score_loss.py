from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor

__all__ = ["FocalLoss"]


class BrierScoreLoss(torch.nn.modules.loss._Loss):
    r"""Brier score.

    The Brier score loss on a single datapoint is given by

    $$
        \ell_n = \frac{1}{C} \lVert \operatorname{onehot}(y_n) - \sigma(f_n) \rVert_2^2
    $$

    where $f_n \in \R^C$ are the logits for a given input and $\operatorname{onehot}(y_n)$ is the one-hot encoded target.

    :param task: Specifies the type of task: 'binary' or 'multiclass'.
    :param reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``.
        ``'none'``: no reduction will be applied,
        ``'mean'``: the weighted mean of the output is taken,
        ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        task: Literal["binary", "multiclass"],
        reduction: Literal["none", "sum", "mean"] = "mean",
    ):
        super().__init__(
            reduction=reduction,
        )
        if reduction not in ["none", "sum", "mean"]:
            raise ValueError(
                f"Unsupported reduction '{self.reduction}'. Use 'none', 'sum', or 'mean'."
            )
        self.task = task

    def forward(
        self,
        pred: Float[Tensor, "*sample batch"] | Float[Tensor, "*sample batch class"],
        target: Float[Tensor, "batch"],
    ):
        if self.task == "binary":
            # Predicted class probabilities
            probs = torch.sigmoid(pred)

            # Expand target appropriately
            # TODO

            return torch.nn.functional.mse_loss(probs, target, reduction=self.reduction)

        elif self.task == "multiclass":
            # Predicted class probabilities
            probs = torch.softmax(pred, dim=-1)
            one_hot_target = torch.nn.functional.one_hot(
                target.to(torch.long), num_classes=probs.shape[-1]
            ).float()

            # TODO: expand one_hot_targets appropriately

            return torch.nn.functional.mse_loss(
                probs, one_hot_target, reduction=self.reduction
            )  # TODO: do we need to divide by the number of classes here?

        else:
            raise ValueError(
                f"Unsupported task '{self.task}'. Use 'binary' or 'multiclass'."
            )
