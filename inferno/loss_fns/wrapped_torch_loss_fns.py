"""Wrappers for torch loss functions to ensure compatibility with models that sample a set of predictions."""

import torch
from torch import Tensor, nn


def inputs_and_expanded_targets(inputs, targets):
    """Ensure loss can be computed with additional dimensions of (sampled) predictions in inputs.

    :param inputs: Inputs (predictions).
    :param targets: Targets.
    """

    if (
        torch.is_floating_point(targets)
        or torch.is_complex(targets)
        or inputs.ndim == targets.ndim
    ):
        num_extra_dims = inputs.ndim - targets.ndim
        if num_extra_dims > 0 and not (inputs.shape[num_extra_dims:] == targets.shape):
            raise ValueError("Shape mismatch between input and target.")
    else:
        # If targets are classes, the inputs should have one additional dimension (for probabilities)
        num_extra_dims = inputs.ndim - targets.ndim - 1

    if num_extra_dims > 0:
        targets = targets.expand(
            *inputs.shape[0:num_extra_dims], *(targets.ndim * (-1,))
        ).reshape(-1, *targets.shape[1:])

        inputs = inputs.reshape(-1, *inputs.shape[num_extra_dims + 1 :])
    elif num_extra_dims < 0:
        raise ValueError(
            f"Shapes of input and targets do not match (input.ndim={inputs.ndim}, target.ndim={targets.ndim}).",
            f" Only predictions may have extra dimensions.",
        )

    return inputs, targets


class MSELoss(nn.MSELoss):
    """Mean Squared Error / L2 Loss.

    This loss function wraps [torch.nn.MSELoss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class L1Loss(nn.L1Loss):
    """L1 Loss.

    This loss function wraps [torch.nn.L1Loss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss.

    This loss function wraps [torch.nn.CrossEntropyLoss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params weight: A manual rescaling weight given to each class.
        If given, has to be a Tensor of size `C`.
    :params ignore_index: Specifies a target value that is ignored
        and does not contribute to the input gradient. When ``reduction`` is
        ``mean``, the loss is averaged over non-ignored targets. Note that
        ``ignore_index`` is only applicable when the target contains class indices.
    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    :params label_smoothing: A float in [0.0, 1.0]. Specifies the amount
        of smoothing when computing the loss, where 0.0 means no smoothing. The targets
        become a mixture of the original ground truth and a uniform distribution as described in
        [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567).
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction="mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class NLLLoss(nn.NLLLoss):
    """Negative Log-Likelihood Loss.

    This loss function wraps [torch.nn.NLLLoss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params weight: A manual rescaling weight given to each class.
        If given, has to be a Tensor of size `C`.
    :params ignore_index: Specifies a target value that is ignored
        and does not contribute to the input gradient. When ``reduction`` is
        ``mean``, the loss is averaged over non-ignored targets. Note that
        ``ignore_index`` is only applicable when the target contains class indices.
    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class BCELoss(nn.BCELoss):
    """Binary Cross Entropy Loss.

    This loss function wraps [torch.nn.BCELoss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params weight: A manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__(
            weight=weight,
            reduction=reduction,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """This loss combines a `Sigmoid` layer and the `BCELoss` in one loss.

    This loss function wraps [torch.nn.BCEWithLogitsLoss][] to be compatible with sampled predictions from
    a probabilistic model.

    :params weight: A manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
    :params reduction: Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed.
    :params pos_weight: A weight of positive examples to be broadcasted with target.
        Must be a tensor with equal size along the class dimension to the number of classes.
        Pay close attention to PyTorch's broadcasting semantics in order to achieve the desired
        operations. For a target of size [B, C, H, W] (where B is batch size) pos_weight of
        size [B, C, H, W] will apply different pos_weights to each element of the batch or
        [C, H, W] the same pos_weights across the batch. To apply the same positive weight
        along all spatial dimensions for a 2D multi-class target [C, H, W] use: [C, 1, 1].
    """

    def __init__(
        self,
        weight: Tensor | None = None,
        reduction: str = "mean",
        pos_weight: Tensor | None = None,
    ):
        super().__init__(
            weight=weight,
            reduction=reduction,
            pos_weight=pos_weight,
        )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(*inputs_and_expanded_targets(input, target))
