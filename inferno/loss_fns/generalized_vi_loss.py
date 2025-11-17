from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


class GeneralizedVILoss(nn.Module):
    """Generalized Variational Inference Loss.




    :param nll:                 Loss function defining the negative log-likelihood.
    :param model:               The probabilistic model.
    :param kl_weight:           Weight for the KL divergence term. If `None`, chooses the
        weight inversely proportional to the number of mean parameters.
    :param reduction:           Specifies the reduction to apply to the output:
        ````'mean'`` | ``'sum'``. ``'mean'``: the weighted mean of the output is taken,
        ``'sum'``: the output will be summed.
    """
