from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from torch import nn

from inferno import bnn

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


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
    :param prior_model: Prior distribution.
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
        self.prior_model = prior_model
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

        if self.loss.reduction == "sum":
            regularization_strength = self.regularization_strength
        elif self.loss.reduction == "mean":
            regularization_strength = self.regularization_strength / target.shape[0]
        else:
            raise NotImplementedError(
                f"Reduction {self.loss.reduction} not supported. Use one of 'mean' or 'sum'."
            )

        return expected_loss + regularization_strength * divergence_term
