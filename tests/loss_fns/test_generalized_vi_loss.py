import copy

import numpy.testing as npt
import torch
from torch import nn

from inferno import bnn, loss_fns, models

import pytest


@pytest.mark.parametrize(
    "divergence",
    [
        loss_fns.WassersteinDistance(),
    ],
    ids=lambda m: m.__class__.__name__,
)
@pytest.mark.parametrize(
    "model",
    [
        bnn.Linear(5, 2),
        bnn.Linear(5, 3, cov=bnn.params.FactorizedCovariance(), bias=False),
        bnn.Linear(5, 5, cov=bnn.params.LowRankCovariance(2)),
        models.MLP(
            in_size=5,
            hidden_sizes=[2, 3],
            out_size=2,
            cov=[
                bnn.params.FactorizedCovariance(),
                None,
                bnn.params.FactorizedCovariance(),
            ],
        ),
        models.LeNet5(cov=bnn.params.LowRankCovariance(10)),
        models.ResNet18(out_size=10, cov=bnn.params.LowRankCovariance(10)),
    ],
    ids=lambda m: m.__class__.__name__,
)
def test_divergence_equals_zero_for_same_distributions(
    divergence: nn.Module, model: bnn.BNNMixin
):
    torch.manual_seed(13423)
    model.reset_parameters()

    with torch.no_grad():
        d = divergence(model, copy.deepcopy(model))
    npt.assert_allclose(d.detach().cpu().numpy(), 0.0, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "model",
    [
        bnn.Linear(5, 2),
        bnn.Linear(5, 3, cov=bnn.params.FactorizedCovariance(), bias=False),
        bnn.Linear(5, 5, cov=bnn.params.LowRankCovariance(2)),
        models.MLP(
            in_size=5,
            hidden_sizes=[2, 3],
            out_size=2,
            cov=[
                bnn.params.FactorizedCovariance(),
                None,
                bnn.params.FactorizedCovariance(),
            ],
        ),
        models.LeNet5(cov=bnn.params.LowRankCovariance(10)),
        models.ResNet18(out_size=10, cov=bnn.params.LowRankCovariance(10)),
    ],
    ids=lambda m: m.__class__.__name__,
)
def test_wasserstein_distance_is_symmetric(model):
    torch.manual_seed(46134)
    model.reset_parameters()
    model1 = copy.deepcopy(model)
    model1.reset_parameters()

    with torch.no_grad():
        npt.assert_allclose(
            loss_fns.WassersteinDistance()(model, model1).detach().cpu().numpy(),
            loss_fns.WassersteinDistance()(model1, model).detach().cpu().numpy(),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize(
    "divergence",
    [
        loss_fns.WassersteinDistance(),
    ],
    ids=lambda m: m.__class__.__name__,
)
@pytest.mark.parametrize(
    "model",
    [
        bnn.Linear(5, 2),
        bnn.Linear(5, 3, cov=bnn.params.FactorizedCovariance(), bias=False),
        bnn.Linear(5, 5, cov=bnn.params.LowRankCovariance(2)),
        models.MLP(
            in_size=5,
            hidden_sizes=[2, 3],
            out_size=2,
            cov=[
                bnn.params.FactorizedCovariance(),
                None,
                bnn.params.FactorizedCovariance(),
            ],
        ),
        models.LeNet5(cov=bnn.params.LowRankCovariance(10)),
        # models.ResNet18(out_size=10, cov=bnn.params.LowRankCovariance(10)), # Increases time for tests considerably.
    ],
    ids=lambda m: m.__class__.__name__,
)
def test_divergence_is_differentiable(divergence, model):
    torch.manual_seed(46134)
    model.reset_parameters()
    model1 = copy.deepcopy(model)
    model1.reset_parameters()
    for p in model1.parameters():
        p.requires_grad_(False)

    model.zero_grad()

    loss = divergence(model, model1)

    loss.backward()

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert not any(
                    torch.isnan(param.grad).flatten()
                ), f"Parameter {name} contains NaNs."


@pytest.mark.parametrize(
    "model",
    [
        bnn.Linear(5, 3, cov=bnn.params.FactorizedCovariance(), bias=False),
        bnn.Linear(5, 5, cov=bnn.params.FactorizedCovariance(), bias=False),
    ],
    ids=lambda m: m.__class__.__name__,
)
def test_wasserstein_distance_against_naive_implementation_for_single_layer(model):
    torch.manual_seed(13423)
    model.reset_parameters()
    model1 = copy.deepcopy(model)
    model1.reset_parameters()

    with torch.no_grad():
        wasserstein_inferno = loss_fns.WassersteinDistance()(model, model1)

        # Naive implementation of Wasserstein for linear model.
        model_cov_factor = model.params.cov.factor.weight.reshape(
            -1, model.params.cov.factor.weight.shape[-1]
        )
        model_cov = model_cov_factor @ model_cov_factor.mT
        model1_cov_factor = model1.params.cov.factor.weight.reshape(
            -1, model1.params.cov.factor.weight.shape[-1]
        )
        model1_cov = model1_cov_factor @ model1_cov_factor.mT
        S1, U1 = torch.linalg.eigh(model1_cov)
        model1_cov_sqrt = (U1 * S1.sqrt()) @ U1.mT
        M = model1_cov_sqrt @ model_cov @ model1_cov_sqrt.mT
        S_eigvals = torch.linalg.eigvalsh(M)
        bures_metric = (
            torch.trace(model_cov)
            + torch.trace(model1_cov)
            - 2 * S_eigvals.sqrt().sum()
        )

        wasserstein_naive = (model.params.weight - model1.params.weight).pow(
            2
        ).sum() + bures_metric

    npt.assert_allclose(wasserstein_inferno, wasserstein_naive, rtol=1e-4, atol=1e-4)
