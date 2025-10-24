from numpy import testing as npt
import torch

from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "covariance,mean_parameters",
    [
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2, 3, 2), "bias": torch.randn((1, 3, 2))},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2), "bias": torch.randn((1,))},
        ),
        (
            params.FactorizedCovariance(),
            {
                "weight": torch.randn(
                    1,
                )
            },
        ),
    ],
)
def test_factor_matmul(covariance, mean_parameters):
    """Test whether the multiplication with the covariance factor is correct."""

    torch.manual_seed(2435)

    numel_parameters = sum(
        [tens.numel() for tens in mean_parameters.values() if tens is not None]
    )

    covariance.initialize_parameters(mean_parameters)
    covariance.reset_parameters({name: 1.0 for name in mean_parameters.keys()})

    matmul_result_dict = covariance.factor_matmul(
        torch.eye(covariance.rank), additive_constant=torch.zeros(numel_parameters)
    )

    for name, tens in matmul_result_dict.items():
        npt.assert_allclose(
            torch.movedim(tens, 0, -1)
            .detach()
            .numpy(),  # Rank dimension is in front for factor_matmul
            covariance.factor[name].detach().numpy(),
        )


@pytest.mark.parametrize(
    "covariance,mean_parameters",
    [
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2, 3, 2), "bias": torch.randn((1, 3, 2))},
        ),
        (
            params.FactorizedCovariance(),
            {"weight": torch.randn(1, 2), "bias": torch.randn((1,))},
        ),
        (
            params.FactorizedCovariance(rank=5),
            {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        ),
    ],
)
def test_covariance_initialized_to_identity(covariance, mean_parameters):
    """Test whether the covariance matrix is initialized to identity."""

    torch.manual_seed(42)

    covariance.initialize_parameters(mean_parameters)

    # Reset parameters with uniform scaling of 1.0
    covariance.reset_parameters(1.0)

    dense_cov = covariance.to_dense()

    # The covariance should be approximately an identity matrix
    # (scaled by the mean parameter scales, which are all 1.0 here and only up to 'rank' diagonal entries)
    expected_identity = torch.zeros_like(dense_cov)
    diag_tensor = torch.zeros(dense_cov.shape[0])
    diag_tensor[0 : covariance.rank] = torch.ones(covariance.rank)
    expected_identity = torch.diagonal_scatter(expected_identity, diag_tensor)

    # Check if the covariance matrix is close to identity
    npt.assert_allclose(
        dense_cov.detach().numpy(),
        expected_identity.numpy(),
        rtol=1e-5,
        atol=1e-5,
        err_msg="Covariance matrix should be initialized to identity",
    )
