import math

from numpy import testing as npt
import torch

from inferno.bnn import params

import pytest


@pytest.mark.parametrize(
    "sample_shape", [(), (1,), (2, 3)], ids=lambda x: f"sample_shape_{str(x)}"
)
@pytest.mark.parametrize(
    "covariance,mean_parameters",
    [
        (
            params.KroneckerCovariance(),
            {"weight": torch.randn(4, 2), "bias": torch.randn(4)},
        ),
        (
            params.KroneckerCovariance(input_rank=2, output_rank=3),
            {"weight": torch.randn(5, 2, 3, 2), "bias": torch.randn((5,))},
        ),
        (
            params.KroneckerCovariance(),
            {"weight": torch.randn(1, 2), "bias": torch.randn((1,))},
        ),
        (
            params.KroneckerCovariance(input_rank=99, output_rank=None),
            {"weight": torch.randn(1, 1)},
        ),
    ],
    ids=["linear", "conv2d", "linear_1d", "linear_large_rank"],
)
def test_factor_matmul(sample_shape, covariance, mean_parameters):
    """Test whether the multiplication with the covariance factor is correct."""

    torch.manual_seed(2435)

    numel_parameters = sum(
        [tens.numel() for tens in mean_parameters.values() if tens is not None]
    )

    covariance.initialize_parameters(mean_parameters)
    covariance.reset_parameters({name: 1.0 for name in mean_parameters.keys()})

    input = torch.randn(*sample_shape, covariance.rank)

    # Efficient Kronecker matmul
    matmul_result_dict = covariance.factor_matmul(
        input, additive_constant=torch.zeros(numel_parameters)
    )

    # Dense Kronecker matmul
    stacked_input_factor, stacked_output_factor = covariance._stacked_parameters()
    kronecker_product_of_factors = torch.kron(
        stacked_input_factor, stacked_output_factor
    )
    result_dense_kron = (kronecker_product_of_factors @ input.unsqueeze(-1)).squeeze(-1)

    dense_matmul_result_dict = {}
    start_idx_param_result = 0
    for name in mean_parameters.keys():
        if name == "weight":
            end_idx_param_result = (
                start_idx_param_result
                + covariance.output_factor.shape[0]
                * math.prod(covariance.input_factor[name].shape[:-1])
            )
            dense_matmul_result_dict[name] = result_dense_kron[
                ..., start_idx_param_result:end_idx_param_result
            ].view(
                *sample_shape,
                covariance.output_factor.shape[0],
                *covariance.input_factor[name].shape[:-1],
            )
        elif name == "bias" and covariance.input_factor["bias"] is not None:
            end_idx_param_result = (
                start_idx_param_result + covariance.output_factor.shape[0]
            )
            dense_matmul_result_dict[name] = result_dense_kron[
                ..., start_idx_param_result:end_idx_param_result
            ]

        start_idx_param_result = end_idx_param_result

    for name, tens in matmul_result_dict.items():
        npt.assert_allclose(
            tens.detach(), dense_matmul_result_dict[name].detach(), atol=1e-5, rtol=1e-5
        )
