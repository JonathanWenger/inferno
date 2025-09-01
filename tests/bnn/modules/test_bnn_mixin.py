import torch
from torch import nn

from inferno import bnn, models

import pytest


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.Linear, {"in_features": 5, "out_features": 2}),
        (nn.Conv1d, {"in_channels": 3, "out_channels": 1, "kernel_size": 1}),
    ],
)
def test_mixin_overrides_torch_module_forward(TorchClass: nn.Module, kwargs: dict):
    """Test when mixing in a BNNMixin into an nn.Module forces reimplementing forward."""

    x = torch.zeros((3, 5))

    # Mixin as first superclass forces reimplementation
    class MyBNNModule(bnn.BNNMixin, TorchClass):
        pass

    my_bnn_module = MyBNNModule(**kwargs)

    with pytest.raises(NotImplementedError):
        my_bnn_module(x)

    # Mixin as second superclass falls back to nn.Module.forward
    class MyBNNModule(TorchClass, bnn.BNNMixin):
        pass

    my_bnn_module = MyBNNModule(**kwargs)

    my_bnn_module(x)  # Does not raise error.


@pytest.mark.parametrize(
    "TorchClass,kwargs",
    [
        (nn.Linear, {"in_features": 5, "out_features": 2}),
        (nn.Conv1d, {"in_channels": 3, "out_channels": 1, "kernel_size": 1}),
    ],
)
@pytest.mark.parametrize(
    "parametrization",
    [bnn.params.SP(), bnn.params.MUP(), bnn.params.NTP()],
    ids=lambda c: c.__class__.__name__,
)
def test_mixin_allows_setting_parametrization(
    TorchClass: nn.Module, kwargs: dict, parametrization: bnn.params.Parametrization
):

    class MyBNNModule(bnn.BNNMixin, TorchClass):
        pass

    my_bnn_module = MyBNNModule(**kwargs, parametrization=parametrization)

    assert isinstance(my_bnn_module.parametrization, parametrization.__class__)


class NNModuleWithSubmodules(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer0 = bnn.Linear(3, 2)
        self.layer1 = nn.Linear(2, 1)


@pytest.mark.parametrize(
    "module",
    [
        bnn.Linear(5, 2),
        bnn.Linear(5, 2, bias=False),
        bnn.Linear(6, 3, cov=bnn.params.LowRankCovariance(2)),
        models.MLP(
            in_size=5,
            hidden_sizes=[8, 8, 8],
            out_size=1,
            cov=[
                bnn.params.FactorizedCovariance(),
                None,
                None,
                bnn.params.FactorizedCovariance(),
            ],
            bias=True,
        ),
        bnn.Sequential(
            NNModuleWithSubmodules(),
            NNModuleWithSubmodules(),
            parametrization=bnn.params.SP(),
        ),
    ],
)
@pytest.mark.parametrize("kwargs", [{"lr": 0.1, "optimizer": "SGD"}])
def test_named_parameter_groups_returns_all_parameters(
    module: bnn.BNNMixin, kwargs: dict
):
    params_via_parameter_groups = {
        name: group
        for name, group in module.named_parameter_groups(groupby=None, **kwargs)
    }

    assert len(list(module.parameters())) == len(params_via_parameter_groups)
