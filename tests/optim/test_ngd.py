import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision.transforms import v2 as transforms

import inferno
from inferno import bnn, models

import pytest


def get_train_test_datasets(
    input_shape: tuple[int, ...],
    num_train_data: int,
    num_test_data: int,
    noise_scale: float,
    generator: torch.Generator = torch.Generator().manual_seed(42),
) -> tuple[data.Dataset, data.Dataset]:

    with torch.no_grad():

        # Latent function
        def f(x):
            return (
                torch.sigmoid(10 * x.reshape(-1, np.prod(x.shape[1:])).mean(dim=-1))
                - 0.5
            )

        # Noisy observations
        def y(x, noise_scale=noise_scale):
            return f(x).squeeze() + noise_scale * torch.randn(x.shape[0])

        # Training data
        X_train_raw = (
            torch.rand((num_train_data, *input_shape), generator=generator) - 0.5
        )
        X_train_mean = X_train_raw.mean(dim=0)
        X_train_std = X_train_raw.std(dim=0)
        train_inputs_normalization_transform = transforms.Lambda(
            lambda x: (x - X_train_mean.unsqueeze(0)) / X_train_std.unsqueeze(0)
        )

        X_train = train_inputs_normalization_transform(X_train_raw)
        y_train = y(X_train_raw)

        y_train_mean = y_train.mean(dim=0)
        y_train_std = y_train.std(dim=0)
        train_targets_normalization_transform = transforms.Lambda(
            lambda y: (y - y_train_mean) / y_train_std
        )
        y_train = train_targets_normalization_transform(y_train)

        # Test data
        X_test_raw = 2 * (
            torch.rand((num_test_data, *input_shape), generator=generator) - 0.5
        )
        X_test = train_inputs_normalization_transform(X_test_raw)
        y_test = y(X_test_raw)
        y_test = train_targets_normalization_transform(y_test)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)

    return train_dataset, test_dataset


@pytest.mark.parametrize(
    "model,input_shape",
    [
        (
            bnn.Sequential(
                models.MLP(in_size=1, hidden_sizes=[8, 8], out_size=1),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1,),
        ),
        (
            bnn.Sequential(
                models.LeNet5(out_size=1),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1, 28, 28),
        ),
        # TODO: add models with attention layers
    ],
)
@pytest.mark.parametrize(
    "optimizer_kwargs",
    [
        {"momentum": 0.0, "weight_decay": 0.0, "maximize": False},
        {"momentum": 0.0, "weight_decay": 0.0, "maximize": True},
        {"momentum": 0.0, "weight_decay": 0.5, "maximize": False},
        {"momentum": 0.9, "weight_decay": 0.0, "maximize": False},
    ],
)
def test_reproduces_sgd_for_models_without_cov_params_in_standard_parametrization(
    model: bnn.BNNMixin, input_shape: tuple[int, ...], optimizer_kwargs: dict
):

    train_losses = {
        "SGD": [],
        "NGD": [],
    }

    for optimizer_name in ["SGD", "NGD"]:
        # RNG
        torch.manual_seed(42)
        rng_dataset = torch.Generator().manual_seed(14532)

        # Initialize model
        model.reset_parameters()

        # Dataset
        num_train_data = 100
        num_test_data = 100
        train_dataset, _ = get_train_test_datasets(
            input_shape=input_shape,
            num_train_data=num_train_data,
            num_test_data=num_test_data,
            noise_scale=0.01,
            generator=rng_dataset,
        )

        # Dataloader(s)
        batch_size = 16
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=rng_dataset,
        )

        # Loss function
        loss_fn = nn.MSELoss()

        # Optimizer
        lr = 0.1
        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(
                model.parameter_groups(lr=lr, optimizer="SGD"), **optimizer_kwargs
            )
        elif optimizer_name == "NGD":
            optimizer = inferno.optim.NGD(
                model.parameter_groups(lr=lr, optimizer="NGD"), **optimizer_kwargs
            )
        else:
            raise ValueError()

        # Training loop
        num_epochs = 5
        model.train()
        for _ in range(num_epochs):
            for X_batch, y_batch in iter(train_dataloader):
                optimizer.zero_grad()

                # Prediction
                y_batch_pred = model(X_batch)

                # Loss
                loss = loss_fn(y_batch_pred, y_batch)

                train_losses[optimizer.__class__.__name__].append(loss.detach().item())

                # Optimizer step
                loss.backward()
                optimizer.step()

    train_losses_sgd = np.asarray(train_losses["SGD"])
    train_losses_ngd = np.asarray(train_losses["NGD"])

    np.testing.assert_allclose(train_losses_sgd, train_losses_ngd)


@pytest.mark.parametrize(
    "model,input_shape,lr",
    [
        (
            bnn.Sequential(
                models.MLP(
                    in_size=1,
                    hidden_sizes=[8, 8],
                    cov=[
                        bnn.params.FactorizedCovariance(),
                        None,
                        bnn.params.FactorizedCovariance(),
                    ],
                    out_size=1,
                ),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1,),
            0.05,
        ),
        (
            bnn.Sequential(
                models.MLP(
                    in_size=1,
                    hidden_sizes=[8, 8],
                    cov=[
                        bnn.params.LowRankCovariance(2),
                        None,
                        bnn.params.LowRankCovariance(2),
                    ],
                    out_size=1,
                ),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1,),
            0.05,
        ),
        (
            bnn.Sequential(
                models.LeNet5(out_size=1, cov=bnn.params.FactorizedCovariance()),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1, 28, 28),
            1e-3,
        ),
        (
            bnn.Sequential(
                models.LeNet5(out_size=1, cov=bnn.params.LowRankCovariance(10)),
                nn.Flatten(-2, -1),
                parametrization=bnn.params.SP(),
            ),
            (1, 28, 28),
            1e-3,
        ),
        # TODO: add model with attention layers
    ],
)
def test_optimizes_models_with_different_covariances(
    model: bnn.BNNMixin, input_shape: tuple[int, ...], lr: float
):

    # RNG
    torch.manual_seed(42)
    rng_dataset = torch.Generator().manual_seed(14532)

    # Initialize model
    model.reset_parameters()

    # Dataset
    num_train_data = 100
    num_test_data = 100
    train_dataset, _ = get_train_test_datasets(
        input_shape=input_shape,
        num_train_data=num_train_data,
        num_test_data=num_test_data,
        noise_scale=0.01,
        generator=rng_dataset,
    )

    # Dataloaders
    batch_size = 16
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=rng_dataset,
    )

    # Loss function
    loss_fn = nn.MSELoss()

    # Optimizer
    optimizer = inferno.optim.NGD(model.parameter_groups(lr=lr, optimizer="NGD"))

    # Training loop
    num_epochs = 10
    train_losses = []
    model.train()
    for _ in range(num_epochs):
        for X_batch, y_batch in iter(train_dataloader):
            optimizer.zero_grad()

            # Prediction
            y_batch_pred = model(X_batch)

            # Loss
            loss = loss_fn(y_batch_pred, y_batch)

            train_losses.append(loss.detach().item())

            # Optimizer step
            loss.backward()
            optimizer.step()

    assert train_losses[0] > train_losses[-1]
