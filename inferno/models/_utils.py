from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

from ..bnn import params

if TYPE_CHECKING:
    from jaxtyping import Float
    from torch import Tensor


def _check_cov(
    cov: params.FactorizedCovariance | dict[str, Any] | None,
    required_cov_keys: list[str],
):
    """
    Converts cov to a dictionary with required_cov_keys or fills in missing keys with default covariance None

    :param cov: covariance or dictionary of covariances or None
    :param required_cov_keys: covariance keys required by this module
    """
    if cov is None:
        cov = {key: None for key in required_cov_keys}
    elif isinstance(cov, params.FactorizedCovariance):
        cov = {key: copy.deepcopy(cov) for key in required_cov_keys}
    elif isinstance(cov, dict):

        # check for unexpected covs
        for key in cov.keys():
            if key not in required_cov_keys:
                raise ValueError(f"Covariance key {key} not recognized")

        # set missing covs to default value (None)
        for key in required_cov_keys:
            if key not in cov.keys():
                cov[key] = None
    return cov
