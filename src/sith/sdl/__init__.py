# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Sparse Dictionary Learning (SDL) algorithms."""

from ._lasso import lasso
from ._matching_pursuit import (
    coherent_orthogonal_matching_pursuit,
    matching_pursuit,
    orthogonal_matching_pursuit,
)

__all__ = [
    "coherent_orthogonal_matching_pursuit",
    "lasso",
    "matching_pursuit",
    "orthogonal_matching_pursuit",
]
