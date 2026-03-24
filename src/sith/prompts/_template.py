# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from collections.abc import Callable

type Template = Callable[[str], str]
"""A callable that takes the class name and returns the corresponding prompt."""
