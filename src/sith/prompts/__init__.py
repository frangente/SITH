# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Templates for image classification prompts."""

from . import openai
from ._template import Template
from ._utils import get_article

__all__ = [
    "Template",
    "get_article",
    "openai",
]
