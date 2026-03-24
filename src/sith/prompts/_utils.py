# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT


def get_article(name: str) -> str:
    """Returns the appropriate indefinite article for a given name."""
    if name[0].lower() in "aeiou":
        return "an"
    return "a"
