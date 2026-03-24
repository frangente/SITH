# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import abc
from typing import Any

from torch.utils.data import Dataset as TorchDataset


class ClassificationDataset(TorchDataset[tuple[Any, int]]):
    """Interface for classification datasets."""

    @property
    @abc.abstractmethod
    def classes(self) -> list[str]:
        """The list of class names in the dataset."""

    @property
    @abc.abstractmethod
    def labels(self) -> list[int]:
        """The list of class indices corresponding to the samples in the dataset.

        Returns:
            A list of integers where each integer is the class index for the
            corresponding sample in the dataset.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[Any, int]:
        """Returns the sample (image, target) at the specified index.

        Args:
            index: The index of the sample to retrieve.

        Returns:
            A tuple containing the sample (image, target), where
            - `image` is the input image (e.g., a tensor or PIL image).
            - `target` is the class index of the sample.
        """
