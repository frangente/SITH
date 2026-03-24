# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

from typing import Any

from sklearn.model_selection import train_test_split

from ._dataset import ClassificationDataset


def split_dataset(
    dataset: ClassificationDataset,
    split_ratio: float,
    *,
    stratify: bool = True,
    shuffle: bool = True,
    seed: int | None = None,
) -> tuple[ClassificationDataset, ClassificationDataset]:
    """Splits a dataset into two parts based on the specified split ratio.

    Args:
        dataset: The dataset to split.
        split_ratio: If an integer, the number of samples in the first split.
            If a float, the proportion of the dataset to include in the first split.
        stratify: If `True`, the split will be stratified based on the labels, i.e.,
            the class distribution will be preserved in both splits.
        shuffle: If `True`, the dataset will be shuffled before splitting.
        seed: Random seed for reproducibility. If `None`, the random number generator
            will not be seeded.

    Returns:
        A tuple containing two datasets:
        - The first dataset containing the samples in the first split.
        - The second dataset containing the samples in the second split.
    """
    indices = list(range(len(dataset)))
    f_indices, s_indices = train_test_split(
        indices,
        train_size=split_ratio,
        stratify=dataset.labels if stratify else None,
        shuffle=shuffle,
        random_state=seed,
    )

    first_split = _SplitDataset(dataset, f_indices)
    second_split = _SplitDataset(dataset, s_indices)
    return first_split, second_split


class _SplitDataset(ClassificationDataset):
    """A dataset that wraps another dataset and includes only a subset of samples."""

    def __init__(self, dataset: ClassificationDataset, indices: list[int]) -> None:
        """Initializes the SplitDataset.

        Args:
            dataset: The original dataset to wrap.
            indices: The indices of the samples to include in the split dataset.
        """
        super().__init__()
        self._dataset = dataset
        self._indices = indices

    @property
    def classes(self) -> list[str]:
        return self._dataset.classes

    @property
    def labels(self) -> list[int]:
        return [self._dataset.labels[i] for i in self._indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        return self._dataset[self._indices[index]]
