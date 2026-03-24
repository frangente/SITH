# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import os
from collections.abc import Callable
from typing import Any, Literal, override

from torchvision.datasets import FGVCAircraft
from torchvision.datasets.folder import default_loader

from ._dataset import ClassificationDataset


class FGVCAircraftDataset(ClassificationDataset):
    """The FGVC Aircraft dataset for image classification."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[str], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
        download: bool = False,
    ) -> None:
        """Initializes the FGVC Aircraft dataset.

        Args:
            root: The root directory where the dataset is stored or will be downloaded.
            split: The dataset split to use, either "train", "val", or "test".
            loader: A callable that loads an image from a file path.
            transform: A callable that takes in input the loaded image and returns
                a transformed version of the image. If `None`, no transformation is
                applied.
            download: If `True`, the dataset will be downloaded if it is not already
                present in the specified root directory.
        """
        super().__init__()

        self._dataset = FGVCAircraft(
            root=str(root),
            split=split,
            loader=loader,
            transform=transform,
            download=download,
        )

    @property
    @override
    def classes(self) -> list[str]:
        return self._dataset.classes

    @property
    @override
    def labels(self) -> list[int]:
        return self._dataset._labels  # type: ignore[]

    @property
    def loader(self) -> Callable[[str], Any]:
        return self._dataset.loader

    @loader.setter
    def loader(self, loader: Callable[[str], Any]) -> None:
        self._dataset.loader = loader

    @property
    def transform(self) -> Callable[[Any], Any] | None:
        return self._dataset.transform

    @transform.setter
    def transform(self, transform: Callable[[Any], Any] | None) -> None:
        self._dataset.transform = transform

    @override
    def __len__(self) -> int:
        return len(self._dataset)

    @override
    def __getitem__(self, index: int) -> tuple[Any, int]:
        return self._dataset[index]
