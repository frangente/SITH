# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, override

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg

from ._dataset import ClassificationDataset

_FOLDER_NAME = "CUB_200_2011"


class CUB200Dataset(ClassificationDataset):
    """The CUB-200-2011 dataset for image classification."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initializes the CUB-200-2011 dataset.

        Args:
            root: The root directory where the dataset is stored or will be downloaded.
            split: The dataset split to use, either "train", "val", or "test".
            loader: A callable that loads an image from a file path.
            transform: A callable that takes in input the loaded image and returns
                a transformed version of the image. If `None`, no transformation is
                applied.
        """
        super().__init__()
        verify_str_arg(split, "split", ("train", "val", "test"))
        if split == "val":
            warnings.warn(
                "'val' split is not available for CUB-200-2011 dataset. Using 'test' split instead.",  # noqa: E501
                UserWarning,
                stacklevel=2,
            )
            split = "test"

        root = Path(root) / _FOLDER_NAME
        with open(root / "classes.txt") as f:
            classes = [line.strip().split(" ")[1] for line in f]
            classes = [c.split(".")[1].replace("_", " ") for c in classes]

        with open(root / "train_test_split.txt") as f:
            split_id = "1" if split == "train" else "0"
            lines = [line.strip().split(" ") for line in f]
            images_id = {line[0] for line in lines if line[1] == split_id}

        with open(root / "images.txt") as f:
            lines = [line.strip().split(" ") for line in f]
            images = {
                int(line[0]): root / "images" / Path(line[1])
                for line in lines
                if line[0] in images_id
            }

        with open(root / "image_class_labels.txt") as f:
            lines = [line.strip().split(" ") for line in f]
            self._files = []
            self._labels = []
            for line in lines:
                img_id, class_id = int(line[0]), int(line[1]) - 1
                if img_id in images:
                    self._files.append(images[img_id])
                    self._labels.append(class_id)

        self._classes = classes
        self.loader = loader
        self.transform = transform

    @property
    @override
    def classes(self) -> list[str]:
        return self._classes

    @property
    @override
    def labels(self) -> list[int]:
        return self._labels

    @override
    def __len__(self) -> int:
        return len(self._files)

    @override
    def __getitem__(self, index: int) -> tuple[Any, int]:
        image = self.loader(self._files[index])
        if self.transform is not None:
            image = self.transform(image)
        target = self._labels[index]

        return image, target
