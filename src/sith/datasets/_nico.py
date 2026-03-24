# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, override

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg

from ._dataset import ClassificationDataset

_FOLDER_NAME = "NICO"


class NICODataset(ClassificationDataset):
    """The NICO++ dataset."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initializes the NICO++ dataset.

        Args:
            root: The root directory where the dataset is stored or will be downloaded.
            split: The dataset split to use, either "train" or "test".
            loader: A callable that loads an image from a file path.
            transform: A callable that takes in input the loaded image and returns
                a transformed version of the image. If `None`, no transformation is
                applied.
            download: If `True`, the dataset will be downloaded if it is not already
                present in the specified root directory.
        """
        super().__init__()
        verify_str_arg(split, "split", ("train", "test"))

        root = Path(root) / _FOLDER_NAME
        image_dir = root / "NICO_unique"

        classes = [d.name for d in image_dir.iterdir() if d.is_dir()]
        self._classes = sorted(classes)
        self._class_to_idx = {cls_name: i for i, cls_name in enumerate(self._classes)}

        metadata_dir = root / "NICO_unique_official"
        self._files: list[Path] = []
        self._labels: list[int] = []
        for file in metadata_dir.glob(f"*_{split}.txt"):
            class_ = file.stem.split("_", maxsplit=1)[0]
            class_ = class_.replace("-", " ")
            class_idx = self._class_to_idx[class_]
            with open(file) as f:
                for row in f:
                    path = row.strip().rsplit(" ", maxsplit=1)[0]
                    if not path.endswith((".jpg", ".png", ".jpeg")):
                        continue

                    path = path.removeprefix("/home/hanyu/dataset/NICO_unique/")
                    self._files.append(image_dir / path)
                    self._labels.append(class_idx)

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
