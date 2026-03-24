# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import csv
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, override

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
    verify_str_arg,
)

from ._dataset import ClassificationDataset

_FOLDER_NAME = "food-101"
_IMAGES_URL = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
_METADATA_URL = "https://raw.githubusercontent.com/vturrisi/disef/refs/heads/main/fine-tune/artifacts/food101/metadata.csv"
_SPLIT_URL = "https://raw.githubusercontent.com/vturrisi/disef/refs/heads/main/fine-tune/artifacts/food101/split_coop.csv"


class Food101Dataset(ClassificationDataset):
    """The Food-101 dataset for image classification."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
        download: bool = False,
    ) -> None:
        """Initializes the Food-101 dataset.

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
        verify_str_arg(split, "split", ("train", "val", "test"))

        if download:
            self.download(root)

        if not self.check_exists(root):
            msg = "Dataset not found. You can use `download=True` to download it."
            raise RuntimeError(msg)

        root = Path(root) / _FOLDER_NAME
        with open(root / "metadata.csv") as f:
            reader = csv.DictReader(f)
            self._folder_to_class = {
                row["folder_name"]: row["class_name"] for row in reader
            }
        self._classes = list(self._folder_to_class.values())
        self._class_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        with open(root / "split_coop.csv") as f:
            reader = csv.DictReader(f)
            self._files = [
                root / "images" / row["filename"]
                for row in reader
                if row["split"] == split
            ]
        self._labels = [
            self._class_to_idx[self._folder_to_class[file.parent.name]]
            for file in self._files
        ]

        self.loader = loader
        self.transform = transform

    @classmethod
    def check_exists(cls, root: os.PathLike[str] | str) -> bool:
        """Checks if the dataset exists in the specified root directory."""
        root = Path(root) / _FOLDER_NAME
        return (
            (root / "metadata.csv").is_file()
            and (root / "split_coop.csv").is_file()
            and (root / "images").is_dir()
        )

    @classmethod
    def download(cls, root: os.PathLike[str] | str, *, force: bool = False) -> None:
        """Downloads the dataset into the specified root directory.

        Args:
            root: The root directory where the dataset will be downloaded.
            force: If `True`, forces the download even if the dataset already exists.
                If `False`, the download will be skipped if the dataset already exists
                in the specified root directory.

        Raises:
            RuntimeError: If the download fails or the dataset cannot be extracted.
        """
        if not force and cls.check_exists(root):
            return

        root = Path(root) / _FOLDER_NAME
        download_and_extract_archive(
            url=_IMAGES_URL,
            download_root=root,
            remove_finished=True,
        )
        shutil.move(root / "food-101" / "images", root)
        shutil.rmtree(root / "food-101")

        download_url(_METADATA_URL, root=root)
        download_url(_SPLIT_URL, root=root)

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
