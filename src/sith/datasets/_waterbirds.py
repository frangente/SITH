# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import csv
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, override

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg

from ._dataset import ClassificationDataset

_SPLITS = {"train": 0, "val": 1, "test": 2}
_FOLDER_NAME = "waterbird_complete95_forest2water2"
_URL = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
_MD5 = "9cde878366a5db3504d3a351d2afeae9"


class WaterbirdsDataset(ClassificationDataset):
    """The Waterbirds dataset for image classification."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
        download: bool = False,
    ) -> None:
        """Initializes the Waterbirds dataset.

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
        verify_str_arg(split, "split", tuple(_SPLITS.keys()))

        if download:
            self.download(root)

        if not self.check_exists(root):
            msg = "Dataset not found. You can use `download=True` to download it."
            raise RuntimeError(msg)

        def get_class(path: Path) -> str:
            """Extracts the class name from a directory/file name."""
            # remove the file extension if present
            dir_ = path if path.is_dir() else path.parent
            _, name = dir_.name.split(".", 1)
            return name.replace("_", " ")

        root = Path(root) / _FOLDER_NAME
        directories = [dir_ for dir_ in root.iterdir() if dir_.is_dir()]
        self._classes = [get_class(dir_) for dir_ in sorted(directories)]
        classes_to_idx = {cls: idx for idx, cls in enumerate(self._classes)}

        with open(root / "metadata.csv") as f:
            reader = csv.reader(f)
            _ = next(reader)  # Skip header row
            split_idx = _SPLITS[split]
            rows = [row for row in reader if int(row[3]) == split_idx]

        self._files = [root / row[1] for row in rows]
        self._places_labels = [int(row[4]) for row in rows]
        self._labels = [classes_to_idx[get_class(file)] for file in self._files]

        self.loader = loader
        self.transform = transform

    @classmethod
    def check_exists(cls, root: os.PathLike[str] | str) -> bool:
        """Checks if the Waterbirds dataset exists in the specified root directory."""
        root = Path(root)
        return (root / _FOLDER_NAME / "metadata.csv").is_file()

    @classmethod
    def download(cls, root: os.PathLike[str] | str, *, force: bool = False) -> None:
        """Downloads the Waterbirds dataset.

        Args:
            root: The root directory where the dataset will be downloaded.
            force: If `True`, forces the download even if the dataset already exists.
                If `False`, the download will be skipped if the dataset already exists
                in the specified root directory.

        Raises:
            RuntimeError: If the download fails or the dataset cannot be extracted.
        """
        if cls.check_exists(root) and not force:
            return

        download_and_extract_archive(
            _URL,
            download_root=Path(root),
            md5=_MD5,
            remove_finished=True,
        )

    @property
    @override
    def classes(self) -> list[str]:
        return self._classes

    @property
    @override
    def labels(self) -> list[int]:
        return self._labels

    @property
    def places_labels(self) -> list[int]:
        """Returns the labels indicating the background type (land or water)."""
        return self._places_labels

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


class BinaryWaterbirdsDataset(ClassificationDataset):
    """The Waterbirds dataset filtered to only include 'land' and 'water' classes."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
        download: bool = False,
    ) -> None:
        """Initializes the BinaryWaterbirds dataset."""
        super().__init__()

        verify_str_arg(split, "split", tuple(_SPLITS.keys()))
        if download:
            self.download(root)
        if not self.check_exists(root):
            msg = "Dataset not found. You can use `download=True` to download it."
            raise RuntimeError(msg)

        self._classes = ["landbird", "waterbird"]

        root = Path(root) / _FOLDER_NAME
        with open(root / "metadata.csv") as f:
            reader = csv.reader(f)
            _ = next(reader)
            split_idx = _SPLITS[split]
            rows = [row for row in reader if int(row[3]) == split_idx]

        self._files = [root / row[1] for row in rows]
        self._labels = [int(row[2]) for row in rows]
        self._places_labels = [int(row[4]) for row in rows]

        self.loader = loader
        self.transform = transform

    @classmethod
    def check_exists(cls, root: os.PathLike[str] | str) -> bool:
        """Checks if the Waterbirds dataset exists in the specified root directory."""
        root = Path(root)
        return (root / _FOLDER_NAME / "metadata.csv").is_file()

    @classmethod
    def download(cls, root: os.PathLike[str] | str, *, force: bool = False) -> None:
        """Downloads the Waterbirds dataset.

        Args:
            root: The root directory where the dataset will be downloaded.
            force: If `True`, forces the download even if the dataset already exists.
                If `False`, the download will be skipped if the dataset already exists
                in the specified root directory.

        Raises:
            RuntimeError: If the download fails or the dataset cannot be extracted.
        """
        if cls.check_exists(root) and not force:
            return

        download_and_extract_archive(
            _URL,
            download_root=Path(root),
            md5=_MD5,
            remove_finished=True,
        )

    @property
    @override
    def classes(self) -> list[str]:
        return self._classes

    @property
    @override
    def labels(self) -> list[int]:
        return self._labels

    @property
    def places_labels(self) -> list[int]:
        """Returns the labels indicating the background type (land or water)."""
        return self._places_labels

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
