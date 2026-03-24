# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import verify_str_arg


class ViSUDataset(Dataset[tuple[Any, str, Any, str]]):
    """The ViSU dataset proposed by SafeCLIP."""

    def __init__(
        self,
        root: os.PathLike[str] | str,
        split: Literal["train", "val", "test"],
        *,
        loader: Callable[[Path], Any] = default_loader,
        transform: Callable[[Any], Any] | None = None,
    ) -> None:
        """Initializes the ViSU dataset.

        Args:
            root: The root directory where the dataset is stored.
            split: The dataset split to use, either "train", "val", or "test".
            loader: A callable that loads an image from a file path.
            transform: A callable that takes in input the loaded image and returns
                a transformed version of the image. If `None`, no transformation is
                applied.
        """
        super().__init__()
        verify_str_arg(split, "split", ("train", "val", "test"))

        self.root = Path(root) / "visu"
        with open(self.root / f"{split}.json") as f:
            self.metadata = json.load(f)

        self.coco_root = self.root / "coco_images"
        self.unsafe_root = self.root / ("test" if split == "test" else "train")

        self.loader = loader
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[Any, str, Any, str]:
        """Returns the sample at the given index.

        Each sample is a quadruple:
        - the safe image taken from COCO
        - the caption associated with the safe image
        - the unsafe image taken from ViSU
        - the caption associated with the unsafe image
        """
        info = self.metadata[index]
        id_ = info["incremental_id"]
        coco_id = info["coco_id"]
        safe_caption = info["safe"]
        unsafe_caption = info["nsfw"]

        safe_path = self.coco_root / f"{coco_id}.jpg"
        unsafe_path = self.unsafe_root / f"{id_}.jpg"
        safe_image = self.loader(safe_path)
        unsafe_image = self.loader(unsafe_path)

        if self.transform is not None:
            safe_image = self.transform(safe_image)
            unsafe_image = self.transform(unsafe_image)

        return safe_image, safe_caption, unsafe_image, unsafe_caption
