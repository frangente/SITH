# Copyright 2026 Francesco Gentile.
# SPDX-License-Identifier: MIT

"""Datasets for X-CLIP."""

from ._caltech101 import Caltech101Dataset
from ._cub import CUB200Dataset
from ._dataset import ClassificationDataset
from ._dtd import DescribableTexturesDataset
from ._euro_sat import EuroSATDataset
from ._fgvc_aircraft import FGVCAircraftDataset
from ._flowers102 import Flowers102Dataset
from ._food101 import Food101Dataset
from ._nico import NICODataset
from ._oxford_iiit_pet import OxfordIIITPetDataset
from ._split import split_dataset
from ._stanford_cars import StanfordCarsDataset
from ._visu import ViSUDataset
from ._waterbirds import BinaryWaterbirdsDataset, WaterbirdsDataset

__all__ = [
    "BinaryWaterbirdsDataset",
    "CUB200Dataset",
    "Caltech101Dataset",
    "ClassificationDataset",
    "DescribableTexturesDataset",
    "EuroSATDataset",
    "FGVCAircraftDataset",
    "Flowers102Dataset",
    "Food101Dataset",
    "NICODataset",
    "OxfordIIITPetDataset",
    "StanfordCarsDataset",
    "ViSUDataset",
    "WaterbirdsDataset",
    "split_dataset",
]
