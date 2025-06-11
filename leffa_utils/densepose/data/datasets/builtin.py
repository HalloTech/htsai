# Copyright (c) Facebook, Inc. and its affiliates.
from .chimpnsee import register_dataset as register_chimpnsee_dataset
from .coco import BASE_DATASETS as BASE_COCO_DATASETS
from .coco import DATASETS as COCO_DATASETS
from .coco import register_datasets as register_coco_datasets
from .lvis import DATASETS as LVIS_DATASETS
from .lvis import register_datasets as register_lvis_datasets
from detectron2.data import DatasetCatalog

DEFAULT_DATASETS_ROOT = "datasets"

# Avoid re-registering if already done
if "densepose_coco_2014_train" not in DatasetCatalog.list():
    register_coco_datasets(COCO_DATASETS, DEFAULT_DATASETS_ROOT)
    register_coco_datasets(BASE_COCO_DATASETS, DEFAULT_DATASETS_ROOT)
    register_lvis_datasets(LVIS_DATASETS, DEFAULT_DATASETS_ROOT)
    register_chimpnsee_dataset(DEFAULT_DATASETS_ROOT)  # pyre-ignore[19]
