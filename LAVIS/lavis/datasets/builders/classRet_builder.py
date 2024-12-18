"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.classRet_datasets import (
    ClassRetDataset,
    ClassRetEvalDataset,
)

from lavis.common.registry import registry
@registry.register_builder("artpedia_classRet")
class ArtpediaClassRetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClassRetDataset
    eval_dataset_cls = ClassRetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpedia_classRet.yaml"}

@registry.register_builder("artpediaLong_classRet")
class ArtpediaClassRetBuilder(BaseDatasetBuilder):
    train_dataset_cls = ClassRetDataset
    eval_dataset_cls = ClassRetEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpediaLong_classRet.yaml"}
