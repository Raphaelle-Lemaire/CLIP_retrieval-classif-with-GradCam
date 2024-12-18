"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from lavis.datasets.datasets.retrieval_datasets import (
    RetrievalDataset,
    RetrievalEvalDataset,
    VideoRetrievalDataset,
    VideoRetrievalEvalDataset,
)

from lavis.common.registry import registry


@registry.register_builder("msrvtt_retrieval")
class MSRVTTRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults_ret.yaml"}


@registry.register_builder("didemo_retrieval")
class DiDeMoRetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoRetrievalDataset
    eval_dataset_cls = VideoRetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/didemo/defaults_ret.yaml"}


@registry.register_builder("coco_retrieval")
class COCORetrievalBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/coco/defaults_ret.yaml"}

@registry.register_builder("artpediaVC_retrieval")
class ArtpediaVslCtxBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpedia_retrieval.yaml"}
    
@registry.register_builder("artpediaVC_retrieval_vslVAL")
class ArtpediaVslCtxBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpedia_retrieval_vslVAL.yaml"}
    
@registry.register_builder("artpediaVC_retrieval_ctxVAL")
class ArtpediaVslCtxBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/artpedia_retrieval_ctxVAL.yaml"}

@registry.register_builder("artpediaV_only_retrieval")
class ArtpediaVslOnlyBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/visuel_retrieval.yaml"}

@registry.register_builder("artpediaC_only_retrieval")
class ArtpediaCtxOnlyBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/context_retrieval.yaml"}


@registry.register_builder("artpediaVCLong_retrieval")
class ArtpediaVslCtxBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/texteLong_retrieval.yaml"}

@registry.register_builder("artpediaVLong_only_retrieval")
class ArtpediaVslOnlyBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/visuelLong_retrieval.yaml"}

@registry.register_builder("artpediaCLong_only_retrieval")
class ArtpediaCtxOnlyBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/artpedia/contextLong_retrieval.yaml"}
    
    
@registry.register_builder("flickr30k")
class Flickr30kBuilder(BaseDatasetBuilder):
    train_dataset_cls = RetrievalDataset
    eval_dataset_cls = RetrievalEvalDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/flickr30k/defaults.yaml"}
