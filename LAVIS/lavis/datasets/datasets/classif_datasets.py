"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
from collections import OrderedDict

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import torch


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        visual_key = "image" if "image" in ann else "video"

        return OrderedDict(
            {
                "file": ann[visual_key],
                "caption": ann["caption"],
                visual_key: sample[visual_key],
            }
        )


class ClassifDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.class_labels = None
        self.classnames= self._build_classnames()

    def __getitem__(self, index):
        ann = self.annotation[index]

        caption = self.text_processor(ann["caption"])
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label=[]

        return {
            "text_input": caption,
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label)
        }

    def _build_class_labels(self):
        pass

    def _build_classnames(self):
        if "label" in self.annotation[0]:
            labels = self.annotation[0]["label"]
        else:
            labels=[]
        if isinstance(labels, list):
            return list(set(labels))
        else:
            return [0,1]

class ClassifLongDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.class_labels = None
        self.classnames= self._build_classnames()

    def __getitem__(self, index):
        ann = self.annotation[index]

        caption = self.text_processor(ann["caption"])
        caption_annotation = self.text_processor(ann["caption_annotation"])
                
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label = []

        return {
            "text_input": caption,
            "text_annotation": caption_annotation,
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label)
        }

    def _build_class_labels(self):
        pass

    def _build_classnames(self):
        if "label" in self.annotation[0]:
            labels = self.annotation[0]["label"]
        else:
            labels=[]
        if isinstance(labels, list):
            return list(set(labels))
        else:
            return [0,1]



class ClassifEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.class_labels = None
        self.classnames= self._build_classnames()

        self.text = []

        for ann in self.annotation:
            for caption in ann["caption"]:
                self.text.append(self.text_processor(caption))

    def __getitem__(self, index):
        ann = self.annotation[index]

        caption = self.text_processor(ann["caption"])
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label=[]

        return {
            "text_input": caption,
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label)
        }

    def _build_class_labels(self):
        pass

    def _build_classnames(self):
        if "label" in self.annotation[0]:
            labels = self.annotation[0]["label"]
        else:
            labels=[]
        if isinstance(labels, list):
            return list(set(labels))
        else:
            return [0,1]

class ClassifLongEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """

        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.class_labels = None
        self.classnames= self._build_classnames()

        self.text = []

        for ann in self.annotation:
            for caption in ann["caption"]:
                self.text.append(self.text_processor(caption))

    def __getitem__(self, index):
        ann = self.annotation[index]

        caption = self.text_processor(ann["caption"])
        caption_annotation = self.text_processor(ann["caption_annotation"])
                
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label = []

        return {
            "text_input": caption,
            "text_annotation": caption_annotation,
            "instance_id": ann["instance_id"],
            "label": torch.tensor(label)
        }

    def _build_class_labels(self):
        pass

    def _build_classnames(self):
        if "label" in self.annotation[0]:
            labels = self.annotation[0]["label"]
        else:
            labels=[]
        if isinstance(labels, list):
            return list(set(labels))
        else:
            return [0,1]


