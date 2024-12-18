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


class ClassRetDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        self.class_labels = None
        self.classnames= self._build_classnames()

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __getitem__(self, index):

        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label=[]
        if not isinstance(label, list):
            label=[label]
        label+= [2] * (60 - len(label))
        #print("ClassRetDataset", caption, "beulelele", ann["caption"])
        
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
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



class ClassRetEvalDataset(BaseDataset, __DisplMixin):
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
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann["image"])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                self.text.append(self.text_processor(caption))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1

    def __getitem__(self, index):
        image_path = os.path.join(self.vis_root, self.annotation[index]["image"])
        image = Image.open(image_path).convert("RGB")
        caption=""
        for phrase in self.annotation[index]["caption"]:
            caption+=(self.text_processor(phrase))
            caption+="[FIN]"
        image = self.vis_processor(image)
        if "label" in self.annotation[index]:
            label = self.annotation[index]["label"]
        else:
            label=[]
        if not isinstance(label, list):
            label=[label]
        label+= [2] * (60 - len(label))

        return {"image": image, "text_input": caption,"index": index, "label": torch.tensor(label),"instance_id": self.annotation[index]["instance_id"]}
        
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


