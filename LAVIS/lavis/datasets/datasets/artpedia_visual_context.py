import os
from collections import OrderedDict
from lavis.datasets.datasets.base_dataset import BaseDataset
from lavis.datasets.datasets.image_text_pair_datasets import __DisplMixin
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


class Visuel_contextDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test

        Load annotations from annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


    def __getitem__(self, index):
        ann = self.annotation[index]
        captions, dict_ctx_vsls=[], []
        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        for i in range (len(ann["texts"])):
            caption, dict_ctx_vsl= self.split_txt(ann["texts"][i])
            captions.append(self.text_processor(caption))
            dict_ctx_vsls.append(dict_ctx_vsl)

        return {
            "image": image,
            "text_input":{"vsl": captions}
        }

    def split_txt(self, txt):
        txtLong = []
        txtClass = []
        txtSplit = txt.replace("[SEP]", ". ").replace("..", ".").split(". ")
        valeur = "None"
        for split in txtSplit:
            if split != "":
                if split.startswith("[CTX]"):
                    valeur = "ctx"      
                    txtLong.append(split.replace("[CTX]", "").strip())
                elif split.startswith("[VSL]"):
                    valeur = "vsl"
                    txtLong.append(split.replace("[VSL]", "").strip())
                else:
                    txtLong.append(split.strip())
                txtClass.append(valeur)
        return '.'.join(txtLong), txtClass
        
class Visuel_contextEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test

        Load annotations from annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)


    def __getitem__(self, index):
        ann = self.annotation[index]
        captions=[]
        image_path = os.path.join(self.vis_root, ann["img_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        for i in range (len(ann["texts"])):
            caption = self.split_txt(ann["texts"][i])
            captions.append(self.text_processor(caption))

        return {
            "image": image,
            "text_input": captions
        }

    def split_txt(self, txt):
        txtLong = []
        txtSplit = txt.split("[SEP]")
        for split in txtSplit:
            if split.startswith("[CTX]"):
                txtLong.append(split.replace("[CTX]", "").strip())
            elif split.startswith("[VSL]"):
                txtLong.append(split.replace("[VSL]", "").strip())
        
        return ' '.join(txtLong)
