# Dataset for Motive AI Challenge dataset in DETR format
# Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb

from typing import Dict

import torch
import numpy as np

from .motive_dataset import MotiveDataset


class DETRMotiveDataset(MotiveDataset):

    def __init__(
            self,
            root: str,
            mode: str,
            feature_extractor: object,
        ) -> None:

        super(DETRMotiveDataset, self).__init__(root, mode)
        self._feature_extractor = feature_extractor


    def __getitem__(self, index: int) -> Dict:
        
        # Get image and annotations/metadata 
        image, target = super(DETRMotiveDataset, self).__getitem__(index)

        # prepare image for DETR feature extractor
        image_detr = np.moveaxis(image, -1, 0)   # (H, W, C) --> (C, H, W)
        image_detr = image_detr / 255.

        # prepare annotations for DETR feature extractor
        labels = None
        if 'bbox' in target[0]:             # if annotation exists
            annotations = []
            for ann in target:
                ann["iscrowd"] = 0
                ann["area"] = ann["bbox"][2] * ann["bbox"][3]
                annotations.append(ann)
            labels = {'image_id': self.ids[index], 'annotations': annotations}

        # preprocess image and target
        # converting target to DETR format,
        # resizing + normalization of both image and target.
        encoding = self._feature_extractor(
            images=image_detr, annotations=labels, return_tensors="pt")

        pixel_values = encoding["pixel_values"].squeeze() # remove batch dimension
        pixel_mask = encoding['pixel_mask'].squeeze() # remove batch dimension
        if labels is not None:
            labels = encoding["labels"][0] # remove batch dimension
        else:
            labels = {
                'class_labels': torch.LongTensor([]),
                'boxes': torch.FloatTensor([]),
            }

        obj = {
            "image" : image,
            "target" : target,
            "pixel_values" : pixel_values,
            "pixel_mask" : pixel_mask,
            "labels" : labels,
        }

        return obj
