# Dataloader for Motive AI Challenge dataset in DETR format
# Adapted from https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForObjectDetection_on_custom_dataset_(balloon).ipynb


import torch
import numpy as np

from typing import List, Dict
from torch.utils.data import DataLoader


class DETRMotiveDataLoader:

    def __init__(self) -> None:
        pass

    def __new__(
            cls,
            dataset: object,
            **kwargs: Dict,
        ) -> object:

        dataloader = DataLoader(
            dataset,
            collate_fn=cls._collate_fn,
            **kwargs,
        )

        return dataloader


    def _collate_fn(batch: List) -> Dict:

        images = []
        pixel_values = []
        pixel_mask = []
        targets = []
        labels = []
        for item in batch:
            images.append(item["image"])
            pixel_values.append(item["pixel_values"])
            pixel_mask.append(item["pixel_mask"])
            targets.append(item["target"])
            labels.append(item["labels"])

        batch = {}
        batch['images'] = np.array(images)
        batch['targets'] = targets
        batch['pixel_values'] = torch.stack(pixel_values)
        batch['pixel_masks'] = torch.stack(pixel_mask)
        batch['labels'] = labels

        return batch
