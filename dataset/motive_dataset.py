# Dataloader for Motive AI Challenge dataset
# Adapted from torchvision.datasets.coco

from typing import Any, Tuple, List, Dict

import os
import numpy as np
from PIL import Image


class MotiveDataset:

    def __init__(
            self,
            root: str,
            mode: str,
        ) -> None:

        self._root = root
        self._mode = mode

        if self._mode == "TRAIN":

            from .coco.coco import COCO

            self._images_path = os.path.join(self._root, 'train', 'train_images')
            self._annotations_path = os.path.join(self._root, 'train', 'train_gt.json')
            self.coco = COCO(self._annotations_path)

        elif self._mode == "TEST":

            from .coco.coco_test import COCOTest

            self._images_path = os.path.join(self._root, "public_test", "test2_images")
            self.coco = COCOTest(self._images_path)

        else:
            assert ("ModeError: mode can only be TRAIN or TEST")

        self.ids = list(sorted(self.coco.imgs.keys()))


    def _load_image(self, id: int) -> Tuple[np.ndarray, Dict]:

        metadata = self.coco.loadImgs(id)[0]
        metadata['size'] = (int(metadata['height']), int(metadata['width']))

        image = Image.open(os.path.join(self._images_path, metadata["file_name"]))
        image = np.asarray(image.convert("RGB"))

        return image, metadata


    def _load_target(self, id: int) -> List[Dict]:

        return self.coco.loadAnns(self.coco.getAnnIds(id))


    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[Dict]]:

        id = self.ids[index]
        image, metadata = self._load_image(id)
        if self._mode == "TRAIN":
            target = self._load_target(id)
            if len(target) == 0:
                target = [{}]
        else:
            target = [{}]
        
        target = [{**x, **metadata} for x in target]
        
        return image, target


    def __len__(self):
        
        return len(self.ids)
