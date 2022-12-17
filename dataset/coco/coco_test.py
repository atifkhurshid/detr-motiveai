# COCO Interface for reading test images without annotations
# Adapted from cocodataset/cocoapi/PythonAPI/pycocotools/coco.py

import os
import time


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class COCOTest:

    def __init__(self, images_dir=None):
        """
        """
        # load dataset
        self.imgs = dict()
        if not images_dir == None:
            print('reading filenames from images_dir...')
            s = time.time()
            self._image_filenames = list(os.listdir(images_dir))
            print('Done (t={:0.2f}s)'.format(time.time()- s))
            self.createIndex()

    def createIndex(self):
        # create index
        print('creating index...')
        imgs = {}
        for i, name in enumerate(self._image_filenames):
            img = {
                "file_name" : name,
                "id" : i,
                "height" : 720,
                "width" : 1280,
            }
            imgs[img['id']] = img

        print('index created!')

        # create class members
        self.imgs = imgs

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]