import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dir = 'Motive AI Challenge - Public Share/data/train/'
annotations_file = dir + 'train_gt.json'

with open(annotations_file) as f:
    data = json.load(f)

images = pd.DataFrame(data['images'])
annotations = pd.DataFrame(data['annotations'])
categories = pd.DataFrame(data['categories'])


img_name = images['file_name'][0]
bbox = annotations['bbox'][0]
print(bbox)

img = cv2.imread(dir + 'train_images/' + img_name)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color=(255, 0, 0), thickness=10)

plt.imshow(img)
plt.show()

