import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection
from dataset import DETRMotiveDataset
from dataset import DETRMotiveDataLoader


# Download pre-processor and model weights (detr-resnet-50 or detr-resnet-101)

feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")


# Load dataset

dataset = DETRMotiveDataset(
    root="./Motive AI Challenge - Public Share/data/", 
    mode="TRAIN",
    feature_extractor=feature_extractor,
)

id = np.random.default_rng().integers(len(dataset))
obj = dataset[id]

print(obj.keys())


# Create dataloader for training

dataloader = DETRMotiveDataLoader(dataset, batch_size=1, shuffle=True)
batch = next(iter(dataloader))
print(len(batch))
print(batch.keys())


# Visualize image and targets

image = obj['image']
target = obj['target']

for item in target:
    label = item["category_id"]
    X, Y, W, H = item["bbox"]
    cv2.rectangle(image, (X, Y), (X+W, Y+H), color=(255, 0, 0), thickness=1)
    cv2.putText(image, str(label), (X, Y-5), fontFace=0, fontScale=0.5,
                color=(255, 0, 0), thickness=1)

plt.imshow(image)
plt.show()

# DETR Inference

outputs = model(
    pixel_values=obj['pixel_values'].unsqueeze(0), # Add batch dimension
    pixel_mask=obj['pixel_mask'].unsqueeze(0), # Add batch dimension
    labels=[obj['labels']], # Add batch dimension
)

# keep only predictions of queries with confidence above threshold (excluding no-object class)

threshold = 0.8
probas = outputs.logits.softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > threshold

# rescale bounding boxes

img_size = obj['labels']['orig_size'].unsqueeze(0) # Add batch dimension
processed_outputs = feature_extractor.post_process(outputs, img_size)[0]
bboxes_scaled = processed_outputs['boxes'][keep]

print(processed_outputs.keys())

#  Plot predictions

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=2))
        cl = p.argmax()
        text = f'{model.config.id2label[cl.item()]}: {p[cl]:0.2f}'
        ax.text(xmin, ymin, text, fontsize=5,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

plot_results(image, probas[keep], bboxes_scaled)
