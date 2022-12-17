# Config file

# System

cuda = True


# Model Configuration

pretrained = "facebook/detr-resnet-50"
id2label = {0: 'N/A', 1:'Car', 2: 'Truck', 3: 'StopSign', 4: 'traffic_lights'}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

checkpoint_dir = "./checkpoint"
checkpoint_frequency = 1000

model_save_dir = "./trained"

# Dataset

data_path = "./Motive AI Challenge - Public Share/data/"
categories = [{'id': id, 'name': name, 'supercategory': 'none'} for id, name in id2label.items()]
val_fraction = 0.1
batch_size = 4


# Training

lr = 1e-4
epochs = 1
val_steps = 10

# Inference

threshold = 0.9
