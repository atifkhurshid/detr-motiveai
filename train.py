import torch

import config as cfg

from pathlib import Path

from model import DETR
from dataset import DETRMotiveDataset
from dataset import DETRMotiveDataLoader


if __name__ == "__main__":

    print('Setting device: ', end='', flush=True)
    device = torch.device("cuda" if cfg.cuda and torch.cuda.is_available() else "cpu")
    print(device)

    # Make directory if not exists
    model_save_dir = Path(cfg.model_save_dir)
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print('Checking if model is already trained: ', end='', flush=True)
    if not any(model_save_dir.iterdir()):
        print('Saved model not found')

        print('Initializing model: ', end='', flush=True)
        detr = DETR(
            pretrained=cfg.pretrained,
            checkpoint_dir=cfg.checkpoint_dir,
            checkpoint_frequency=cfg.checkpoint_frequency,
            device=device,
            id2label=cfg.id2label,
            label2id=cfg.label2id,
            num_labels=cfg.num_labels,
        )
        print(cfg.pretrained)

        print('Loading dataset: ', end='', flush=True)

        dataset = DETRMotiveDataset(
            root=cfg.data_path, 
            mode="TRAIN",
            feature_extractor=detr.feature_extractor,
        )
        
        val_size = int(cfg.val_fraction * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = DETRMotiveDataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        val_dataloader = DETRMotiveDataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)

        print(f'{train_size} training and {val_dataset} validation images')

        print('Freezing non-classifier layers: ', end='', flush=True)
        detr.model.requires_grad_(False)
        detr.model.class_labels_classifier.requires_grad_(True)
        print('Done')

        print('Training ... ')
        detr.fit(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            lr=cfg.lr,
            epochs=cfg.epochs,
            val_steps=cfg.val_steps,
        )

        print('Saving model: ', end='', flush=True)
        detr.save(cfg.model_save_dir)
        print(cfg.model_save_dir)

    else:
        print('Saved model found')

    print('Exiting')