import pathlib
from typing import List, Dict

import torch
import numpy as np

from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import DetrConfig
from transformers import DetrFeatureExtractor
from transformers import DetrForObjectDetection

from dataset import DETRMotiveDataLoader


class DETR:

    def __init__(
            self,
            pretrained: str,
            checkpoint_dir: str,
            checkpoint_frequency: int,
            device: str,
            **kwargs,
        ) -> None:

        self.device = device

        self.config = DetrConfig.from_pretrained(pretrained, **kwargs)

        self.feature_extractor = DetrFeatureExtractor.from_pretrained(
            pretrained, config=self.config, do_resize=False)

        self.model = DetrForObjectDetection.from_pretrained(
            pretrained, config=self.config, ignore_mismatched_sizes=True)

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency

        if self.checkpoint_dir is not None:
            # Create checkpoint directory if it doesn't exist
            self.checkpoint_dir = Path(self.checkpoint_dir)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoint = self.__load_latest_checkpoint()
        else:
            self.checkpoint = {}

        self.model.to(self.device)


    def fit(
            self,
            train_dataloader: DETRMotiveDataLoader,
            val_dataloader: DETRMotiveDataLoader,
            lr: float = 5e-5,
            epochs: int = 1,
            val_steps: int = 10,
        ) -> None:

        # Load saved data from checkpoint if it exists
        lr = float(self.checkpoint.get('lr', lr))
        start_epoch = int(self.checkpoint.get('epoch', 1))
        start_iteration = int(self.checkpoint.get('iteration', 0))
        optimizer_state_dict = self.checkpoint.get('optimizer_state_dict', None)
        scheduler_state_dict = self.checkpoint.get('scheduler_state_dict', None)

        # Initialize optimizer and load state if it exists
        optimizer = AdamW(self.model.parameters(), lr=lr)
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)

        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=50,
            threshold=1e-3, min_lr=1e-6, verbose=True)
        if scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

        self.model.train()

        for epoch in range(start_epoch, epochs + 1):

            iter_val_dataloader = iter(val_dataloader)

            for iteration, batch in enumerate(train_dataloader):

                # Skip data until reached saved iteration number
                if start_iteration > iteration:
                    continue
                start_iteration = 0

                outputs = self.__forward(batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                if iteration % val_steps == 0:

                    if val_dataloader is not None:

                        try:
                            val_batch = next(iter_val_dataloader)
                        except StopIteration:
                            iter_val_dataloader = iter(val_dataloader)
                            val_batch = next(iter_val_dataloader)

                        val_outputs = self.__forward(val_batch)
                        val_loss = val_outputs.loss

                        scheduler.step(val_loss)

                    else:
                        val_loss = np.nan

                    print('Epoch {}/{} - Batch {}/{} - Train Loss {:.2f} - Val Loss {:.2f}'.format(
                        epoch, epochs, iteration, len(train_dataloader), loss, val_loss))

                if iteration % self.checkpoint_frequency == 0 and self.checkpoint_dir is not None:

                    state_dict = {
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'lr': lr,
                        'epoch': epoch,
                        'iteration': iteration,
                    }
                    filepath = self.checkpoint_dir / f"model_{state_dict['epoch']}_{state_dict['iteration']}.pt"
                    self.__save_checkpoint(state_dict, filepath)

                if iteration >= len(train_dataloader):
                    break
            
            if self.checkpoint_dir is not None:

                state_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'lr': lr,
                    'epoch': epoch + 1,
                    'iteration': 0,
                }
                filepath = self.checkpoint_dir / f"model_{state_dict['epoch']}_{state_dict['iteration']}.pt"
                self.__save_checkpoint(state_dict, filepath)


    def predict(
            self,
            inputs: Dict,
            threshold: float = 0.9,
        ) -> object:

        self.model.eval()

        outputs = self.__forward(inputs)

        predictions = self.__postprocess(inputs, outputs, threshold)

        return predictions


    def save(self, save_directory: str) -> None:

        self.config.save_pretrained(save_directory)
        self.feature_extractor.save_pretrained(save_directory)
        self.model.save_pretrained(save_directory)


    def __forward(self, inputs: Dict) -> object:

        pixel_values = inputs['pixel_values'].to(self.device)
        pixel_mask = inputs['pixel_masks'].to(self.device)
        labels = inputs['labels']
        if labels is not None:
            labels = [{k: v.to(self.device) for k, v in t.items()} for t in labels]

        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )

        return outputs


    def __postprocess(
            self,
            inputs: Dict,
            outputs: object,
            threshold: float,
        ) -> List:

        # process predictions and rescale bounding boxes
        target_size = torch.tensor([x[0]['size'] for x in inputs['targets']])
        processed_outputs = self.feature_extractor.post_process(outputs, target_size)

        # Keep only predictions of queries with confidence above threshold
        predictions = []
        for processed_output in processed_outputs:
            keep = torch.logical_and(processed_output['scores'] > threshold, processed_output['labels'] > 0)
            boxes = processed_output['boxes'][keep].cpu().detach().numpy()
            labels = processed_output['labels'][keep].cpu().detach().numpy()
            scores = processed_output['scores'][keep].cpu().detach().numpy()
            categories = np.array([self.model.config.id2label[x] for x in labels])

            prediction = {
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'categories': categories,
            }
            predictions.append(prediction)
        
        return predictions


    def __save_checkpoint(
            self,
            state_dict: Dict,
            filepath: Path
        ) -> None:

        print('Saving checkpoint {} ... '.format(filepath), end='', flush=True)
        torch.save(state_dict, filepath)
        print('DONE')


    def __load_latest_checkpoint(self) -> Dict:
        
        state_dict = {}
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        if len(checkpoints):
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_ctime)
            print('Loading checkpoint {} ... '.format(latest_checkpoint), end='', flush=True)
            state_dict = torch.load(latest_checkpoint)
            print('DONE')

        return state_dict
