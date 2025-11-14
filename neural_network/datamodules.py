import os

from pytorch_lightning import LightningDataModule
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from datasets import CustomDataset
from datasets import lightning_collate_fn


class DataModule(LightningDataModule):
    def __init__(
        self,
        input_size,
        batch_size,
        data_folders,
        split_ratio=10,
        split_seed=42,
        k=1,
        fract = 1.,
        overfit: bool = False
    ):
        """
        Args:
            input_size: The size of the input images (width, height)
            data_folders: The folders containing the dataset
            batch_size: The batch size to use for training
            split_ratio: The number of parts to split the dataset into (default=10)
            split_seed: The seed to use for the random split (default=42)
            k: The k-th fold to use for training (default=1)
        """

        super().__init__()
        self.batch_size = batch_size
        self.k = k

        transform = A.Compose(
            [
                A.Resize(width=input_size[0], height=input_size[1], p=1),
                # A.VerticalFlip(p=0.2),
                # A.HorizontalFlip(p=0.2),
                # A.Normalize(
                #     mean=(0.485, 0.456, 0.406),
                #     std=(0.229, 0.224, 0.225),
                #     max_pixel_value=255.0,
                #     p=1.0
                # ),
                ToTensorV2(),
            ]
        )

        self.full_dataset = CustomDataset(data_folders, transform, fract=fract)
        self.kfolds = KFold(n_splits=split_ratio, shuffle=True, random_state=split_seed)
        self.splits = [split for split in self.kfolds.split(self.full_dataset)]
        self.overfit = overfit
        print(f"Dataset size: {len(self.full_dataset)}")

    def setup(self, stage: str):
        train_indices, val_indices = self.splits[self.k]
        self.train_data = Subset(self.full_dataset, train_indices)
        self.val_data = Subset(self.full_dataset, val_indices)
        self.test_data = Subset(self.full_dataset, val_indices) # No test set from KFolds

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=not self.overfit, num_workers=1, collate_fn=lightning_collate_fn, drop_last=True)
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=lightning_collate_fn, drop_last=True)
    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=1, collate_fn=lightning_collate_fn, drop_last=True)