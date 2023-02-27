import os

import torch.utils.data
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset


# Define Global
custom_transformer = transforms.Compose(
    [
        models.ResNet152_Weights.IMAGENET1K_V2.transforms()
    ]
)


class FolderDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.all_images = os.listdir(root)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.root, self.all_images[idx])
        image = Image.open(img_loc).convert("RGB")

        return custom_transformer(image), custom_transformer(image)


class TrainTestFullLoader:
    def __init__(self, root, train_size, test_size, batch_size):
        self.fulldataset = FolderDataset(root)
        self.train_size = train_size
        self.test_size = test_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            self.fulldataset,
            [
                self.train_size,
                self.test_size
            ]
        )
        self.batch_size = batch_size

    def get_train_test_full_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
        full_loader = torch.utils.data.DataLoader(self.fulldataset, batch_size=self.batch_size)

        return train_loader, test_loader, full_loader
