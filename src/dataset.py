"""File fo handling the dataset for the digit recogniser."""
import numpy as np
from torch.utils.data import Dataset


class DigitRecogniserTraining(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.labels = data[:, 0]
        self.images = data[:, 1:].reshape(-1, 28, 28).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class DigitRecogniserPrediction(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.images = data.reshape(-1, 28, 28).astype(np.uint8)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == '__main__':
    from torchvision import transforms
    import pandas as pd
    data = pd.read_csv('../data/train.csv').to_numpy()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])

    dataset = DigitRecogniserTraining(data, transform)
    print(dataset[0])
    print(dataset[0][0].shape)
    print(dataset[0][1])
    print(len(dataset))
