import os
from PIL import Image

## PyTorch
import torch

from torchvision import transforms
from torch.utils.data import Dataset


class DiscretizeTransform:
    def __call__(self, sample):
        return (transforms.functional.to_tensor(sample) * 255).to(torch.int32)

# Now, you can use this custom transform directly



class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        """
        Args:
            image_folder (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.image_folder = image_folder
        self.transform = transform
        self.image_filenames = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')  # Convert to RGB in case images are in different formats

        if self.transform:
            image = self.transform(image)

        return image