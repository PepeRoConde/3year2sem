from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import torch

class RandomLargestSquareCrop(object):
    def __init__(self):
        pass
    def __call__(self, img):

        width, height = img.size
        min_dim = min(width, height)
        
        if width > height:
            left = random.randint(0, width - min_dim)
            top = 0
        elif height > width:
            left = 0
            top = random.randint(0, height - min_dim)
        else:
            left = 0
            top = 0
        
        img = img.crop((left, top, left + min_dim, top + min_dim))
        
        return img


class ShipDataset(Dataset):
    def __init__(self, root_dir, train=True, dataAugmentation=False, docked=False, transform=None, train_ratio=0.8):
        """
        Args:
            root_dir (string): Directory containing the image folders
            dataAugmentation (bool): Whether to apply data augmentation and include cropped ship images
            docked (bool): Whether to include docked status in labels
            transform (callable, optional): Optional additional transform to be applied
        """
        self.root_dir = root_dir
        self.train = train
        self.dataAugmentation = dataAugmentation
        self.docked = docked
        
        self.base_transform = transforms.Compose([
            RandomLargestSquareCrop(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),    
            transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=7, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2),
            transforms.RandomGrayscale(p=0.15), 
        ])

        self.crop_transform = transforms.Compose([
            RandomLargestSquareCrop(),
            transforms.Resize((350,350)),
            transforms.RandomCrop(224),  # Adjust size as needed
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.additional_transform = transform

        # Collect image paths and labels
        self.images = []
        self.labels = []

        # codigo de etiquetas
        # 0 - no barco
        # 1 - barco (undocked)
        # 2 - barco (docked)

        
        # 1. No-ship images (images/ns-xxx.jpg)
        no_ship_dir = os.path.join(root_dir, "images")
        for filename in os.listdir(no_ship_dir):
            if filename.startswith("ns-") and filename.endswith(".jpg"):
                img_path = os.path.join(no_ship_dir, filename)
                if self.docked:
                    self.images.append((img_path, "no_ship"))
                    self.labels.append(0)
                else:
                    self.images.append((img_path, "no_ship"))
                    self.labels.append(0)

        # 2. Cropped-ship images (only if dataAugmentation is True)
        
        if self.dataAugmentation:
            cropped_dir = os.path.join(root_dir, "cropedImages")
            for filename in os.listdir(cropped_dir):
                if filename.startswith("s-") and filename.endswith(".jpg"):
                    img_path = os.path.join(cropped_dir, filename)
                    if self.docked:
                        is_docked = 1 if filename.endswith("undocked.jpg") else 2
                        self.images.append((img_path, "cropped_ship"))
                        self.labels.append(is_docked)
                    else:
                        self.images.append((img_path, "cropped_ship"))
                        self.labels.append(1)

        # 3. Regular ship images (images/s-xxx-docked.jpg)
        for filename in os.listdir(no_ship_dir):
            if filename.startswith("s-") and filename.endswith(".jpg"):
                img_path = os.path.join(no_ship_dir, filename)
                if self.docked:
                    is_docked = 1 if filename.endswith("undocked.jpg") else 2
                    self.images.append((img_path, "regular_ship"))
                    self.labels.append(is_docked)
                else:
                    self.images.append((img_path, "regular_ship"))
                    self.labels.append(1)

        no_ship_count = sum(1 for img, type in self.images if type == "no_ship")
        cropped_ship_count = sum(1 for img, type in self.images if type == "cropped_ship")
        regular_ship_count = sum(1 for img, type in self.images if type == "regular_ship")
        
        no_ship_split = int(no_ship_count * train_ratio)
        cropped_ship_split = int(cropped_ship_count * train_ratio)
        regular_ship_split = int(regular_ship_count * train_ratio)
        
        # Create indices for each category
        no_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "no_ship"]
        cropped_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "cropped_ship"]
        regular_ship_indices = [i for i, (_, type) in enumerate(self.images) if type == "regular_ship"]
        
        # Select validation indices
        if self.train:
            no_ship_indices = no_ship_indices[:no_ship_split]
            cropped_ship_indices = cropped_ship_indices[:cropped_ship_split]
            regular_ship_indices = regular_ship_indices[:regular_ship_split]
        else:
            no_ship_indices = no_ship_indices[no_ship_split:]
            cropped_ship_indices = cropped_ship_indices[cropped_ship_split:]
            regular_ship_indices = regular_ship_indices[regular_ship_split:]
        
        # Combine all indices
        valid_indices = no_ship_indices + cropped_ship_indices + regular_ship_indices
        
        # Filter images and labels
        self.images = [self.images[i] for i in valid_indices]
        self.labels = [self.labels[i] for i in valid_indices]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, img_type = self.images[idx]
        label = self.labels[idx]

        # Load the image
        image = Image.open(img_path).convert('RGB')

        # Apply appropriate transforms
        if self.dataAugmentation:
            if img_type == "no_ship" or img_type == "regular_ship":
                # Apply crop transform for no-ship and regular-ship images
                image = self.crop_transform(self.augmentation_transform(image))
            else:
                # Cropped ship images already processed
                image = self.base_transform(self.augmentation_transform(image))
        else:
            # No data augmentation, just apply base transform
            image = self.base_transform(image)

        # Apply any additional transforms if provided
        if self.additional_transform:
            image = self.additional_transform(image)

        return image, label

    def plot_grid(self, cols=8, rows=5):
        figure = plt.figure(figsize=(cols*2, rows*2))
        view = random.permutation(cols * rows)
        
        indices_aleatorios = random.choice(np.arange(len(self)),cols * rows)
        for i, j in zip(range(1, cols * rows + 1), indices_aleatorios):
            sample, label = self[j]
            sample = torch.Tensor.permute(sample,(1,2,0)).numpy()
            sample -= np.min(sample)
            sample /= np.max(sample)
        
            figure.add_subplot(rows, cols, i)
            plt.title(f'Clase {label}')
            plt.axis("off")
            plt.imshow(sample, cmap="gray")
        plt.show();
