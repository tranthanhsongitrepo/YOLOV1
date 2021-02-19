import os

import torchvision
from PIL import Image
import torch
import pandas as pd
from torchvision import transforms


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, image_root, label_root, S=7, B=2, C=20, transforms=None):
        self.annotations = pd.read_csv(csv_file)
        self.image_root = image_root
        self.label_root = label_root
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_root, self.annotations.iloc[index, 1])
        image_path = os.path.join(self.image_root, self.annotations.iloc[index, 0])

        label_tensor = torch.zeros([self.S, self.S, self.C + 5])
        boxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_label, x, y, w, h = line.rstrip().split()
                boxes.append([int(class_label), float(x), float(y), float(w), float(h)])

        image = Image.open(image_path)

        if self.transforms is not None:
            image, boxes = self.transforms(image, boxes)

        # Transform box's coordinates so that it's relative to each cell
        for box in boxes:
            class_label, x, y, w, h = box
            grid_x = int(self.S * x)
            grid_y = int(self.S * y)

            if label_tensor[grid_y, grid_x, self.C] != 1:
                label_tensor[grid_y, grid_x, class_label] = 1
                label_tensor[grid_y, grid_x, self.C] = 1

                label_tensor[grid_y, grid_x, self.C + 1] = x * self.S - grid_x
                label_tensor[grid_y, grid_x, self.C + 2] = y * self.S - grid_y
                label_tensor[grid_y, grid_x, self.C + 3] = w * self.S
                label_tensor[grid_y, grid_x, self.C + 4] = h * self.S
                # print(image_path, grid_x, grid_y, class_label)

        
        return image, label_tensor
