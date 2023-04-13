import numpy as np
import torch
from torch.utils import data
from torchvision.datasets import CocoDetection


class PidrayDataset(data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        """
        Args:
            root (String): dataset folder address
            annotation (String): annotation files address
            transforms (torch.transforms, optional): Transformer for the input data. Defaults to None.
        """
        self.root = root
        self.annotation = annotation
        self.transforms = transforms
        self.coco = CocoDetection(root=root, annFile=annotation, transform=transforms)
    

    def __getitem__(self, index):
        image, target = self.coco[index]
        bboxes = []
        labels = []

        for obj in target:
            # Convert class label to index
            label = self.class_map[obj['category_id']]
            bbox = obj['bbox']
            # COCO format: [x, y, width, height]
            # Convert to PyTorch format: [ymin, xmin, ymax, xmax]
            ymin, xmin, ymax, xmax = bbox[1], bbox[0], bbox[1] + bbox[3], bbox[0] + bbox[2]
            bboxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.tensor([target[0]['image_id']])
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': areas,
            'iscrowd': torch.zeros((len(target),), dtype=torch.int64)
        }

        return image, target


    def __len__(self):
        return len(self.coco)


def dataloader(root, annotation, train=False):
    """This function loads and prepares the data for the PyTorch model. 
    If the train flag is True, the dataset is split into training and validation sets using torch.utils.data.random_split. 
    The training dataset is passed to a PyTorch DataLoader train_loader, which batches the data into groups of 64 and shuffles them.
    Similarly, the validation dataset is passed to val_loader.
    When the train flag is False, the test dataset is loaded into a PyTorch DataLoader called test_loader,
    which is not shuffled since this is not needed for testing.

    Args:
        root (String): The root directory of the dataset
        annotation (String): The path to the annotation file
        train (bool, optional): A boolean flag that specifies whether to prepare the data for training or testing. Defaults to False.

    Returns:
        torch.Dataloader: processed dataloader
    """
    # Initialize the transforms, only for train dataset
    transforms = transforms.Compose([
        transforms.Resize(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load the dataset
    dataset = PidrayDataset(root=root, annotation=annotation, transforms=transforms)
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if train:
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
        
        return train_loader, val_loader
    
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    return test_loader

