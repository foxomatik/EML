# COCO (Common Objects in COntext) 
# 80 classes. The train/val data has over 200,000 images.


import torch
from torchinfo import summary
import os, json
from PIL import Image

import torchvision
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision import tv_tensors
from torchvision.datasets import CocoDetection

from torchvision import transforms as tf
from pycocotools.coco import COCO

CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    )

def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx

def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'


class COCODatasetPerson(torch.utils.data.Dataset):
    def __init__(self, root, year='2017', image_set='train', transform=None):
        # Check if dataset exists
        self.dataset_path = os.path.join(root, f"{image_set}{year}") # root: ./data/COCO
        self.ann_file = os.path.join(root, f"coco_{image_set}{year}_person.json") # root: ./data/COCO
        self.image_set = image_set
        self.transform = transform
        
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found at {self.dataset_path}.")
            # ERROR
        else:
            print(f"Dataset already exists at {self.dataset_path}.")
            self.dataset = CocoDetection(self.dataset_path, annFile=self.ann_file)

        if self.image_set == 'train':
            self.transform = transform
        elif self.image_set == 'val':
            if transform is not(None):
                self.transform = None
                print("Validation set cannot be augmented, transform is set to None.")
        elif self.image_set == 'trainval':
            self.transform = None
            print("Dataset has to be split. Define the transform member variable for training dataset separately.")
        else:
            raise ValueError(f"Invalid image_set. Must be 'train' or 'val', {self.image_set} was passed")

    
    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, idx):
        image, target = self.dataset[idx]

        num_bboxes = 20
        width, height = 320, 320

        img_width, img_height = image.size

        scale = min(width/ img_width, height/img_height)
        new_width, new_height = int(img_width * scale), int( img_height * scale)

        diff_width, diff_height = width - new_width, height - new_height
        image = F.resize(image, size=(new_height, new_width))
        image = F.pad(image, padding = (diff_width//2,
                                            diff_height//2,
                                            diff_width//2 + diff_width % 2,
                                            diff_height//2 + diff_height % 2))
        # target = target['annotation']['object']

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        bboxes = []
        names = []
        for item in target:
            x0 = (item['bbox'][0])*scale + diff_width//2
            y0 = (item['bbox'][1])*scale + diff_height//2
            w = ((item['bbox'][2])) * scale
            h = ((item['bbox'][3])) * scale
            name = 'person'

            names.append(name)

            bbox = [x0, y0, w, h]
            bboxes.append(bbox)

        # Convert bounding boxes list to torchvision.BoundingBoxes (convert to tensor structure)
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYWH", canvas_size=(height, width))

        # Augment Image and Bounding Boxes
        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        # Convert to YOLO format (x_center, y_center, width, height) - all normalized
        target_vectors = []
        for idx, bbox in enumerate(bboxes):
            target_vector = [((bbox[0]) + (bbox[2])/2) / width,
                            ((bbox[1]) + (bbox[3])/2) / height,
                            (bbox[2])/width,
                            (bbox[3])/height,
                            1.0,
                            class_to_num(names[idx])]

            if target_vector[5] == class_to_num("person"):
                target_vector[5] = 0.0
                target_vectors.append(target_vector)

        target_vectors = list(sorted(target_vectors, key=lambda x: x[2]*x[3]))
        target_vectors = torch.tensor(target_vectors)
        if target_vectors.shape[0] < num_bboxes:
            zeros = torch.zeros((num_bboxes - target_vectors.shape[0], 6))
            zeros[:, -1] = -1
            target_vectors = torch.cat([target_vectors, zeros], 0)
        elif target_vectors.shape[0] > num_bboxes:
            target_vectors = target_vectors[:num_bboxes]

        return image, target_vectors