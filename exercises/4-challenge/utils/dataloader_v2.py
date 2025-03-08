import os, json
import torch
from torchvision import tv_tensors
from torchvision.datasets import VOCDetection
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
    )

def class_to_num(class_str):
    for idx, string in enumerate(CLASSES):
        if string == class_str: return idx

def num_to_class(number):
    for idx, string in enumerate(CLASSES):
        if idx == number: return string
    return 'none'

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, root, year='2012', image_set='train', transform=None, only_person=False):
        # Check if dataset exists
        self.dataset_path = os.path.join(root, "VOCdevkit", f"VOC{year}")
        self.only_person = only_person
        self.image_set = image_set
        self.transform = transform
        if not os.path.exists(self.dataset_path):
            print(f"Dataset not found at {self.dataset_path}. Downloading...")
            self.dataset = VOCDetection(root, year=year, image_set=self.image_set, download=True)
        else:
            print(f"Dataset already exists at {self.dataset_path}. Skipping download.")
            self.dataset = VOCDetection(root, year=year, image_set=self.image_set, download=False)

        if self.only_person:
            if os.path.exists("data/person_indices.json"):
                with open("data/person_indices.json", "r") as fd: indices = list(json.load(fd)[self.image_set])
                self.dataset = torch.utils.data.Subset(self.dataset, indices)
            else:
                raise FileNotFoundError("person_indices.json not found. Dataset cannot be filtered.")

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
        target = target['annotation']['object']

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        bboxes = []
        names = []
        for item in target:
            x0 = int(item['bndbox']['xmin'])*scale + diff_width//2
            w = (int(item['bndbox']['xmax']) - int(item['bndbox']['xmin']))* scale
            y0 = int(item['bndbox']['ymin'])*scale + diff_height//2
            h = (int(item['bndbox']['ymax']) - int(item['bndbox']['ymin'])) * scale
            name = item['name']

            names.append(name)

            bbox = [x0, y0, w, h]
            bboxes.append(bbox)

        # Convert bounding boxes list to torchvision.BoundingBoxes
        bboxes = tv_tensors.BoundingBoxes(bboxes, format="XYWH", canvas_size=(height, width))

        # Augment Image and Bounding Boxes
        if self.transform:
            image, bboxes = self.transform(image, bboxes)

        target_vectors = []
        for idx, bbox in enumerate(bboxes):
            target_vector = [(int(bbox[0]) + int(bbox[2])/2) / width,
                            (int(bbox[1]) + int(bbox[3])/2) / height,
                            int(bbox[2])/width,
                            int(bbox[3])/height,
                            1.0,
                            class_to_num(names[idx])]

            if self.only_person:
                if target_vector[5] == class_to_num("person"):
                    target_vector[5] = 0.0
                    target_vectors.append(target_vector)
            else:
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