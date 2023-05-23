import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from PIL import Image
from os.path import join as opj
import cv2
from tqdm import tqdm

from prompter import load_clip_to_cpu


VOC_CLASSNAMES = [
    # excluding "background"
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
    "tvmonitor"
]


def resize_long_edge(semseg: np.ndarray, size, ignore_index):
    H, W = semseg.shape

    output = np.full((size, size), fill_value=ignore_index, dtype=np.int32)

    if H > W:
        new_H = size
        new_W = int(size * W / H)
        output[:, :new_W] = cv2.resize(
            semseg, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
    else:
        new_H = int(size * H / W)
        new_W = size
        output[:new_H, :] = cv2.resize(
            semseg, (new_W, new_H), interpolation=cv2.INTER_NEAREST)

    return output


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class VOC2012Dataset(data.Dataset):
    def __init__(self, data_root, split_file, emb_folder, ignore_classes=None) -> None:
        super().__init__()
        pairs = []
        for file in tqdm(open(split_file, mode="r").readlines()):
            file = file.rstrip()
            label = os.path.join(data_root, "SegmentationClass", file + ".png")
            label = np.array(Image.open(label))
            for class_index in np.unique(label):
                if class_index == 0 or class_index == 255:
                    continue

                if ignore_classes is not None and class_index in ignore_classes:
                    continue

                pairs.append((file, class_index))

        self.data_root = data_root
        self.pairs = pairs
        self.emb_folder = emb_folder

        self.image_transform = _transform(load_clip_to_cpu("RN50").visual.input_resolution)

        print(len(self))

    def __getitem__(self, index):
        name, class_index = self.pairs[index]
        emb_name = os.path.join(self.emb_folder, name + ".npy")

        semseg_name = os.path.join(
            self.data_root, "SegmentationClass", name + ".png")
        semseg = np.array(Image.open(semseg_name))

        emb = np.load(emb_name)
        emb = torch.from_numpy(emb)[0]

        pos = semseg == class_index
        ignore = semseg == 255
        neg = ~(pos | ignore)
        semseg[neg] = 0
        semseg[pos] = 1

        SIZE = 256
        semseg = resize_long_edge(semseg, SIZE, ignore_index=255)

        image_name = os.path.join(
            self.data_root, "JPEGImages", name + ".jpg")
        image = Image.open(image_name)
        image = self.image_transform(image)

        return emb, (class_index-1), torch.from_numpy(semseg).long(), image

    def __len__(self):
        return len(self.pairs)
