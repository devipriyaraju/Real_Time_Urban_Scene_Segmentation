"""Cityscapes data loading and preprocessing used for Experiment A."""
from pathlib import Path
import glob
import os
import random
import cv2
import numpy as np
import torch
from torch.utils import data

CLASS_NAMES = ["road","sidewalk","building","wall","fence","pole","traffic light",
               "traffic sign","vegetation","terrain","sky","person","rider","car",
               "truck","bus","train","motorcycle","bicycle"]

LABEL_MAPPING = {-1:255, 0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
                 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255,
                 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11,
                 25:12, 26:13, 27:14, 28:15, 29:255, 30:255, 31:16, 32:17, 33:18}

CLASS_WEIGHTS = torch.FloatTensor([
    0.8373,0.918,0.866,1.0345,1.0166,0.9969,0.9754,1.0489,0.8786,1.0023,
    0.9539,0.9843,1.1116,0.9037,1.0865,1.0955,1.0865,1.1529,1.0507
])
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
_YK, _XK = 6, 6

def build_pairs(images_dir, labels_dir, split):
    images_dir, labels_dir = str(images_dir), str(labels_dir)
    imgs = sorted(glob.glob(os.path.join(images_dir, split, "*", "*_leftImg8bit.png")))
    pairs, missing = [], 0
    for ip in imgs:
        name = os.path.basename(ip).replace("_leftImg8bit.png", "")
        city = os.path.basename(os.path.dirname(ip))
        lp = os.path.join(labels_dir, split, city, name + "_gtFine_labelIds.png")
        if os.path.isfile(lp):
            pairs.append((ip, lp))
        else:
            missing += 1
    return pairs, missing

class CityscapesDataset(data.Dataset):
    def __init__(self, pairs, crop_size=(1024,1024), base_size=2048, scale_factor=16,
                 multi_scale=True, flip=True, ignore_label=255, bd_dilate=4):
        self.pairs = pairs
        self.crop_size = tuple(crop_size)
        self.base_size = base_size
        self.scale_factor = scale_factor
        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label
        self.bd_dilate = bd_dilate

    def __len__(self): return len(self.pairs)

    def convert_label(self, label, inverse=False):
        temp = label.copy()
        if inverse:
            for v, k in LABEL_MAPPING.items(): label[temp == k] = v
        else:
            for k, v in LABEL_MAPPING.items(): label[temp == k] = v
        return label

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1] / 255.0
        image -= MEAN
        image /= STD
        return image

    def pad_image(self, img, h, w, size, pad):
        ph, pw = max(size[0]-h, 0), max(size[1]-w, 0)
        if ph > 0 or pw > 0:
            img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=pad)
        return img

    def rand_crop(self, image, label, edge):
        h, w = image.shape[:-1]
        image = self.pad_image(image, h, w, self.crop_size, (0.,0.,0.))
        label = self.pad_image(label, h, w, self.crop_size, (self.ignore_label,))
        edge = self.pad_image(edge, h, w, self.crop_size, (0.,))
        nh, nw = label.shape
        x = random.randint(0, nw-self.crop_size[1])
        y = random.randint(0, nh-self.crop_size[0])
        return (image[y:y+self.crop_size[0], x:x+self.crop_size[1]],
                label[y:y+self.crop_size[0], x:x+self.crop_size[1]],
                edge[y:y+self.crop_size[0], x:x+self.crop_size[1]])

    def multi_scale_aug(self, image, label, edge, rand_scale=1):
        long_size = int(self.base_size*rand_scale + 0.5)
        h, w = image.shape[:2]
        if h > w:
            nh, nw = long_size, int(w*long_size/h + 0.5)
        else:
            nw, nh = long_size, int(h*long_size/w + 0.5)
        image = cv2.resize(image, (nw,nh), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (nw,nh), interpolation=cv2.INTER_NEAREST)
        edge = cv2.resize(edge, (nw,nh), interpolation=cv2.INTER_NEAREST)
        return self.rand_crop(image, label, edge)

    def _edge(self, label):
        edge = cv2.Canny(label, 0.1, 0.2)
        kernel = np.ones((self.bd_dilate,self.bd_dilate), np.uint8)
        edge = edge[_YK:-_YK, _XK:-_XK]
        edge = np.pad(edge, ((_YK,_YK),(_XK,_XK)), mode="constant")
        return (cv2.dilate(edge, kernel, iterations=1) > 50) * 1.0

    def gen_sample(self, image, label):
        edge = self._edge(label)
        if self.multi_scale:
            rand_scale = 0.5 + random.randint(0,self.scale_factor)/10.0
            image, label, edge = self.multi_scale_aug(image,label,edge,rand_scale)
        image = self.input_transform(image)
        label = np.asarray(label).astype(np.uint8)
        image = image.transpose((2,0,1))
        if self.flip:
            f = np.random.choice(2)*2-1
            image, label, edge = image[:,:,::f], label[:,::f], edge[:,::f]
        return image, label, edge

    def __getitem__(self, index):
        ip, lp = self.pairs[index]
        image = cv2.imread(ip, cv2.IMREAD_COLOR)
        label = cv2.imread(lp, cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise FileNotFoundError(f"Could not read image/label: {ip} | {lp}")
        label = self.convert_label(label)
        if self.multi_scale or self.flip:
            image, label, edge = self.gen_sample(image,label)
        else:
            edge = self._edge(label)
            image = self.input_transform(image).transpose((2,0,1))
            label = np.asarray(label).astype(np.uint8)
        return image.copy(), label.copy(), edge.copy()
