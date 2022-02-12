import torch
import numpy as np
import torchvision.transforms as transforms
from numpy import random
import cv2

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        
        return img, boxes, labels
    
class ConvertFromInts():
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32), boxes, labels

class Resize():
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size,self.size))
        
        return image, boxes, labels
    
class ToAbsoluteCoords():
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels
    
class ToTensor():
    def __call__(self, image, boxes=None, labels=None):
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).div(255)
        return image.type(torch.FloatTensor), boxes, labels

class RandomBrightness():
    def __init__(self, delta=32):
        self.delta=delta
        
    def __call__(self, image, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        
        return image, boxes, labels

class RandomMirror():
    def __call__(self, image, boxes=None, labels=None):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes[:, 0], boxes[:, 2] = width - boxes[:, 2], width - boxes[:, 0]
            
        return image, boxes, labels

class SSDAugmentation():
    def __init__(self, size=300, delta=32):
        self.size = size
        self.delta = delta
        self.augment = Compose([
            ConvertFromInts(),
            Resize(self.size),
            ToAbsoluteCoords(),
            RandomMirror(),
            RandomBrightness(self.delta),
            ToTensor(),
        ])

    def __call__(self, img, targets=None, labels=None):
        img, targets, labels = self.augment(img, targets, labels)
        
        return img, targets, labels
    
class BaseTransform():
    def __init__(self, size=300):
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),
            Resize(self.size),
            ToAbsoluteCoords(),
            ToTensor(),
        ])

    def __call__(self, img, targets=None, labels=None):
        img, targets, labels = self.augment(img, targets, labels)
        
        return img, targets, labels