from torch.utils.data import Dataset
import xml.etree.cElementTree as ET
import cv2
import torch
import os.path as osp
import os
import numpy as np

CLASSES = ('background',
           'nine', 'ten', 'jack', 
           'queen', 'king', 'ace')
    
class AnnotationTransform():
    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(
            zip(CLASSES, range(len(CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        res = {'boxes': None, 'labels': None}
        boxes = []
        labels = []
        
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1           
            if not self.keep_difficult and difficult:
                continue                
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
                
            label_idx = self.class_to_ind[name]
            boxes += [bndbox]
            labels += [label_idx]

            res['boxes'] = torch.tensor(boxes, dtype=torch.float)
            res['labels'] = torch.tensor(labels, dtype=torch.int64)

        return res
        
class CardsDataset(Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = AnnotationTransform()
        
        modes = ['train', 'test']
        if mode not in modes:
            print(f'{mode} is not correct, please enter correct mode from: {modes}')
            raise NameError
        self.mode = mode
        
        self.ids = list()
        self._annopath = osp.join('%s', mode+'_anno', '%s.xml')
        self._imgpath = osp.join('%s', mode, '%s.jpg')
        
        files = os.listdir(osp.join(self.root, mode))
        for file in files:
            self.ids.append((self.root, file.split('.')[0]))

    def __getitem__(self, index):
        img_id = self.ids[index]
        
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        target = ET.parse(self._annopath % img_id).getroot()
        target = self.target_transform(target, width, height)       
        
        if self.transform is not None:
            img, target['boxes'], target['labels'] = self.transform(img, target['boxes'], target['labels'])
        
        return img, target

    def __len__(self):
        return len(self.ids) 
    
    def pull_image(self, index):
        img_id = self.ids[index]
        
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)