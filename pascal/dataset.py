import os
import numpy as np
import torch
import torch.utils.data as data
from imageio import imread
# from scipy.misc import imresize
from scipy.sparse import csr_matrix
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms as transforms
import glob
import matplotlib.pyplot as plt
import torchvision

class VOC2012(data.Dataset):
    def __init__(self, data_path, split, transform, random_crops=0):
        self.data_path = data_path
        self.transform = transform
        self.random_crops = random_crops
        self.slpit = split
        self.__init_classes()
        self.names, self.labels = self.__dataset_info(split)
    
    def __getitem__(self, index):
        x = imread(self.data_path+'/JPEGImages/'+self.names[index]+'.jpg', pilmode='RGB')
        x = Image.fromarray(x)
        
        scale = np.random.rand()*2+0.25
        w = int(x.size[0]*scale)
        h = int(x.size[1]*scale)
        if min(w,h)<227:
            scale = 227/min(w,h)
            w = int(x.size[0]*scale)
            h = int(x.size[1]*scale)
        if self.random_crops==0:
            x = self.transform(x)
        else:
            crops = []
            for i in range(self.random_crops):
                crops.append(self.transform(x))
            x = torch.stack(crops)
        
        y = self.labels[index]
        return x, y
    
    def __len__(self):
        return len(self.names)

    def __dataset_info(self, split):
        with open(self.data_path+'/ImageSets/Main/'+'train'+'.txt') as f:
            annotations_train = f.readlines()

        with open(self.data_path+'/ImageSets/Main/'+'val'+'.txt') as g:
            annotations_val = g.readlines()

        annotations_train = [n[:-1] for n in annotations_train]
        annotations_val = [n[:-1] for n in annotations_val]

        validation_set = []
        train_set = []

        if split == 'train':
            annotations = annotations_train
        elif split == 'val':
            annotations = annotations_val

        names  = []
        labels = []
        for af in annotations:
            filename = os.path.join(self.data_path,'Annotations',af)
            tree = ET.parse(filename+'.xml')
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.uint16)
            boxes_cl = np.zeros((num_objs), dtype=np.int32)
            
            for ix, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                # Make pixel indexes 0-based
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                
                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes[ix, :] = [x1, y1, x2, y2]
                boxes_cl[ix] = cls

            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)
        
        return np.array(names), np.array(labels).astype(np.float32)
    
    def __init_classes(self):

        #BackGround Class 제거
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes  = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))

class VOC_segmentation(data.Dataset):
    def __init__(self, data_path, split= 'val', transform=None):
        self.base_path = data_path
        self.img_path = os.path.join(self.base_path, 'JPEGImages')
        self.ann_path = os.path.join(self.base_path, 'SegmentationClass')

        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self.split = split
        self.image_list = []
        self.ann_list= []
        _splits_dir = os.path.join(self.base_path, 'ImageSets', 'Segmentation')

        with open(os.path.join(os.path.join(_splits_dir, split + '.txt')), "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            _image = os.path.join(self.img_path, line + ".jpg")
            _ann = os.path.join(self.ann_path, line + ".png")
            assert os.path.isfile(_image)
            assert os.path.isfile(_ann)
            self.image_list.append(_image)
            self.ann_list.append(_ann)

        self.image_list.sort()
        self.ann_list.sort()

        self.transform = transform
        self.num_classes = 21

    def __getitem__(self, index):
        img = Image.open(self.image_list[index])
        ann = Image.open(self.ann_list[index]).convert('RGB').resize((227, 227), Image.NEAREST)
        img = self.transform(img)
        
        mask_label, class_labels = self.encode_segmap(np.array(ann))
        mask_label = torch.from_numpy(mask_label)
        return img, mask_label, class_labels

    def __len__(self):
        return len(self.ann_list)
    def get_pascal_labels(self, ):
        return np.asarray([
                            [0, 0, 0],#0
                            [128, 0, 0],#1
                            [0, 128, 0],#2
                            [128, 128, 0],#3
                            [0, 0, 128],#4
                            [128, 0, 128],#5
                            [0, 128, 128],#6
                            [128, 128, 128],#7
                            [64, 0, 0],#8
                            [192, 0, 0],#9
                            [64, 128, 0],#10
                            [192, 128, 0],#11
                            [64, 0, 128],#12
                            [192, 0, 128],#13
                            [64, 128, 128],#14
                            [192, 128, 128],#15
                            [0, 64, 0],#16
                            [128, 64, 0],#17
                            [0, 192, 0],#18
                            [128, 192, 0],#19
                            [0, 64, 128],#20
                ])

    def visualize(self, rgb):
        rgb = rgb.permute(2, 0, 1)
        utils.save_image(rgb, 'results.jpeg')

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        class_labels = np.zeros(self.num_classes)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            loc = np.where(np.all(mask == label, axis=-1))[:2]
            label_mask[loc] = ii
            if (loc[0].sum() + loc[1].sum()) > 0:
                class_labels[ii] = 1
        #label_mask = label_mask.astype(int)
        return label_mask, np.array(class_labels).astype(np.float32)

    def decode_segmap(self, label_mask):
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

def get_loader(data_name, data_path, split, batch_size):

    shuffle = {'train':True, 'val':False, 'test':False}
    if data_name == 'VOC2012':

        normalize = transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])
        if split == 'train':
            transform = transforms.Compose([
                                    transforms.RandomResizedCrop(227),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
        elif split == 'val':
            transform = transforms.Compose([
                                    transforms.Resize((227, 227)),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
        else: 
            raise NotImplementedError

        data = VOC2012(data_path =data_path, split=split, transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, 
                                                shuffle=shuffle[split], num_workers=4, pin_memory=True)

        return data_loader

    elif data_name == 'VOC-seg':
        normalize = transforms.Normalize([0.4589, 0.4355, 0.4032],[0.2239, 0.2186, 0.2206])
        if split == 'val':
            transform = transforms.Compose([
                                    transforms.Resize((227, 227)),
                                    transforms.ToTensor(),
                                    normalize,
                                    ])
        else: 
            raise NotImplementedError

        data = VOC_segmentation(data_path =data_path, 
                                split=split, 
                                transform=transform)
        data_loader = torch.utils.data.DataLoader(dataset=data, batch_size=batch_size, 
                                                shuffle=shuffle[split], num_workers=4, pin_memory=True)
        return data_loader

    elif data_name =='imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        if split == 'train':
            transform = transforms.Compose([
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                            ])
        elif split == 'val':
            transform = transforms.Compose([
                            transforms.Scale(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                            ])
        else:
            raise NotImplementedError

        print('Imagenet ')
        data = torchvision.datasets.ImageNet(root=data_path, split=split, download=True, transform = transform)
        
        print('Imagenet Dataset ready')
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, 
                                                        shuffle=shuffle[split], num_workers=4)

        print('Imagenet DataLoader ready')
        return data, data_loader
    else:
        print("NOT DEFINED DATASET")
        raise NotImplementedError

if __name__ == '__main__':
    voc_segment = get_loader(data_name='VOC-seg', 
                data_path= '/home/server14/geondo_workspace/data/VOCdevkit/VOC2012',
                split= 'val', 
                batch_size=64)
    
    import pdb
    pdb.set_trace()