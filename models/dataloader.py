import torch
import pickle
import numpy as np
from PIL import Image
import utils

class CifarDataset(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""
    
    def __init__(self, image_path, target_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.targets)

class CelebaDatasetLff(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""
    
    def __init__(self, idx_dataset, target_attr_idx, bias_attr_idx, num_classes, num_domain):
        # self.data = []
        # self.target = []
        # self.domain = []
        self.idx_dataset = idx_dataset
        self.target_attr_idx = target_attr_idx
        self.bias_attr_idx = bias_attr_idx
        self.num_classes = num_classes
        self.num_domain = num_domain
        # for i in idx_dataset:
        #     self.data.append(i[0])
        #     self.target.append(i[1][target_attr_idx])
        #     self.domain.append(i[1][bias_attr_idx])

        # print(self.data[0])
        # print(self.target[0])
        # iitr

    def __getitem__(self, index):
        img = self.idx_dataset[index][0]
        target = self.idx_dataset[index][1][self.target_attr_idx] + self.idx_dataset[index][1][self.bias_attr_idx] * self.num_classes
        # img = self.data[index]
        return img, target

    def __len__(self):
        return len(self.idx_dataset)

class CelebaDatasetLff_test(torch.utils.data.Dataset):
    """Cifar dataloader, output image and target"""
    
    def __init__(self, idx_dataset, target_attr_idx, bias_attr_idx, num_classes, num_domain):
        # self.data = []
        # self.target = []
        # self.domain = []
        self.idx_dataset = idx_dataset
        self.target_attr_idx = target_attr_idx
        self.bias_attr_idx = bias_attr_idx
        self.num_classes = num_classes
        self.num_domain = num_domain
        # for i in idx_dataset:
        #     self.data.append(i[0])
        #     self.target.append(i[1][target_attr_idx])
        #     self.domain.append(i[1][bias_attr_idx])

        # print(self.data[0])
        # print(self.target[0])
        # iitr

    def __getitem__(self, index):
        img = self.idx_dataset[index][0]
        target = self.idx_dataset[index][1][self.target_attr_idx]
        # img = self.data[index]
        return img, target

    def __len__(self):
        return len(self.idx_dataset)
    
class CifarDatasetWithWeight(torch.utils.data.Dataset):
    """Cifar dataloader, output image, target and the weight for this sample"""
    
    def __init__(self, image_path, target_path, weight_list, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(target_path, 'rb') as f:
            self.targets = pickle.load(f)
        self.transform = transform
        self.weight_list = weight_list

    def __getitem__(self, index):
        img, target = self.images[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        weight = self.weight_list[index]
        return img, target, weight

    def __len__(self):
        return len(self.targets)
    
class CifarDatasetWithDomain(torch.utils.data.Dataset):
    """Cifar dataloader, output image, class target and domain for this sample"""
    
    def __init__(self, image_path, class_label_path, domain_label_path, transform=None):
        with open(image_path, 'rb') as f:
            self.images = pickle.load(f)
        with open(class_label_path, 'rb') as f:
            self.class_label = pickle.load(f)
        with open(domain_label_path, 'rb') as f:
            self.domain_label = pickle.load(f)    
        self.transform = transform

    def __getitem__(self, index):
        img, class_label, domain_label = \
            self.images[index], self.class_label[index], self.domain_label[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, class_label, domain_label

    def __len__(self):
        return len(self.class_label)

class CelebaDataset(torch.utils.data.Dataset):
    """Celeba dataloader, output image and target"""
    
    def __init__(self, key_list, image_feature, target_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.transform = transform

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(target)

    def __len__(self):
        return len(self.key_list)

class CelebaDatasetWithWeight(torch.utils.data.Dataset):
    """Celeba dataloader, output image, target and weight for this sample"""
    
    def __init__(self, key_list, image_feature, target_dict, transform=None):
        self.key_list = key_list
        self.image_feature = image_feature
        self.target_dict = target_dict
        self.transform = transform
        target = np.array([target_dict[key] for key in key_list])
        self.per_class_weight = utils.compute_class_weight(target)

    def __getitem__(self, index):
        key = self.key_list[index]
        img, target = Image.fromarray(self.image_feature[key][()]), self.target_dict[key]
        weight = [class_weight[index] for class_weight in self.per_class_weight]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.FloatTensor(target), torch.FloatTensor(weight)

    def __len__(self):
        return len(self.key_list)




