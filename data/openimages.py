# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import munch
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from config import openimages_root, openimages_meta


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def configure_metadata(metadata_root):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.image_ids_lb = os.path.join(metadata_root, 'image_ids_labeled.txt')
    metadata.class_labels = os.path.join(metadata_root, 'class_labels.txt')
    return metadata

def configure_metadata_infer(metadata_root, cluster_preds_name):
    metadata = mch()
    metadata.image_ids = os.path.join(metadata_root, 'image_ids.txt')
    metadata.class_labels = os.path.join(metadata_root, cluster_preds_name)
    metadata.partitions = os.path.join(metadata_root, 'partitions.txt')
    metadata.image_sizes = os.path.join(metadata_root, 'image_sizes.txt')
    metadata.localization = os.path.join(metadata_root, 'localization.txt')
    return metadata


def get_image_ids(metadata):
    """
    image_ids.txt has the structure
    <path>
    path/image1.jpg
    path/image2.jpg
    path/image3.jpg
    ...
    """
    image_ids = []
    with open(metadata.image_ids) as f:
        for line in f.readlines():
            image_ids.append(line.strip('\n'))
    return image_ids


def get_image_ids_lb(metadata):
    """
    image_ids_labeled.txt has the structure
    <path>
    path/image1.jpg
    path/image2.jpg
    path/image3.jpg
    ...
    """
    image_ids_lb = []
    with open(metadata.image_ids_lb) as f:
        for line in f.readlines():
            image_ids_lb.append(line.strip('\n'))
    return image_ids_lb


def get_class_labels(metadata):
    """
    class_labels.txt has the structure

    <path>,<integer_class_label>
    path/image1.jpg,0
    path/image2.jpg,1
    path/image3.jpg,1
    ...
    """
    class_labels = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string = line.strip('\n').split(',')
            class_labels[image_id] = int(class_label_string)
    return class_labels


def get_partitions(metadata):
    """
    partitions.txt has the structure

    <path>,<integer_partition> 0 --> known, 1 --> nov-s, 2 --> nov-d
    path/image1.jpg,0
    path/image2.jpg,1
    path/image3.jpg,2
    ...
    """
    partitions = {}
    with open(metadata.partitions) as f:
        for line in f.readlines():
            image_id, partition_string = line.strip('\n').split(',')
            partitions[image_id] = int(partition_string)
    return partitions


def get_class_labels_and_pred(metadata):
    """
    class_labels_preds.txt has the structure

    <path>,<integer_class_label>,<integer_pred>
    path/image1.jpg,0,0
    path/image2.jpg,1,1
    path/image3.jpg,2,2
    ...
    """
    class_labels_and_preds = {}
    with open(metadata.class_labels) as f:
        for line in f.readlines():
            image_id, class_label_string, pred_string = line.strip('\n').split(',')
            class_labels_and_preds[image_id] = (int(class_label_string), int(pred_string))
    return class_labels_and_preds


def get_image_sizes(metadata):
    """
    image_sizes.txt has the structure

    <path>,<w>,<h>
    path/image1.jpg,500,300
    path/image2.jpg,1000,600
    path/image3.jpg,500,300
    ...
    """
    image_sizes = {}
    with open(metadata.image_sizes) as f:
        for line in f.readlines():
            image_id, ws, hs = line.strip('\n').split(',')
            w, h = int(ws), int(hs)
            image_sizes[image_id] = (w, h)
    return image_sizes


def get_bounding_boxes(metadata):
    """
    localization.txt (for bounding box) has the structure

    <path>,<x0>,<y0>,<x1>,<y1>
    path/image1.jpg,156,163,318,230
    path/image1.jpg,23,12,101,259
    path/image2.jpg,143,142,394,248
    path/image3.jpg,28,94,485,303
    ...

    One image may contain multiple boxes (multiple boxes for the same path).
    """
    boxes = {}
    with open(metadata.localization) as f:
        for line in f.readlines():
            image_id, x0s, x1s, y0s, y1s = line.strip('\n').split(',')
            x0, x1, y0, y1 = int(x0s), int(x1s), int(y0s), int(y1s)
            if image_id in boxes:
                boxes[image_id].append((x0, x1, y0, y1))
            else:
                boxes[image_id] = [(x0, x1, y0, y1)]
    return boxes

def pil_loader(path: str):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class TrainDataset(Dataset):
    def __init__(self, data_root, meta_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.metadata = configure_metadata(meta_root)
        self.image_ids = get_image_ids(self.metadata)
        self.image_ids_lb = get_image_ids_lb(self.metadata)
        self.image_labels = get_class_labels(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_lb_id = self.image_ids_lb[idx]
        image_lb_label = self.image_labels[image_lb_id]
        image = pil_loader(os.path.join(self.data_root, image_id))
        image_lb = pil_loader(os.path.join(self.data_root, image_lb_id))
        image = self.transform(image)
        image_lb = self.transform(image_lb)


        return image, image_lb, image_lb_label, idx

    def __len__(self):
        return len(self.image_ids)


class MclDataset(Dataset):
    def __init__(self, data_root, meta_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.metadata = configure_metadata(meta_root)
        self.image_ids = get_image_ids(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = pil_loader(os.path.join(self.data_root, image_id))
        image = self.transform(image)
        return image, idx

    def __len__(self):
        return len(self.image_ids)


class EvalDataset(Dataset):
    def __init__(self, data_root, meta_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.metadata = configure_metadata(meta_root)
        self.image_ids = get_image_ids(self.metadata)
        self.image_labels = get_class_labels(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = pil_loader(os.path.join(self.data_root, image_id))
        image = self.transform(image)
        return image, image_label

    def __len__(self):
        return len(self.image_ids)


class ClusterDataset(Dataset):
    def __init__(self, data_root, meta_root, transform):
        self.data_root = data_root
        self.transform = transform
        self.metadata = configure_metadata(meta_root)
        self.image_ids = get_image_ids(self.metadata)
        self.image_labels = get_class_labels(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label = self.image_labels[image_id]
        image = pil_loader(os.path.join(self.data_root, image_id))
        image = self.transform(image)
        return image, image_label, image_id

    def __len__(self):
        return len(self.image_ids)


class GcamDataSet(Dataset):
    def __init__(self, data_root, meta_root, preds_name, transform):
        self.data_root = data_root
        self.transform = transform
        self.metadata = configure_metadata_infer(meta_root, preds_name)
        self.image_ids = get_image_ids(self.metadata)
        self.image_labels_and_pred = get_class_labels_and_pred(self.metadata)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_label, pred = self.image_labels_and_pred[image_id]
        image = pil_loader(os.path.join(self.data_root, image_id))
        image = self.transform(image)
        return image, pred, image_id

    def __len__(self):
        return len(self.image_ids)

def get_openimages_datasets(train_transform, test_transform):

    meta_train = os.path.join(openimages_meta, 'train')
    meta_val = os.path.join(openimages_meta, 'val')
    meta_test = os.path.join(openimages_meta, 'test')

    train_dataset = TrainDataset(openimages_root, meta_train, train_transform)
    cluster_dataset = MclDataset(openimages_root, meta_train, test_transform)
    val_dataset = EvalDataset(openimages_root, meta_val, test_transform)

    # return train_dataset, test_dataset, val_dataset
    return train_dataset, cluster_dataset, val_dataset


def get_openimages_datasets_cluster(test_transform):

    meta_test = os.path.join(openimages_meta, 'test')
    test_dataset = ClusterDataset(openimages_root, meta_test, test_transform)

    return test_dataset


def get_openimages_datasets_gcam(test_transform, target_and_pred):

    meta_test = os.path.join(openimages_meta, 'test')
    test_dataset = GcamDataSet(openimages_root, meta_test, target_and_pred, test_transform)

    return test_dataset

def get_openimages_datasets_estimate_k(test_transform):

    val_test = os.path.join(openimages_meta, 'val')
    val_dataset = ClusterDataset(openimages_root, val_test, test_transform)

    return val_dataset



if __name__=='__main__':
    import numpy as np
    np.set_printoptions(threshold=np.inf)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.ToTensor(),
        normalize
    ]
    dataset = TrainDataset('/data/zhaochuan/NCL/code/wsolevaluation-master/dataset/ILSVRC/', '/data/zhaochuan/NCL/code/GWSOL/metadata/openimages/train/', transforms.Compose(augmentation))
    train_loader = DataLoader(
        dataset, batch_size=16, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)
    print(len(train_loader))
    for i, (images, labels, mask_lab) in enumerate(train_loader):
        print(images.shape)
        print(labels.shape)
        print(mask_lab)

