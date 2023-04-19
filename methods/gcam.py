import os
import cv2
import argparse
import random
import shutil
import munch
import numpy as np
from sklearn.cluster import KMeans
from os.path import join as ospj
from os.path import dirname as ospd

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models.resnet import resnet50
from methods.metric import BoxEvaluator
from data.imagenet import configure_metadata_infer
from data.get_datasets import get_class_splits, get_datasets_cluster, get_datasets_gcam

from project_utils.utils import t2n
from project_utils.utils import Logger
from project_utils.cluster_and_log_utils import log_accs_from_preds_infer



def mch(**kwargs):
    return munch.Munch(dict(**kwargs))

def set_random_seed(seed):
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def configure_log_folder(args):
    log_folder = ospj('infer_log', args.dataset_name, args.experiment_name)

    if os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    os.makedirs(log_folder)
    return log_folder


def configure_log(args):
    log_file_name = ospj(args.log_folder, 'log.log')
    Logger(log_file_name)


parser = argparse.ArgumentParser()

# Util
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('--experiment_name', type=str, default='G-CAM_')
parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--override_cache', type=str2bool, nargs='?', const=True, default=False)


# Data
parser.add_argument('--dataset_name', type=str, default='ImageNet')
parser.add_argument('--metadata_root', type=str, default='./metadata/')
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--interpolation', default=3, type=int)
parser.add_argument('--crop_pct', type=float, default=0.875)
parser.add_argument('--partitions', nargs='+', default=['all', 'known', 'nov_s', 'nov_d'])


# G-CAM Setting
parser.add_argument('--cam_curve_interval', type=float, default=.001, help='CAM curve interval')
parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?', const=True, default=True)
parser.add_argument('--iou_threshold_list', nargs='+', type=int, default=[30, 50, 70])

# Save name
parser.add_argument('--weight_dict_name', type=str, default='weight_')
parser.add_argument('--weight_dict_path', type=str, default=None)
parser.add_argument('--labels_and_preds_name', type=str, default='labels_and_preds_')
parser.add_argument('--pred_save_path', type=str, default='test')



args = parser.parse_args()
args = get_class_splits(args)
args.num_known_categories = len(args.known_categories)
args.num_novel_categories = len(args.novel_categories)

model_name = args.model_path.split('/')[-2]
args.experiment_name = args.experiment_name + model_name
args.log_folder = configure_log_folder(args)
configure_log(args)

args.metadata_root = ospj(args.metadata_root, args.dataset_name)
args.pred_save_path = ospj(args.metadata_root, args.pred_save_path)
args.weight_dict_name = args.weight_dict_name + model_name
args.labels_and_preds_name = args.labels_and_preds_name + model_name
args.weight_dict_path = ospj(args.log_folder, args.weight_dict_name)



def normalize_scoremap(gcam):
    """
    Args:
        gcam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(gcam).any():
        return np.zeros_like(gcam)
    if gcam.min() == gcam.max():
        return np.zeros_like(gcam)
    gcam -= gcam.min()
    gcam /= gcam.max()
    return gcam

def get_batch_weight(targets, weight_dict):
    weight_list = []
    for item in targets:
        weight_list.append(torch.from_numpy(weight_dict[item]))
    gcam_weights = torch.stack(weight_list)
    return gcam_weights



class GcamComputer(object):
    def __init__(self, model, loader, metadata_root, labels_and_preds,
                 iou_threshold_list, dataset_name, partitions, multi_contour_eval,
                 gcam_curve_interval=.001, log_folder=None, weight_dict=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.log_folder = log_folder
        self.weight_dict = weight_dict


        metadata = configure_metadata_infer(metadata_root, labels_and_preds)
        gcam_threshold_list = list(np.arange(0, 1, gcam_curve_interval))

        self.evaluator = BoxEvaluator(metadata=metadata,
                                      dataset_name=dataset_name,
                                      gcam_threshold_list=gcam_threshold_list,
                                      iou_threshold_list=iou_threshold_list,
                                      partitions=partitions,
                                      multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_gcams(self):
        print("Computing and evaluating G-CAMs.")
        for images, targets, image_ids in self.loader:
            image_size = images.shape[2:]
            images = images.cuda()
            feature_map = self.model(images)['feature_map']
            targets = targets.detach().clone().cpu().numpy()
            gcam_weights = get_batch_weight(targets, self.weight_dict).cuda()
            gcams = (gcam_weights.view(*feature_map.shape[:2], 1, 1) *
                    feature_map).mean(1, keepdim=False)
            gcams = t2n(gcams)

            for gcam, image_id in zip(gcams, image_ids):
                gcam_resized = cv2.resize(gcam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                gcam_normalized = normalize_scoremap(gcam_resized)
                gcam_path = ospj(self.log_folder, 'scoremaps', image_id)
                if not os.path.exists(ospd(gcam_path)):
                    os.makedirs(ospd(gcam_path))
                np.save(ospj(gcam_path), gcam_normalized)
                self.evaluator.accumulate(gcam_normalized, image_id)
        return self.evaluator.compute()



if __name__=='__main__':
    set_random_seed(args.seed)


    print("=> creating model '{}'".format(args.arch))
    model = resnet50(num_classes=128, pretrained=None)

    # load pretrained
    state_dict = {}
    old_state_dict = torch.load(args.model_path, map_location='cpu')['state_dict']
    for key in old_state_dict.keys():
        if key.startswith('module.encoder_q'):
            print(key)
            new_key = key.split('encoder_q.')[1]
            state_dict[new_key] = old_state_dict[key]
    model.load_state_dict(state_dict, strict=False)
    model = nn.DataParallel(model)
    for name,parameter in model.named_parameters():
        print(name)
        print(parameter)
    model.cuda()
    model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    ############################################### Clustering ##############################################
    # --------------------
    # DATASETS
    # --------------------


    cluster_dataset = get_datasets_cluster(dataset_name=args.dataset_name, test_transform=test_transform)
    cluster_loader = DataLoader(cluster_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # --------------------
    # EXTRACT FEATURE TO CLUSTER
    # --------------------

    ft_list = []
    id_list = []
    targets = np.array([])
    mask = np.array([])
    weight_dict = {}
    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, img_ids) in enumerate(cluster_loader):

        feats = model(images.cuda())['feature']
        feats = torch.nn.functional.normalize(feats, dim=-1)
        print(feats.shape)
        ft_list.append(feats.detach().cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # distinguish labeled classes
        mask = np.append(mask, np.array([True if x.item() in args.known_categories
                                         else False for x in label]))
        id_list += img_ids

    all_feats = np.concatenate(ft_list)
    print("ft_all.shape: ", all_feats.shape)
    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=args.num_known_categories + args.num_novel_categories, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    centroids = kmeans.cluster_centers_

    print('labels.shape:' , preds.shape)
    print('weight.shape: ', centroids.shape)

    all_acc, known_acc, novel_acc, ind_map_pre2gt = log_accs_from_preds_infer(y_true=targets, y_pred=preds, mask=mask)

    print('Accuracies: All {:.4f} | Known {:.4f} | Novel {:.4f}'.format(all_acc, known_acc, novel_acc))



    # match pred with target
    pred2gt = np.array([])
    pred2gt = np.append(pred2gt, np.array([int(ind_map_pre2gt[pred]) for pred in preds]))
    print('pred2gt.shape:', pred2gt.shape)

    # use pred2gt to save the weight_dict
    for i,item in enumerate(centroids):
        weight_dict[int(ind_map_pre2gt[i])] = item
    print("len(weight_dict): ", len(weight_dict.keys()))
    np.save(args.weight_dict_path, weight_dict)

    with open(ospj(args.pred_save_path, args.labels_and_preds_name), 'w') as f:
        for i,item in enumerate(id_list):
            f.write(item + ',' + str(int(targets[i])) + ','  + str(int(pred2gt[i])))
            f.write('\n')


    ############################################## INFER G-CAM ##############################################


    # --------------------
    # DATASETS
    # --------------------
    gcam_dataset = get_datasets_gcam(dataset_name=args.dataset_name, test_transform=test_transform, target_and_pred=args.labels_and_preds_name)
    gcam_loader = DataLoader(gcam_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)



    gcam_computer = GcamComputer(
        model=model,
        loader=gcam_loader,
        metadata_root=os.path.join(args.metadata_root, 'test'),
        labels_and_preds=args.labels_and_preds_name,
        iou_threshold_list=args.iou_threshold_list,
        dataset_name=args.dataset_name,
        partitions=args.partitions,
        gcam_curve_interval=args.cam_curve_interval,
        multi_contour_eval=args.multi_contour_eval,
        log_folder=args.log_folder,
        weight_dict = weight_dict
    )

    loc_acc, clus_loc_acc, clus_acc = gcam_computer.compute_and_evaluate_gcams()

    print('######################## Clus Acc ########################')
    for k,v in clus_acc.items():
        print(k, ":", v)
    print('###################### Clus Loc Acc ######################')
    for k,v in clus_loc_acc.items():
        print(k, ":", np.average(v))
    print('######################## Loc Acc #########################')
    for k,v in loc_acc.items():
        print(k, ":", np.average(v))

