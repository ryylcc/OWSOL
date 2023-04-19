import argparse
import os
import numpy as np
from scipy.optimize import minimize_scalar
from functools import partial

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from models.resnet import resnet50
from data.get_datasets import get_class_splits, get_datasets_estimate_k
from project_utils.cluster_and_log_utils import cluster_acc



# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_kmeans(K, all_feats=None, targets=None, mask_cls=None, args=None, verbose=False):

    """
    In this case, the val loader needs to have known and novel categories
    """

    if K is None:
        K = args.num_known_categories + args.num_novel_categories


    print('Fitting K-Means...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_cls


    known_acc, known_nmi, known_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                      nmi_score(targets[mask], preds[mask]), \
                                      ari_score(targets[mask], preds[mask])

    novel_acc, novel_nmi, novel_ari = cluster_acc(targets.astype(int)[~mask],
                                                  preds.astype(int)[~mask]), \
                                      nmi_score(targets[~mask], preds[~mask]), \
                                      ari_score(targets[~mask], preds[~mask])

    if verbose:
        print('K')
        print('Known Categories acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(known_acc, known_nmi,
                                                                             known_ari))
        print('Novel Categories acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(novel_acc, novel_nmi,
                                                                               novel_ari))


    return known_acc



def test_kmeans_for_scipy(K, all_feats=None, targets=None, mask_cls=None, args=None, verbose=False):

    """
    In this case, the val loader needs to have known and novel categories
    """

    K = int(K)

    print(f'Fitting K-Means for K = {K}...')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(all_feats)
    preds = kmeans.labels_

    # -----------------------
    # EVALUATE
    # -----------------------
    mask = mask_cls


    known_acc, known_nmi, known_ari = cluster_acc(targets.astype(int)[mask], preds.astype(int)[mask]), \
                                      nmi_score(targets[mask], preds[mask]), \
                                      ari_score(targets[mask], preds[mask])

    novel_acc, novel_nmi, novel_ari = cluster_acc(targets.astype(int)[~mask],
                                                  preds.astype(int)[~mask]), \
                                      nmi_score(targets[~mask], preds[~mask]), \
                                      ari_score(targets[~mask], preds[~mask])

    print(f'K = {K}')
    print('Known Categories acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(known_acc, known_nmi,
                                                                         known_ari))
    print('Novel Categories acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(novel_acc, novel_nmi,
                                                                           novel_ari))

    return -known_acc


def binary_search(all_feats, targets, mask_cls, args):

    min_classes = args.num_known_categories

    # Iter 0
    big_k = args.max_classes
    small_k = min_classes
    diff = big_k - small_k
    middle_k = int(0.5 * diff + small_k)

    known_acc_big = test_kmeans(big_k, all_feats, targets, mask_cls, args)
    known_acc_small = test_kmeans(small_k, all_feats, targets, mask_cls, args)
    known_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_cls, args)

    print(f'Iter 0: BigK {big_k}, Acc {known_acc_big:.4f} | MiddleK {middle_k}, Acc {known_acc_middle:.4f} | SmallK {small_k}, Acc {known_acc_small:.4f} ')
    all_accs = [known_acc_small, known_acc_middle, known_acc_big]
    best_acc_so_far = np.max(all_accs)
    best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
    print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')

    for i in range(1, int(np.log2(diff)) + 1):

        if known_acc_big > known_acc_small:

            best_acc = max(known_acc_middle, known_acc_big)

            small_k = middle_k
            known_acc_small = known_acc_middle
            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)

        else:

            best_acc = max(known_acc_middle, known_acc_small)
            big_k = middle_k

            diff = big_k - small_k
            middle_k = int(0.5 * diff + small_k)
            known_acc_big = known_acc_middle

        known_acc_middle = test_kmeans(middle_k, all_feats, targets, mask_cls, args)

        print(f'Iter {i}: BigK {big_k}, Acc {known_acc_big:.4f} | MiddleK {middle_k}, Acc {known_acc_middle:.4f} | SmallK {small_k}, Acc {known_acc_small:.4f} ')
        all_accs = [known_acc_small, known_acc_middle, known_acc_big]
        best_acc_so_far = np.max(all_accs)
        best_acc_at_k = np.array([small_k, middle_k, big_k])[np.argmax(all_accs)]
        print(f'Best Acc so far {best_acc_so_far:.4f} at K {best_acc_at_k}')


def scipy_optimise(all_feats, targets, mask_cls, args):

    small_k = args.num_known_categories
    big_k = args.max_classes

    test_k_means_partial = partial(test_kmeans_for_scipy, all_feats=all_feats, targets=targets, mask_cls=mask_cls, args=args, verbose=True)
    res = minimize_scalar(test_k_means_partial, bounds=(small_k, big_k), method='bounded', options={'disp': True})
    print(f'Optimal K is {res.x}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='estimate k',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--workers', default=16, type=int)
    parser.add_argument('--max_classes', default=1000, type=int)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
    parser.add_argument('--search_mode', type=str, default='brent', help='Mode for black box optimisation')
    parser.add_argument('--dataset_name', type=str, default='iNatLoc', help='options: ImageNet, iNatLoc, OpenImages')
    parser.add_argument('--model_path', type=str, default='/data/zhaochuan/NCL/code/GWSOL/save_imagenet/checkpoint_0009.pth.tar')

    # ----------------------
    # INIT
    # ----------------------
    args = parser.parse_args()
    args = get_class_splits(args)
    args.num_known_categories = len(args.known_categories)
    args.num_novel_categories = len(args.novel_categories)
    print(args)


    print("=> creating model '{}'".format(args.arch))
    model = resnet50(num_classes=1000)

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

    # --------------------
    # DATASETS
    # --------------------
    print('Building datasets...')

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_dataset = get_datasets_estimate_k(args.dataset_name, test_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)


    all_feats = []
    targets = np.array([])
    mask_cls = np.array([])     # From all the data, which instances belong to seen classes

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, img_ids) in enumerate(val_loader):

        feats = model(images.cuda())['feature']

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.detach().cpu().numpy())

        targets = np.append(targets, label.cpu().numpy())
        mask_cls = np.append(mask_cls, np.array([True if x.item() in args.known_categories
                                                 else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    mask_cls = mask_cls.astype(bool)
    all_feats = np.concatenate(all_feats)

    print('Testing on the val set...')
    if args.search_mode == 'brent':
        print('Optimising with Brents algorithm')
        scipy_optimise(all_feats=all_feats, targets=targets, mask_cls=mask_cls, args=args)
    else:
        binary_search(all_feats=all_feats, targets=targets, mask_cls=mask_cls, args=args)