# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import math
import random
import shutil
import time
import argparse
import builtins
import datetime
import pickle
import warnings
import faiss
import numpy as np
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed


from models.builder import MoCo
from models.resnet import resnet50

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from project_utils.cluster_and_log_utils import log_accs_from_preds


from config import pretrained_path, imagenet_sup_queue_path, inatloc_sup_queue_path, openimages_sup_queue_path


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='PyTorch OWSOL Training')
parser.add_argument('--dataset_name', type=str, default='ImageNet',
                    help='options: ImageNet, iNatLoc, OpenImages')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--eval_batch_size', default=64, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default="env://", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
# augmentation
parser.add_argument('--n_views', default=2, type=int)
parser.add_argument('--interpolation', default=3, type=int)
parser.add_argument('--crop_pct', type=float, default=0.875)

# model specific configs
parser.add_argument('--mlp_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco_m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--scl_t', default=0.07, type=float,
                    help='softmax temperature for scl(default: 0.07)')
parser.add_argument('--mcl_t', default=0.2, type=float,
                    help='base temperature for mcl(default: 0.07)')
parser.add_argument('--num_cluster', default='50000', type=int,
                    help='number of clusters')
parser.add_argument('--num_multi_centroids', default=5, type=int,
                    help='number of multi centroids')
parser.add_argument('--mcl_k', default=4096, type=int,
                    help='number of negative centroids')
parser.add_argument('--scl_weight', type=float, default=0.5)
parser.add_argument('--mcl_weight', type=float, default=0.5)



class SupConLoss(nn.Module):
    def __init__(self):
        super(SupConLoss, self).__init__()


    def forward(self, logits, mask):

        # logits --> Nx(1+scl_k)
        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()


        # compute log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))      # log(A/B)=log(A)-log(B) --> torch.log(exp_logits) - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -1 * mean_log_prob_pos
        loss = loss.mean()

        return loss


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]


def main():
    args = parser.parse_args()
    args = get_class_splits(args)
    args.num_known_categories = len(args.known_categories)
    args.num_novel_categories = len(args.novel_categories)


    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print('num_cluster:', args.num_cluster)
    print('num_multi_centroids:', args.num_multi_centroids )
    print('learning rate:', args.lr)
    print('scl_weight: ', args.scl_weight)
    print('mcl_weight: ', args.mcl_weight)

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,timeout=datetime.timedelta(0, 7200),
                                world_size=args.world_size, rank=args.rank)

    # create model
    if args.dataset_name == 'ImageNet':
        f = open(imagenet_sup_queue_path, 'rb')
        sup_queue = pickle.load(f)
    elif args.dataset_name == 'iNatLoc':
        f = open(inatloc_sup_queue_path, 'rb')
        sup_queue = pickle.load(f)
    elif args.dataset_name == 'OpenImages':
        f = open(openimages_sup_queue_path, 'rb')
        sup_queue = pickle.load(f)




    print("=> creating model '{}'".format(args.arch))
    model = MoCo(
        resnet50, pretrained_path, sup_queue, args.num_known_categories, args.num_multi_centroids,
        args.mlp_dim, args.mcl_k, args.moco_m, args.scl_t)
    print(model)

    for name, parms in model.named_parameters():
        print('-->name:', name)
        print('-->grad_requirs:',parms.requires_grad)

    for name,parameter in model.named_parameters():
        print(name)
        print(parameter)
        print('#'*80)


    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")



    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    criterion_scl = SupConLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), args.lr,
    #                               weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.dataset_name, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, eval_dataset, val_dataset = get_datasets(args.dataset_name, train_transform, test_transform)




    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        eval_sampler = torch.utils.data.distributed.DistributedSampler(eval_dataset,shuffle=False)
        # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,shuffle=False)
    else:
        train_sampler = None
        eval_sampler = None
        # val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
        sampler=eval_sampler, num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=args.workers, batch_size=args.eval_batch_size, shuffle=False)

    best_val_acc = 0
    for epoch in range(args.start_epoch, args.epochs):

        # compute momentum features for center-cropped images
        features = compute_features(eval_loader, model, args)

        # placeholder for clustering result
        cluster_result = {}
        cluster_result['im2cluster'] = torch.zeros(args.num_multi_centroids, len(eval_dataset), dtype=torch.long).cuda()
        cluster_result['centroids'] = torch.zeros(int(args.num_cluster), args.mlp_dim).cuda()
        cluster_result['density'] = torch.zeros(int(args.num_cluster)).cuda()

        if args.gpu == 0:
            features[torch.norm(features,dim=1)>1.5] /= 2 #account for the few samples that are computed twice
            features = features.numpy()
            cluster_result = run_kmeans(features, args)  #run kmeans clustering on master node
            # save the clustering result
            # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))

        dist.barrier()
        # broadcast clustering result
        print('----- broadcast -----')
        dist.broadcast(cluster_result['im2cluster'], 0, async_op=False)
        dist.broadcast(cluster_result['centroids'], 0, async_op=False)
        dist.broadcast(cluster_result['density'], 0, async_op=False)
        print("im2cluster:", cluster_result['im2cluster'].shape)
        print("centroids:", cluster_result['centroids'].shape)
        print("density:", cluster_result['density'].shape)


        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, criterion_scl, optimizer, epoch, cluster_result, args)

        with torch.no_grad():
            print('Testing on val set...')
            all_acc, known_acc, novel_acc = test_kmeans(model, val_loader, args=args)
        print('Val Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, known_acc, novel_acc))

        if known_acc > best_val_acc:
            print('Best val Accuracies: All {:.4f} | Old {:.4f} | New {:.4f}'.format(all_acc, known_acc, novel_acc))
            best_val_acc = known_acc

        if (epoch + 1) % 10 == 0:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                        and args.rank % ngpus_per_node == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, save_path='save/{}/lr{}_scl{}_mcl{}_mc{}_e{}'.format(args.dataset_name, args.lr, args.scl_weight, args.mcl_weight, args.num_multi_centroids,
                                                                                       args.epochs), filename='checkpoint_{:04d}.pth.tar'.format(epoch))




def train(train_loader, model, criterion, criterion_scl, optimizer, epoch, cluster_result, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    loss_mcl = AverageMeter('loss_mcl', ':.4e')
    loss_scl = AverageMeter('loss_scl', ':.4e')
    acc_mcl = AverageMeter('Acc@Proto', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [losses, loss_mcl, loss_scl, acc_mcl],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, images_lb, labels, index) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)
            images_lb[0] = images_lb[0].cuda(args.gpu, non_blocking=True)
            images_lb[1] = images_lb[1].cuda(args.gpu, non_blocking=True)
            labels = labels.cuda(args.gpu, non_blocking=True)


        # compute output
        scl_logits, scl_labels, mcl_logits, mcl_labels = model(im_q=images[0], im_q_lb=images_lb[0], im_k_lb=images_lb[1],
                                                               targets=labels, is_eval=False, cluster_result=cluster_result, index=index)


        loss1 = criterion(mcl_logits, mcl_labels)
        acc = accuracy(mcl_logits, mcl_labels)[0]
        acc_mcl.update(acc[0], images[0].size(0))

        loss2 = criterion_scl(scl_logits, scl_labels)
        loss = args.mcl_weight * loss1 + args.scl_weight * loss2


        losses.update(loss.item(), images[0].size(0))
        loss_mcl.update(loss1.item(), images[0].size(0))
        loss_scl.update(loss2.item(), images[0].size(0))


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)



def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(state, os.path.join(save_path, filename))
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def test_kmeans(model, test_loader, args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label) in enumerate(test_loader):

        images = images.cuda(args.gpu, non_blocking=True)

        # Pass features through base model and then additional learnable transform (linear layer)
        feats =  model.module.forward_feature(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        # distinguish labeled classes and
        mask = np.append(mask, np.array([True if x.item() in args.known_categories
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_known_categories + args.num_novel_categories, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, known_acc, novel_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask)

    return all_acc, known_acc, novel_acc



def compute_features(eval_loader, model, args):
    print('Computing features to cluster for mcl...')
    model.eval()
    features = torch.zeros(len(eval_loader.dataset), args.mlp_dim).cuda()
    for i, (images, index) in enumerate(eval_loader):
        with torch.no_grad():
            images = images.cuda(non_blocking=True)
            feat = model(images,is_eval=True)
            features[index] = feat
    dist.barrier()
    dist.all_reduce(features, op=dist.ReduceOp.SUM)
    return features.cpu()


def run_kmeans(x, args):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {}
    # intialize faiss clustering parameters
    d = x.shape[1]
    k = int(args.num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    res.setTempMemory(2048 * 1024 * 1024)
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = args.gpu
    index = faiss.GpuIndexFlatL2(res, d, cfg)

    clus.train(x, index)

    D, I = index.search(x, args.num_multi_centroids) # for each sample, find cluster distance and top-k assignments
    im2cluster = []
    for i in range(args.num_multi_centroids):
        im2cluster_sub = [int(n[i]) for n in I] # [x.shape[0]]
        im2cluster.append(im2cluster_sub)
        if i == 0:
            im2cluster0 = im2cluster_sub


    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

    # sample-to-centroid distances for each cluster
    Dcluster = [[] for c in range(k)]               # [[], [], [], [], [] ...k]
    for im, i in enumerate(im2cluster0):
        Dcluster[i].append(D[im][0])

    # concentration estimation (phi)
    density = np.zeros(k)
    for i,dist in enumerate(Dcluster):
        if len(dist)>1:
            d = (np.asarray(dist)**0.5).mean()
            density[i] = d

    #if cluster only has one point, use the max to estimate its concentration
    dmax = density.max()
    for i,dist in enumerate(Dcluster):
        if len(dist)<=1:
            density[i] = dmax

    density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    density = args.mcl_t * density/density.mean()  #scale the mean to temperature



    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda()
    centroids = nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster

    print("Kmean Done!")
    return results


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
