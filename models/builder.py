# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from random import sample


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, pretrained_path, scl_queue, num_labeled_classes=500, num_multi_centers=5, dim=128, K=16384, m=0.999, T=0.07, step=12):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.step = step
        self.num_labeled_classes = num_labeled_classes
        self.num_multi_centers = num_multi_centers
        self.scl_queue_loc = torch.arange(0, num_labeled_classes * step, step=step, dtype=torch.long)

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim, pretrained=pretrained_path)
        self.encoder_k = base_encoder(num_classes=dim, pretrained=pretrained_path)

        dim_mlp = self.encoder_q.fc.weight.shape[1]
        self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
        self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)    # initialize
            param_k.requires_grad = False       # not update by gradient


        # create the queue for supervised contrastive learning
        self.register_buffer("scl_queue", scl_queue)
        self.scl_queue = nn.functional.normalize(self.scl_queue, dim=0)
        self.register_buffer("scl_queue_ptr", torch.arange(0, num_labeled_classes * step, step=step, dtype=torch.long))



    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _scl_dequeue_and_enqueue(self, keys, labels):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        for i, key in enumerate(keys):

            ptr = int(self.scl_queue_ptr[labels[i]])


            # replace the keys at ptr (dequeue and enqueue) --> self.scl_queue: (dim, K_supervised)
            self.scl_queue[:, ptr] = key.T
            ptr = (ptr + 1) % self.step + self.scl_queue_loc[labels[i]]  # move pointer

            self.scl_queue_ptr[labels[i]] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, im_q=None, im_q_lb=None, im_k_lb=None, targets=None, is_eval=False, cluster_result=None, index=None):

        # extract features for mcl
        if is_eval:
            k = self.encoder_k(im_q)['logits']
            k = nn.functional.normalize(k, dim=1)
            return k

        """
        Input:
            im_q: a batch of images for mcl
            im_q_lb: a batch of query images for scl
            im_k_lb: a batch of key images for scl
        Output:
            scl_logits, scl_labels, mcl_logits, mcl_labels
        """

        # compute query features for mcl
        q = self.encoder_q(im_q)['logits']  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        q_lb = self.encoder_q(im_q_lb)['logits']
        q_lb = nn.functional.normalize(q_lb, dim=1)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k_lb, idx_unshuffle = self._batch_shuffle_ddp(im_k_lb)

            k_lb = self.encoder_k(im_k_lb)['logits']  # keys: NxC
            k_lb = nn.functional.normalize(k_lb, dim=1)

            # undo shuffle
            k_lb = self._batch_unshuffle_ddp(k_lb, idx_unshuffle)
            # print(k_lb.shape)

        # --------------------
        # compute scl_logits
        # --------------------
        # Einstein sum is more intuitive
        l_aug = torch.einsum('nc,nc->n', [q_lb, k_lb]).unsqueeze(-1)
        l_sup = torch.einsum('nc,ck->nk', [q_lb, self.scl_queue.clone().detach()])

        # scl_logits: N x (1 + self.num_labeled_classes * step)
        scl_logits = torch.cat([l_aug, l_sup], dim=1)

        # apply temperature
        scl_logits /= self.T



        # generate scl_labels to mask negative
        '''
        batch_size = 2
        one_hot:
            [[0, 0, 1],
             [1, 0, 0]]
        scl_labels(step = 2):
            [[0, 0, 0, 0, 1, 1],
             [1, 1, 0, 0, 0, 0]]
        '''
        batch_size = q_lb.shape[0]
        labels = targets.unsqueeze(1).cuda()

        one_hot = torch.zeros((batch_size, self.num_labeled_classes), dtype=torch.long).cuda()
        one_hot = one_hot.scatter(1, labels, 1)
        scl_labels = one_hot.unsqueeze(-1).repeat(1, 1, self.step).view(batch_size, -1)

        aug_labels = torch.ones([batch_size,1], dtype=torch.long).cuda()
        scl_labels = torch.cat([aug_labels, scl_labels], dim=1)  # N x (1 + self.num_labeled_classes * step)

        # dequeue and enqueue
        self._scl_dequeue_and_enqueue(k_lb,labels.squeeze(1))



        # --------------------
        # compute mcl_logits
        # --------------------
        im2cluster, centroids, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density']

        centroids = centroids / density.unsqueeze(dim=1)
        # get positive centroids
        pos_centroid_ids = im2cluster[:, index].reshape(self.num_multi_centers * batch_size)   # 5, batch_size --> 5 * batch_size
        pos_centroids = centroids[pos_centroid_ids].reshape(self.num_multi_centers, batch_size, -1).mean(dim=0)

        # sample negative centroids
        all_centroids_id = [i for i in range(im2cluster.max()+1)]
        neg_centroids_id = set(all_centroids_id) - set(pos_centroid_ids.tolist())
        neg_centroids_id = sample(neg_centroids_id, self.K)                                   #sample 16384 negative centroids
        neg_centroids = centroids[neg_centroids_id]

        # q --> (Batch, 128)    pos_centroids --> (Batch, 128)    neg_centroids --> (K, 128)
        pos_logits = torch.matmul(q.unsqueeze(dim=1), pos_centroids.unsqueeze(dim=-1)).reshape(-1, 1) # (Batch, 1)
        neg_logits = torch.matmul(q, neg_centroids.T)                                                 # (Batch, K)
        mcl_logits = torch.cat([pos_logits, neg_logits], dim=1)
        mcl_labels = torch.zeros(mcl_logits.shape[0], dtype=torch.long).cuda()


        return scl_logits, scl_labels, mcl_logits, mcl_labels


    def forward_feature(self, im_q):
        k = self.encoder_k(im_q)['feature']
        k = nn.functional.normalize(k, dim=1)
        return k

    def forward_feature_map(self, im_q):
        q = self.encoder_q(im_q)['feature_map']
        return q


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output



