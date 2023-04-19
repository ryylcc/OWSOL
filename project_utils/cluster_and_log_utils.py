import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `known' and `novel' categories in OWSOL setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    known_acc, ind_known, w_known = cluster_acc(y_true[mask], y_pred[mask], return_ind=True)
    novel_acc, ind_novel, w_novel  = cluster_acc(y_true[~mask], y_pred[~mask], return_ind=True)
    total_acc = weight * known_acc + (1 - weight) * novel_acc

    # j:gt    i:pred
    ind_map_known = {i: j for i, j in ind_known}
    ind_map_novel = {i: j for i, j in ind_novel}

    return total_acc, known_acc, novel_acc, ind_map_known, ind_map_novel

def split_cluster_acc_v2(y_true, y_pred, mask):

    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from known categories (True) and which ones come from novel categories (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    known_categories_gt = set(y_true[mask])
    novel_categories_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    # max - ori --> min
    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    # j:gt    i:pred
    ind_map = {j: i for i, j in ind}
    ind_map_pre2gt = {i: j for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    known_acc = 0
    total_known_instances = 0
    for i in known_categories_gt:
        known_acc += w[ind_map[i], i]
        total_known_instances += sum(w[:, i])
    known_acc /= total_known_instances

    novel_acc = 0
    total_novel_instances = 0
    for i in novel_categories_gt:
        novel_acc += w[ind_map[i], i]
        total_novel_instances += sum(w[:, i])
    novel_acc /= total_novel_instances

    return total_acc, known_acc, novel_acc, ind_map_pre2gt



def log_accs_from_preds(y_true, y_pred, mask):


    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)


    all_acc, known_acc, novel_acc, _, = split_cluster_acc_v2(y_true, y_pred, mask)
    to_return = (all_acc, known_acc, novel_acc)

    return to_return


def log_accs_from_preds_infer(y_true, y_pred, mask):


    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    all_acc, known_acc, novel_acc, ind_map_pre2gt = split_cluster_acc_v2(y_true, y_pred, mask)
    to_return = (all_acc, known_acc, novel_acc, ind_map_pre2gt)


    return to_return