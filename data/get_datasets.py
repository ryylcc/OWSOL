
from data.imagenet import get_imagenet_datasets, get_imagenet_datasets_cluster, get_imagenet_datasets_gcam, get_imagenet_datasets_estimate_k
from data.inatloc import get_inatloc_datasets, get_inatloc_datasets_cluster, get_inatloc_datasets_gcam, get_inatloc_datasets_estimate_k
from data.openimages import get_openimages_datasets, get_openimages_datasets_cluster, get_openimages_datasets_gcam, get_openimages_datasets_estimate_k



get_dataset_funcs = {
    'ImageNet': get_imagenet_datasets,
    'iNatLoc': get_inatloc_datasets,
    'OpenImages': get_openimages_datasets
}

get_dataset_cluster_funcs = {
    'ImageNet': get_imagenet_datasets_cluster,
    'iNatLoc': get_inatloc_datasets_cluster,
    'OpenImages': get_openimages_datasets_cluster
}

get_dataset_gcam_funcs = {
    'ImageNet': get_imagenet_datasets_gcam,
    'iNatLoc': get_inatloc_datasets_gcam,
    'OpenImages': get_openimages_datasets_gcam
}

get_dataset_estimate_k_funcs = {
    'ImageNet': get_imagenet_datasets_estimate_k,
    'iNatLoc': get_inatloc_datasets_estimate_k,
    'OpenImages': get_openimages_datasets_estimate_k
}

def get_datasets(dataset_name, train_transform, test_transform):

    """
    :return: train_dataset: labelled and unlabelled
             test_dataset,
             val_dataset,
    """
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    train_dataset, eval_dataset, val_dataset = get_dataset_f(train_transform=train_transform, test_transform=test_transform)

    return train_dataset, eval_dataset, val_dataset



def get_datasets_cluster(dataset_name, test_transform):

    if dataset_name not in get_dataset_cluster_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_cluster_funcs[dataset_name]
    test_dataset = get_dataset_f(test_transform=test_transform)

    return test_dataset



def get_datasets_gcam(dataset_name, test_transform, target_and_pred):

    if dataset_name not in get_dataset_gcam_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_gcam_funcs[dataset_name]
    test_dataset = get_dataset_f(test_transform=test_transform, target_and_pred=target_and_pred)

    return test_dataset



def get_datasets_estimate_k(dataset_name, test_transform):

    if dataset_name not in get_dataset_estimate_k_funcs.keys():
        raise ValueError

    # Get datasets
    get_dataset_f = get_dataset_estimate_k_funcs[dataset_name]
    val_dataset = get_dataset_f(test_transform=test_transform)

    return val_dataset



def get_class_splits(args):

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'ImageNet':

        args.image_size = 224
        args.known_categories = range(500)
        args.novel_categories = range(500, 1000)
        args.nov_s_categories = range(500, 750)
        args.nov_d_categories = range(750, 1000)


    elif args.dataset_name == 'iNatLoc':
        args.image_size = 224
        args.known_categories = range(250)
        args.novel_categories = range(250, 500)
        args.nov_s_categories = range(250, 375)
        args.nov_d_categories = range(375, 500)


    elif args.dataset_name == 'OpenImages':

        args.image_size = 224
        args.known_categories = range(75)
        args.novel_categories = range(75, 150)

    else:

        raise NotImplementedError

    return args