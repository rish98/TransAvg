import torch
from torch.utils.data import random_split, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose
import torchvision.datasets as datasets
from torchvision.datasets import MNIST
import numpy as np



def get_mnist(data_path: str = "./data"):
    """Download MNIST and apply minimal transformation."""
    tr = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(data_path, train=True, download=True, transform=tr)
    testset = MNIST(data_path, train=False, download=True, transform=tr)
    return trainset, testset

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Scale(scale_size)] + t_list

    return transforms.Compose(t_list)

def get_cifar10(data_path: str = "./data"):
    """Download MNIST and apply minimal transformation."""
    normalize= {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}

    tr =scale_crop(input_size=32,scale_size=32, normalize=normalize)
    
    trainset = datasets.CIFAR10(data_path, train=True, download=True, transform=tr)
    testset = datasets.CIFAR10(data_path, train=False, download=True, transform=tr)
    return trainset, testset

def partition_data_dirichlet(num_clients, alpha,batch_size, seed=42) :
    """Partition according to the Dirichlet distribution.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = get_mnist()
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.default_rng(seed)

    # get the targets
    tmp_t = trainset.targets
    if isinstance(tmp_t, list):
        tmp_t = np.array(tmp_t)
    if isinstance(tmp_t, torch.Tensor):
        tmp_t = tmp_t.numpy()
    num_classes = len(set(tmp_t))
    total_samples = len(tmp_t)
    while min_samples < min_required_samples_per_client:
        idx_clients: List[List] = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(tmp_t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array(
                [
                    p * (len(idx_j) < total_samples / num_clients)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
            min_samples = min([len(idx_j) for idx_j in idx_clients])

    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    val_ratio=0.1
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets_per_client:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2,drop_last=True)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2,drop_last=True)
        )
        testloader = DataLoader(testset, batch_size=128,drop_last=True)


    return trainloaders,valloaders, testloader
    
def prepare_dataset(num_partitions: int, batch_size: int, val_ratio: float = 0.1):
    """Download MNIST and generate IID partitions."""

    trainset, testset = get_mnist()
    # trainset, testset = get_cifar10()
    
    # split trainset into `num_partitions` trainsets (one per client)
    # figure out number of training examples per partition
    num_images = len(trainset) // num_partitions

    # a list of partition lenghts (all partitions are of equal size)
    partition_len = [num_images] * num_partitions

    # split randomly. This returns a list of trainsets, each with `num_images` training examples
    # Note this is the simplest way of splitting this dataset. A more realistic (but more challenging) partitioning
    # would induce heterogeneity in the partitions in the form of for example: each client getting a different
    # amount of training examples, each client having a different distribution over the labels (maybe even some
    # clients not having a single training example for certain classes). If you are curious, you can check online
    # for Dirichlet (LDA) or pathological dataset partitioning in FL. A place to start is: https://arxiv.org/abs/1909.06335
    trainsets = random_split(
        trainset, partition_len, torch.Generator().manual_seed(2023)
    )

    # create dataloaders with train+val support
    trainloaders = []
    valloaders = []
    # for each train set, let's put aside some training examples for validation
    for trainset_ in trainsets:
        num_total = len(trainset_)
        num_val = int(val_ratio * num_total)
        num_train = num_total - num_val

        for_train, for_val = random_split(
            trainset_, [num_train, num_val], torch.Generator().manual_seed(2023)
        )

        # construct data loaders and append to their respective list.
        # In this way, the i-th client will get the i-th element in the trainloaders list and the i-th element in the valloaders list
        trainloaders.append(
            DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2)
        )
        valloaders.append(
            DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2)
        )

    # We leave the test set intact (i.e. we don't partition it)
    # This test set will be left on the server side and we'll be used to evaluate the
    # performance of the global model after each round.
    # Please note that a more realistic setting would instead use a validation set on the server for
    # this purpose and only use the testset after the final round.
    # Also, in some settings (specially outside simulation) it might not be feasible to construct a validation
    # set on the server side, therefore evaluating the global model can only be done by the clients. (see the comment
    # in main.py above the strategy definition for more details on this)
    testloader = DataLoader(testset, batch_size=128)

    return trainloaders, valloaders, testloader