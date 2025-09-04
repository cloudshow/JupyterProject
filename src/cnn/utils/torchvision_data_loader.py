import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Optional


def get_dataloaders(
        dataset_name: str,
        data_root: str = './data',
        train_batch_size: int = 64,
        val_batch_size: int = 1000,
        num_workers: int = 4,
        pin_memory: bool = True,
        download: bool = True,
        use_augmentation: bool = True,
        val_split: float = 0.1,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    通用函数：创建训练和验证 DataLoader

    Args:
        dataset_name (str): 数据集名称，支持 'MNIST', 'CIFAR10', 'CIFAR100', 'FashionMNIST', 'SVHN'
        data_root (str): 数据存储根目录
        train_batch_size (int): 训练集 batch size
        val_batch_size (int): 验证集 batch size
        num_workers (int): DataLoader 使用的子进程数
        pin_memory (bool): 是否使用 pinned memory
        download (bool): 是否自动下载数据集
        use_augmentation (bool): 是否使用数据增强（仅训练集）
        val_split (float): 从训练集中划分验证集的比例
        seed (int): 随机种子，确保划分可复现

    Returns:
        Tuple[DataLoader, DataLoader, int, int]:
            train_loader, val_loader, num_classes, input_channels
    """

    # 数据集参数定义（均值、标准差、输入通道、类别数）
    dataset_params = {
        'MNIST': {
            'mean': (0.1307,), 'std': (0.3081,), 'channels': 1, 'num_classes': 10,
            'resize': None
        },
        'FashionMNIST': {
            'mean': (0.2860,), 'std': (0.3530,), 'channels': 1, 'num_classes': 10,
            'resize': None
        },
        'CIFAR10': {
            'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010),
            'channels': 3, 'num_classes': 10,
            'resize': None
        },
        'CIFAR100': {
            'mean': (0.5071, 0.4867, 0.4408), 'std': (0.2675, 0.2565, 0.2761),
            'channels': 3, 'num_classes': 100,
            'resize': None
        },
        'SVHN': {
            'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970),
            'channels': 3, 'num_classes': 10,
            'resize': None
        }
    }

    if dataset_name not in dataset_params:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported: {list(dataset_params.keys())}")

    params = dataset_params[dataset_name]
    num_classes = params['num_classes']
    input_channels = params['channels']
    mean, std = params['mean'], params['std']

    # 基础预处理（验证和测试）
    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]

    if params['resize']:
        base_transform = [transforms.Resize(params['resize'])] + base_transform

    # 训练集变换
    train_transform_list = []
    if use_augmentation and dataset_name in ['CIFAR10', 'CIFAR100']:
        train_transform_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_transform_list.extend(base_transform)
    train_transform = transforms.Compose(train_transform_list)

    # 验证集变换
    val_transform = transforms.Compose(base_transform)

    # 加载数据集
    dataset_class = getattr(datasets, dataset_name)

    if dataset_name == 'SVHN':
        # SVHN 特殊：使用 'train' 和 'extra' 作为训练，'test' 作为测试
        train_dataset = dataset_class(root=data_root, split='train', download=download, transform=train_transform)
        extra_dataset = dataset_class(root=data_root, split='extra', download=download, transform=train_transform)
        train_dataset = torch.utils.data.ConcatDataset([train_dataset, extra_dataset])
        val_dataset = dataset_class(root=data_root, split='test', download=download, transform=val_transform)
    else:
        # 其他数据集
        train_dataset = dataset_class(root=data_root, train=True, download=download, transform=train_transform)
        val_dataset = dataset_class(root=data_root, train=False, download=download, transform=val_transform)

    # 如果需要从训练集中划分验证集
    if val_split > 0:
        # 计算划分大小
        val_size = int(val_split * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
        )
        # 注意：这里我们只取训练部分，验证集仍用原始测试集
        # 如果你想用训练集划分出验证集，可以修改此处逻辑

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader, num_classes, input_channels