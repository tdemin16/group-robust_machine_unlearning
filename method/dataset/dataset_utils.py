import torch
from torchvision.transforms import v2 as transforms
from typing import Tuple

from method.dataset.dataset_classes import (
    CelebA,
    FairFace,
    Imagenet_R,
    MultiNLI,
    WaterBird,
    BiasUnlearningDataset,
    RandomUnlearningDataset,
)


def get_transforms(args) -> Tuple[transforms.Compose, transforms.Compose]:
    if args.dataset in ("celeba"):
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)

        if "vit" in args.model:
            args.size = 224

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize((args.size, args.size), antialias=None),
                *base_transform,
            ]
        )

    elif args.dataset in ("fairface", "imagenetr", "waterbird"):
        args.size = 224
        args.mean = (0.5, 0.5, 0.5)
        args.std = (0.5, 0.5, 0.5)

        base_transform = [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(args.mean, args.std),
        ]

        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(args.size, antialias=None),
                transforms.RandomHorizontalFlip(),
                *base_transform,
            ]
        )
        test_transform = transforms.Compose(
            [
                transforms.Resize(args.size * 256 // 224, antialias=None),
                transforms.CenterCrop(args.size),
                *base_transform,
            ]
        )
    
    elif args.dataset == "multinli":
        args.size = 128
        train_transform = None
        test_transform = None

    return train_transform, test_transform


def get_datasets(args, splits: dict) -> BiasUnlearningDataset | RandomUnlearningDataset:
    """
    Returns a dataset manager:

    Returns:
    ------
    datasets: IdentityUnlearningDataset
    """
    train_transform, test_transform = get_transforms(args)

    UnlearningDataset = (
        BiasUnlearningDataset if args.unlearning_type == "bias" else RandomUnlearningDataset
    )
    if args.dataset == "celeba":
        dataset = CelebA
        dataset_args = {
            "root": args.data_dir,
            "target_type": ["attr"],
            "download": args.download,
            "split_number": args.split_number,
        }
    elif args.dataset == "fairface":
        dataset = FairFace
        dataset_args = {
            "root": args.data_dir,
            "split_number": args.split_number,
            "num_unlearning_groups": args.multi_group,
        }
    elif args.dataset == "waterbird":
        dataset = WaterBird
        dataset_args = {"root": args.data_dir, "split_number": args.split_number}
    elif args.dataset == "imagenetr":
        dataset = Imagenet_R
        dataset_args = {"root": args.data_dir, "download": args.download}
    elif args.dataset == "multinli":
        dataset = MultiNLI
        dataset_args = {"root": args.data_dir}
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
        
    dataset_args["tmlr_rebuttal_exp"] = args.tmlr_rebuttal_exp

    datasets = UnlearningDataset(
        dataset=dataset,
        train_transform=train_transform,
        test_transform=test_transform,
        dataset_args=dataset_args,
        unlearning_ratio=args.unlearning_ratio,
        splits=splits,
        dataset_name=args.dataset,
        tmlr_rebuttal_exp=args.tmlr_rebuttal_exp,
    )

    args.num_classes = datasets.get_num_classes()

    return datasets
