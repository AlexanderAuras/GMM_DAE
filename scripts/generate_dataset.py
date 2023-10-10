import argparse
import os
from pathlib import Path
import shutil
import tempfile
from typing import cast

import PIL.Image
import torch
import torch.utils.data
import torchvision
import tqdm


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("dataset", choices=["MNIST", "FashionMNIST", "SVHN", "CelebA"], help="The name of the dataset to generate")
    args_parser.add_argument("output_dir", type=Path, help="The target directory of the prepared data")
    args_parser.add_argument("--input_dir", type=Path, help="The directory of the original dataset (will be downloaded if missing), by default a temporary directory is used")
    args = args_parser.parse_args()

    using_tmp_dir = False
    if args.input_dir is None:
        using_tmp_dir = True
        args.input_dir = tempfile.mkdtemp()

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.CenterCrop((140,140)) if args.dataset == "CelebA" else torchvision.transforms.Lambda(lambda x: x),
        torchvision.transforms.Resize((64,64) if args.dataset == "CelebA" else (32, 32), antialias=cast(str, True)),
    ])
    if args.dataset == "MNIST":
        trainval_dataset = torchvision.datasets.MNIST(str(args.input_dir.resolve()), train=True, download=True, transform=transform)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [48000, 12000])
        test_dataset = torchvision.datasets.MNIST(str(args.input_dir.resolve()), train=False, download=True, transform=transform)
    elif args.dataset == "FashionMNIST":
        trainval_dataset = torchvision.datasets.FashionMNIST(str(args.input_dir.resolve()), train=True, download=True, transform=transform)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [48000, 12000])
        test_dataset = torchvision.datasets.FashionMNIST(str(args.input_dir.resolve()), train=False, download=True, transform=transform)
    elif args.dataset == "SVHN":
        trainval_dataset = torchvision.datasets.SVHN(str(args.input_dir.resolve()), split="train", download=True, transform=transform)
        train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [58606, 14651])
        test_dataset = torchvision.datasets.SVHN(str(args.input_dir.resolve()), split="test", download=True, transform=transform)
        extra_dataset = torchvision.datasets.SVHN(str(args.input_dir.resolve()), split="extra", download=True, transform=transform)
    elif args.dataset == "CelebA":
        train_dataset = torchvision.datasets.CelebA(str(args.input_dir.resolve()), split="train", target_type="identity", download=True, transform=transform)
        val_dataset = torchvision.datasets.CelebA(str(args.input_dir.resolve()), split="valid", target_type="identity", download=True, transform=transform)
        test_dataset = torchvision.datasets.CelebA(str(args.input_dir.resolve()), split="test", target_type="identity", download=True, transform=transform)
    
    for subset, dataset in {"train": train_dataset, "val": val_dataset, "test": test_dataset, **({"extra": extra_dataset} if args.dataset == "SVHN" else {})}.items():
        subset_dir = args.output_dir.joinpath(args.dataset, subset)
        subset_dir.mkdir(parents=True, exist_ok=False)
        for i in tqdm.trange(len(dataset), desc=subset):
            if args.dataset in ["MNIST", "FashionMNIST"]:
                img = (dataset[i][0][0]*255.0).to(torch.uint8).numpy()
            elif args.dataset in ["SVHN", "CelebA"]:
                img = (dataset[i][0].permute(1,2,0)*255.0).to(torch.uint8).numpy()
            if args.dataset == "CelebA":
                id_ = dataset[i][1].item()
                target_dir = subset_dir.joinpath(str(id_))
                target_dir.mkdir(parents=True, exist_ok=True)
            else:
                target_dir = subset_dir
            PIL.Image.fromarray(img).save(str(target_dir.joinpath(str(i)+".png").resolve()))
    
    if using_tmp_dir:
        shutil.rmtree(args.input_dir)
