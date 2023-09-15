import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from celeba import CelebA
import os


class ImageDataset(object):
    def __init__(self, args):
        if args.dataset.lower() == 'cifar10':
            Dt = datasets.CIFAR10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            args.n_classes = 10
        elif args.dataset.lower() == 'stl10':
            Dt = datasets.STL10
            transform = transforms.Compose([
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        elif args.dataset.lower() == 'celeba':
            Dt = CelebA
            transform = transforms.Compose([
                transforms.Resize(size=(args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        else:
            raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

        if args.dataset.lower() == 'stl10':
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='train+unlabeled', transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split='test', transform=transform),
                batch_size=args.dis_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        elif args.dataset.lower() == 'celeba':   
            data_dir = os.path.join(args.data_path,args.dataset)
            self.train = torch.utils.data.DataLoader(
                Dt(root=data_dir, transform=transform),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=data_dir, transform=transform),
                batch_size=args.dis_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid     
        else:
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True),
                batch_size=args.dis_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True)

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)

            self.test = self.valid
        