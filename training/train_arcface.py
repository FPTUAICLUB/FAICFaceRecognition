import custom_training
from utils.model.model import *
from utils.model.loss import *
from facenet_pytorch import InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.nn import DataParallel
import os.path as osp
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('-d', '--data-dir', type=str, default='')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-c', '--num-classes', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=16)
    parser.add_argument('-w', '--num-workers', type=int, default=4)
    parser.add_argument('-sd', '--save-ckpt-dir', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    cfg = get_args()

    train_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.RandomRotation(degrees=(30, 70)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # the validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    train_dataset = datasets.ImageFolder(
        root=osp.join(cfg.data_dir, 'train'),
        transform=train_transform
        )
    val_dataset = datasets.ImageFolder(
        root=osp.join(cfg.data_dir, 'val'),
        transform=val_transform
        )

    train_loader = DataLoader(
        train_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        drop_last=True
    )

    resnet = InceptionResnetV1(
        classify=False,
        # pretrained='vggface2',
        pretrained=None,
        num_classes=cfg.num_classes
    )

    model = ArcNet(resnet, cfg.num_classes).to(device)
    model = DataParallel(model)        

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])

    loss_fn = FocalLoss(gamma=2.0)
    # loss_fn = nn.CrossEntropyLoss()
    metrics = {
        'fps': custom_training.BatchTimer(),
        'acc': custom_training.accuracy
    }

    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    model.eval()
    custom_training.pass_epoch(
        model, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(cfg.epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, cfg.epochs))
        print('-' * 10)

        model.train()
        custom_training.pass_epoch(
            model, loss_fn, train_loader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        model.eval()
        custom_training.pass_epoch(
            model, loss_fn, val_loader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    writer.close()
    torch.save(model.module.base.state_dict(), f'arcface_{cfg.epochs}_scr.pth')