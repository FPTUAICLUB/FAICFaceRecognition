import os
# os.chdir('../')
print(os.getcwd())

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn import DataParallel
import numpy as np
import os
import os.path as osp
from tqdm import tqdm
import shutil
import glob
import random
from icecream import ic
import math

data_dir = 'face_aic'
log_file = 'train_logs.txt'
ckpt = 'facenet'
nc = 126 + 1000
loss = 'focal'

batch_size = 16
epochs = 20
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

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

train_dataset = torchvision.datasets.ImageFolder(
    root=osp.join(data_dir, 'train'),
    transform=train_transform
    )
val_dataset = torchvision.datasets.ImageFolder(
    root=osp.join(data_dir, 'val'),
    transform=val_transform
    )

train_loader = DataLoader(
    train_dataset,
    num_workers=workers,
    batch_size=batch_size,
    drop_last=True
)
val_loader = DataLoader(
    val_dataset,
    num_workers=workers,
    batch_size=batch_size,
    drop_last=True
)

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()

model = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=nc
).to(device)


list_layers = list(model.named_children())[-3:]

for param in model.parameters():
    param.requires_grad = False

for layer in list_layers:
    ic(layer)
    for param in layer[1].parameters():
        param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

if loss == 'focal':
    loss_fn = FocalLoss(gamma=2.0)
else:
    loss_fn = nn.CrossEntropyLoss()

metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
model.eval()
training.pass_epoch(
    model, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    model.train()
    training.pass_epoch(
        model, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    model.eval()
    training.pass_epoch(
        model, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()
torch.save(model.state_dict(), ckpt + '_' + loss + '_' + epochs + '.pth')

with open(log_file, 'a+') as f:
    f.write(f'{ckpt} \t {data_dir}')
    f.write('\n')

f.close()