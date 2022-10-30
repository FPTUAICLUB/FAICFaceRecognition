import os
# os.chdir('../')
print(os.getcwd())

from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, custom_training
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
nc = 1000+126

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

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, train, label=False):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        if train:
            one_hot = torch.zeros(cosine.size(), device='cuda')
            one_hot.scatter_(1, label.cuda().view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        else:
            output = cosine
        output *= self.s

        return output

resnet = InceptionResnetV1(
    classify=False,
    # pretrained='vggface2',
    pretrained=None,
    num_classes=nc
)

# list_layers = list(resnet.named_children())[-3:]

# for param in resnet.parameters():
#     param.requires_grad = False

# for layer in list_layers:
#     ic(layer)
#     for param in layer[1].parameters():
#         param.requires_grad = True

class ArcNet(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.base = net
        self.arcface = ArcMarginProduct(512, nc, s=30, m=0.5, easy_margin=False)
    
    def forward(self, x, label):
        x = self.base(x)
        if self.training:
            x = self.arcface(x, self.training, label)
        else:
            x = self.arcface(x, self.training)
        return x

model = ArcNet(resnet).to(device)
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

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
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
torch.save(model.module.base.state_dict(), f'arcface_{epochs}_scr.pth')