import custom_training
from utils.model.eval import *
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
from tqdm import tqdm
import glob
from sklearn.metrics import accuracy_score

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('-td', '--train-dir', type=str, default='')
    parser.add_argument('-ed', '--eval-dir', type=str, default='')
    parser.add_argument('-e', '--epochs', type=int, default=20)
    parser.add_argument('-c', '--num-classes', type=int, default=1000)
    parser.add_argument('-b', '--batch-size', type=int, default=32)
    parser.add_argument('-w', '--num-workers', type=int, default=8)
    parser.add_argument('-sd', '--save-ckpt-dir', type=str, default='')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    cfg = get_args()
    train_dir = cfg.train_dir
    eval_dir = cfg.eval_dir

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
        root=osp.join(train_dir, 'train'),
        transform=train_transform
        )
    val_dataset = datasets.ImageFolder(
        root=osp.join(train_dir, 'val'),
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

    # model.eval()
    # custom_training.pass_epoch(
    #     model, loss_fn, val_loader,
    #     batch_metrics=metrics, show_running=True, device=device,
    #     writer=writer
    # )

    # Warm-up
    print('\n\nInitial')
    print('-' * 10)
    model.eval()
    emb_vecs = []
    labels = []

    for image_path in tqdm(glob.glob(eval_dir+'/train/*/*.jpg')):
        label = image_path.split('/')[-2]
        emb = embed_image(image_path, model, val_transform, device)
        emb_vecs.append(emb)
        labels.append(label)

    # save_pickle(emb_vecs, f'/home/dungmaster/MLProjects/AI-Checkin/data/embedding_data/embed_faces.pkl')
    # save_pickle(labels, f'/home/dungmaster/MLProjects/AI-Checkin/data/embedding_data/labels.pkl')

    embed_faces = np.stack(emb_vecs)
    embed_faces = np.squeeze(embed_faces, axis=1)

    y_preds = []
    y_gt = []

    for image_path in tqdm(glob.glob(eval_dir+'/val/*/*.jpg')):
        label = image_path.split('/')[-2]
        emb = embed_image(image_path, model, val_transform, device)
        y_pred = most_similarity(embed_faces, emb, labels)
        y_preds.append(y_pred)
        y_gt.append(label)

    acc = accuracy_score(y_preds, y_gt)
    writer.add_scalars('val_acc', {'Valid': acc}, writer.iteration)
    print(f'Valid | acc: {acc:.4f}')

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
        # custom_training.pass_epoch(
        #     model, loss_fn, val_loader,
        #     batch_metrics=metrics, show_running=True, device=device,
        #     writer=writer
        # )
        emb_vecs = []
        labels = []

        for image_path in tqdm(glob.glob(eval_dir+'/train/*/*.jpg')):
            label = image_path.split('/')[-2]
            emb = embed_image(image_path, model, val_transform, device)
            emb_vecs.append(emb)
            labels.append(label)

        # save_pickle(emb_vecs, f'/home/dungmaster/MLProjects/AI-Checkin/data/embedding_data/embed_faces.pkl')
        # save_pickle(labels, f'/home/dungmaster/MLProjects/AI-Checkin/data/embedding_data/labels.pkl')

        embed_faces = np.stack(emb_vecs)
        embed_faces = np.squeeze(embed_faces, axis=1)

        y_preds = []
        y_gt = []

        for image_path in tqdm(glob.glob(eval_dir+'/val/*/*.jpg')):
            label = image_path.split('/')[-2]
            emb = embed_image(image_path, model, val_transform, device)
            y_pred = most_similarity(embed_faces, emb, labels)
            y_preds.append(y_pred)
            y_gt.append(label)

        acc = accuracy_score(y_preds, y_gt)
        writer.add_scalars('val_acc', {'Valid': acc}, writer.iteration)
        print(f'Valid | acc: {acc:.4f}')

    writer.close()
    torch.save(model.module.base.state_dict(), f'arcface_{cfg.epochs}_scr.pth')