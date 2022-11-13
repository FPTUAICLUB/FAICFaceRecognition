from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import os
import torch
import glob
import os.path as osp
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import pickle
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

resnet = InceptionResnetV1(
    classify=False,
    pretrained=None,
    num_classes=126
).to(device)
resnet.eval()

resnet.load_state_dict(torch.load('/workspaces/Phe_AutoParking/AI-Checkin/arcface_20_scr.pth'))

def _save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def _load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def _embed_image(path, model):
    image = Image.open(path)
    image = val_transform(image).unsqueeze(0).to(device)
    emb = model(image).detach().cpu().numpy()
    return emb

emb_vecs = []
labels = []

for image_path in tqdm(glob.glob('/workspaces/Phe_AutoParking/AI-Checkin/face_aic/train/*/*.*')):
    label = image_path.split('/')[-2]
    emb = _embed_image(image_path, resnet)
    emb_vecs.append(emb)
    labels.append(label)

_save_pickle(emb_vecs, '/workspaces/Phe_AutoParking/AI-Checkin/data/arcface_embed_faces.pkl')
_save_pickle(labels, "/workspaces/Phe_AutoParking/AI-Checkin/data/arcface_labels.pkl")

embed_faces = np.stack(_load_pickle("/workspaces/Phe_AutoParking/AI-Checkin/data/arcface_embed_faces.pkl"))
embed_faces = np.squeeze(embed_faces, axis=1)
y_labels = _load_pickle("/workspaces/Phe_AutoParking/AI-Checkin/data/arcface_labels.pkl")

def _most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

y_preds = []
y_gt = []

for image_path in tqdm(glob.glob('/workspaces/Phe_AutoParking/AI-Checkin/face_aic/val/*/*.*')):
  label = image_path.split('/')[-2]
  vec = _embed_image(image_path, resnet)
  y_pred = _most_similarity(embed_faces, vec, y_labels)
  y_preds.append(y_pred)
  y_gt.append(label)

print(accuracy_score(y_preds, y_gt))