import pickle
from PIL import Image
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def save_pickle(obj, file_path):
  with open(file_path, 'wb') as f:
    pickle.dump(obj, f)

def load_pickle(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

def embed_image(path, model, transform, device):
    image = Image.open(path)
    image = transform(image).unsqueeze(0).to(device)
    emb = model(image).detach().cpu().numpy()
    return emb

def most_similarity(embed_vecs, vec, labels):
  sim = cosine_similarity(embed_vecs, vec)
  sim = np.squeeze(sim, axis = 1)
  argmax = np.argsort(sim)[::-1][:1]
  label = [labels[idx] for idx in argmax][0]
  return label

if __name__ == '__main__':
    image = np.array(Image.open('/home/dungmaster/MLProjects/AI-Checkin/data/full_aligned_data/5/3.jpg'))
    cv2.imshow('Test', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()