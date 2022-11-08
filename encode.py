import face_recognition
from tqdm import tqdm
import os
import os.path as osp
from icecream import ic
import numpy as np
from utils.data import *
import shutil
import imgaug.augmenters as iaa

data_dir = 'face_data_aic'
# out_data_dir = 'test'
out_data_dir = 'enc_face_data'

if osp.exists(out_data_dir):
    shutil.rmtree(out_data_dir)
os.mkdir(out_data_dir)

names = []
img_encodes = []

for root, dirs, paths in tqdm(os.walk(data_dir)):
    for img_name in paths:
        img_path = osp.join(root, img_name)
        name = img_path.split('/')[1]

        out_dir = osp.join(out_data_dir, name)
        out_path = osp.join(out_dir, img_name.replace('jpg', 'txt'))
        if osp.exists(out_path):
            continue

        img = load_image(img_path) 
        h, w = img.shape[:2]
        encode = face_recognition.face_encodings(img, known_face_locations=[[0, 0, int(h), int(w)]])

        # if len(encode) == 0:
        #     for sigma in np.linspace(0.0, 3.0, num=11).tolist():
        #         seq = iaa.GaussianBlur(sigma)
        #         image_aug = seq.augment_image(img)
        #         encode = face_recognition.face_encodings(image_aug)
        #         if len(encode) > 0:
        #             print('sigma:', sigma)
        #             break
        
        if len(encode) == 0:
            print('Corrupted:', img_path)

        names.append(name)
        img_encodes.append(encode)

        if not osp.exists(out_dir):
            os.mkdir(out_dir)

        # cv2.imwrite(out_path.replace('txt', 'png'), img[0:int(h), 0:int(w)])
        np.savetxt(out_path, encode)

# save_pickle(names, 'data/names.pkl')
# save_pickle(img_encodes, 'data/encodes.pkl')




