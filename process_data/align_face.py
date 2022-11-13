# import the necessary packages
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import argparse
import imutils
import dlib
import cv2
from icecream import ic
import os.path as osp
import os
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Align Face')
    parser.add_argument('-d', '--data-dir', type=str, default='')
    parser.add_argument('-o', '--out-dir', type=str, default='')
    parser.add_argument('-s', '--size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg = get_args()
    data_dir = cfg.data_dir
    out_data_dir = cfg.out_dir
    predictor = '/home/dungmaster/MLProjects/AI-Checkin/checkpoints/shape_predictor_68_face_landmarks.dat'
    size = cfg.size

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor)
    fa = FaceAligner(predictor, desiredFaceWidth=size)

    if not osp.exists(out_data_dir):
        os.mkdir(out_data_dir)

    for root, dirs, files in tqdm(os.walk(data_dir)):
        for f in files:
            name = root.split('/')[-1]

            img_path = osp.join(root, f)
            image = cv2.imread(img_path)

            image = imutils.resize(image, width=size)   
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faceAligned = fa.align(image, gray, dlib.rectangle(int(0), int(0), size, size))

            name_dir = osp.join(out_data_dir, name)
            if not osp.exists(name_dir):
                os.mkdir(name_dir)

            out_img_path = osp.join(name_dir, f)
            cv2.imwrite(out_img_path, faceAligned)


