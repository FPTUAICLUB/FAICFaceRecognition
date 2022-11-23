from utils import *
import numpy as np
from utils.metrics import *
import argparse
import onnxruntime as ort
from unidecode import unidecode
import shutil
import cv2
from icecream import ic
from utils.audio import *
import os
import time
from main import preprocess

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('-e', '--enlarge', help='Enlarging face bboxes in 4 directions', type=int, default=20)
    parser.add_argument('-n', '--name', help='Name of saved face',type=str, default='')
    parser.add_argument('-r', '--restart', help='Restart saving faces from zero', action='store_true')
    parser.add_argument('-s', '--save-embeddings', help='Save face embedding', action='store_true')
    args = parser.parse_args()
    return args    

class Detector:
    def __init__(self, enlarge, name, restart, save_embeddings=False, threshold=0.5, use_cuda=True):

        self.name = name
        self.thr = threshold
        self.enlarge = enlarge
        self.restart = restart
        self.save_embeddings = save_embeddings
        if save_embeddings:
            self.known_face_embs = load_pickle('embedding_data/embed_faces.pkl')
            self.known_names = load_pickle('embedding_data/labels.pkl')
            self.ort_sess = ort.InferenceSession('checkpoints/webface_r50.onnx', providers=['CUDAExecutionProvider'])
            ic(len(self.known_face_embs))

        # load model
        self.faceModel = cv2.dnn.readNetFromCaffe('checkpoints/res10_300x300_ssd_iter_140000.prototxt',
                                                  caffeModel='checkpoints/res10_300x300_ssd_iter_140000.caffemodel')
        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processFrame(self):
        bboxes = []
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (300, 300), (104.0, 107.0, 123.0), swapRB=False, crop=False)
        self.faceModel.setInput(blob)

        # detect the faces
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > self.thr:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                # bbox = predictions[0, 0, i, 3:7]
                xmin, ymin, xmax, ymax = bbox.astype('int') # xmin, ymin, xmax, ymax
                bboxes.append([xmin-self.enlarge, ymin-self.enlarge, xmax+self.enlarge, ymax+self.enlarge])

        return bboxes

    def checkInVideo(self, mode):
        new_face_dir = osp.join('new_faces', self.name)
        if restart:
            shutil.rmtree(new_face_dir)
        os.makedirs(new_face_dir, exist_ok=True)

        t1 = time.time()
        face = None
        count = len(os.listdir(new_face_dir))+1
        limit = count + 4

        cap = cv2.VideoCapture(mode)

        if not cap.isOpened():
            print('Failed to open video')
            return

        while cap.isOpened():
            ret, self.img = cap.read()

            self.height, self.width = self.img.shape[:2]
            while count <= limit:
                t2 = time.time()
                bboxes = self.processFrame()
                
                for bbox in bboxes:
                    face = self.img[bbox[1]:bbox[3],
                                    bbox[0]:bbox[2]]
                                            

                    cv2.rectangle(self.img, bbox[:2], bbox[2:4], (255, 0, 255), 2)

                if face is not None and face.shape[0]*face.shape[1] != 0 and t2 - t1 >= 1 and self.save_embeddings:
                    face_pr = preprocess(face)
                    input_name = self.ort_sess.get_inputs()[0].name
                    emb = self.ort_sess.run([], {input_name: face_pr})[0]
                    # self.known_face_embs = np.append(self.known_face_embs, emb, axis=0)
                    self.known_face_embs.append(emb)
                    self.known_names.append(self.name)

                    save_pickle(self.known_face_embs, 'embedding_data/embed_faces.pkl')
                    save_pickle(self.known_names, 'embedding_data/labels.pkl')

                    cv2.imwrite(osp.join(new_face_dir, str(count)+'.jpg'), face)
                    count += 1
                    t1 = t2

                cv2.imshow('Check In Camera', self.img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                ret, self.img = cap.read()

            ic(len(self.known_face_embs))
            cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    cfg = get_args()
    enlarge = cfg.enlarge
    save_name = cfg.name
    restart = cfg.restart
    save_embeddings = cfg.save_embeddings
    det = Detector(enlarge, save_name, restart, save_embeddings)
    det.checkInVideo(0)
