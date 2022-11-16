from utils import *
import numpy as np
from utils.metrics import *
import argparse
import onnxruntime as ort
from unidecode import unidecode
import cv2
from icecream import ic
from utils.audio import *
import os

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition')
    parser.add_argument('-a', '--audio-dir', type=str, default='')
    parser.add_argument('-e', '--enlarge', type=int, default=20)
    args = parser.parse_args()
    return args    

def preprocess(face):
    face = cv2.resize(face, (112, 112))
    face = face.astype(np.float32)
    face /= 255.0
    face = (face - 0.5) / 0.5
    face = face.transpose(2, 0, 1)
    face = face[np.newaxis]
    return face

class Detector:
    def __init__(self, enlarge, audios, threshold=0.5, use_cuda=False):
        self.known_face_embs = np.squeeze(load_pickle('embedding_data/embed_faces.pkl'), axis=1)
        self.known_names = load_pickle('embedding_data/labels.pkl')
        self.thr = threshold
        self.enlarge = enlarge
        self.ort_sess = ort.InferenceSession('checkpoints/webface_r50.onnx', providers=['CUDAExecutionProvider'])
        self.audio_dir = audios

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
        pre_name = None
        cap = cv2.VideoCapture(mode)

        if not cap.isOpened():
            print('Failed to open video')
            return
        
        face = None

        while cap.isOpened():
            ret, self.img = cap.read()
            self.height, self.width = self.img.shape[:2]
            while ret:
                bboxes = self.processFrame()
                
                for bbox in bboxes:
                    face = self.img[bbox[1]:bbox[3],
                                    bbox[0]:bbox[2]]
                    
                    if face.shape[0] == 0:
                        break

                    # ic(face.shape)
                    # face_pr = preprocess(face)
                    face_pr = preprocess(face)
                    
                    # using face recognition model
                    input_name = self.ort_sess.get_inputs()[0].name
                    emb = self.ort_sess.run([], {input_name: face_pr})[0]
                    max_sim, name = most_similarity(self.known_face_embs, emb, self.known_names)

                    if max_sim < 0.3:
                        name = 'Người lạ'

                    if pre_name != name:
                        play(self.audio_dir, name)
                        pre_name = name

                    name = unidecode(name)                    
                    
                    cv2.putText(self.img, name+f' {max_sim:.2f}', (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                1, (255, 0, 255), 2)    
                    cv2.rectangle(self.img, bbox[:2], bbox[2:4], (255, 0, 255), 2)

                if face is not None and face.shape[0] != 0:
                    cv2.imshow('Face', face)
                cv2.imshow('Check In Camera', self.img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                ret, self.img = cap.read()

            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    cfg = get_args()
    enlarge = cfg.enlarge

    audio_dir = cfg.audio_dir
    if not osp.exists(audio_dir):
        os.mkdir(audio_dir)

    det = Detector(enlarge, audio_dir)
    # det.checkInVideo('http://192.168.1.2:4747/video')
    det.checkInVideo(0)
