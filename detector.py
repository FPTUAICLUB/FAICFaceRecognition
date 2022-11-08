import cv2
import numpy as np
import os.path as osp
from utils.recognition import *
from icecream import ic

ckpt = ['models/res10_300x300_ssd_iter_140000.prototxt', 
        'models/res10_300x300_ssd_iter_140000.caffemodel']
enc_data_dir = '/home/dungmaster/MLProjects/face_recognition/AI-Checkin/enc_face_data'

class Detector:

    def __init__(self, use_cuda=False):
        # load model
        self.faceModel = cv2.dnn.readNetFromCaffe(ckpt[0],
                                                  caffeModel=ckpt[1])
        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self, imgName):
        self.img = cv2.imread(imgName)
        (self.height, self.width) = self.img.shape[:2]
        self.processFrame()
        cv2.imshow('Output', self.img)
        cv2.waitKey(0)

    def processFrame(self, known_face_encodings, known_face_names):
        bboxes = []
        face_names = []
        # distances = []
        blob = cv2.dnn.blobFromImage(self.img, 1.0, (224, 224), (104.0, 107.0, 123.0), swapRB=False, crop=False)
        self.faceModel.setInput(blob)
        # detect the faces
        predictions = self.faceModel.forward()

        for i in range(0, predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array([self.width, self.height, self.width, self.height])
                (xmin, ymin, xmax, ymax) = bbox.astype('int')
                bboxes.append((xmin, ymin, xmax - xmin, ymax - ymin))
                # face_bbox = [(ymin // 4, math.ceil(xmax / 4), math.ceil(ymax / 4), xmin // 4)]
                face_bbox = [(ymin, xmax, ymax, xmin)]
                label = get_face_label(known_face_encodings, known_face_names, face_bbox, self.img)
                face_names.append(label)
                # distances.append(dist)
                # cv2.rectangle(self.img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        return face_names, bboxes

    def run(self, videoName):
        known_face_encodings, known_face_names = get_from_data(osp.join(enc_data_dir, '*/*.txt'))
        num_known_faces = len(os.listdir(enc_data_dir))
        ic(num_known_faces)
        enlarge = 50
        saving_phase = False

        # cap = cv2.VideoCapture(videoName, cv2.CAP_DSHOW)
        cap = cv2.VideoCapture(videoName)

        if not cap.isOpened():
            print('False to open video')
            return

        while cap.isOpened():
            ret, self.img = cap.read()
            self.height, self.width = self.img.shape[:2]
            while ret:
                key = cv2.waitKey(1) & 0xFF

                face_names, bboxes = self.processFrame(known_face_encodings, known_face_names)

                if len(bboxes) > 0:
                    # name = face_names[0]
                    bbox = bboxes[0]
                    (xmin, ymin, length, height) = bbox

                    # Press 'a' to take picture of new person
                    if key == ord('a'):
                        num_known_faces += 1
                        idx = num_known_faces
                        count = 1
                        print(f'Move to customer {idx}')

                    # Press 's' to save face of the person
                    elif key == ord('s'):
                        saving_phase = True
                        print('Processing')

                    if saving_phase:
                        imgROI = self.img[ymin - enlarge:ymin + height + enlarge,
                                 xmin - enlarge:xmin + length + enlarge]

                        # cv2.rectangle(self.img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                        # cv2.putText(self.img, "Processing", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                        #             2, (0, 0, 255), 2)

                        # encode = face_recognition.face_encodings(imgROI)
                        # if encode:
                        #     cv2.imshow('Crop', imgROI)
                        #     # print(encode)
                        #     if not os.path.exists(f"./checkout_images/raw/{idx}"):
                        #         os.makedirs(f"./checkout_images/raw/{idx}")
                        #     if not os.path.exists(f"./checkout_images/encode/{idx}"):
                        #         os.makedirs(f"./checkout_images/encode/{idx}")
                        #     np.savetxt(f"./checkout_images/encode/{idx}/{count}.txt", encode)
                        #     cv2.imwrite(f"./checkout_images/raw/{idx}/{count}.jpg", imgROI)
                        #     # cv2.rectangle(self.img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
                        #     # cv2.putText(self.img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
                        #     #             2, (0, 0, 255), 2)
                        #     known_face_encodings, known_face_names = get_from_data("./checkout_images/encode/*/*.txt")
                        #     print('Scan saved')
                        #     count += 1
                        #     saving_phase = False

                    cv2.putText(self.img, 'Checkout face', (bbox[0], bbox[1] + bbox[3] + 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 1)

                for bbox, name in zip(bboxes, face_names):
                    cv2.putText(self.img, str(name), (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)

                    cv2.rectangle(self.img, bbox, (255, 0, 255), 2)

                cv2.imshow('Check Out Camera', self.img)

                if key == ord('q'):
                    break
                ret, self.img = cap.read()

            cap.release()
            cv2.destroyAllWindows()