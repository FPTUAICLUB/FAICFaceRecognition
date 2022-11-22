import cv2
from facenet_pytorch import MTCNN
import torch
import os

device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def add_new_user(cap, mode, DATA_PATH='/home/khamduong/Study/AI-Checkin/test'):
    count = 50
    usr_name = input("Input ur name: ")
    USR_PATH = os.path.join(DATA_PATH, usr_name)
    leap = 1

    mtcnn = MTCNN(margin = 20, keep_all=False, post_process=False, device = device)
    cap = cv2.VideoCapture(mode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    while cap.isOpened() and count:
        isSuccess, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if mtcnn(frame) is not None and leap%2:
            path = str(USR_PATH+f'/{count}.jpg')
            face_img = mtcnn(frame, save_path = path)
            count-=1
        leap+=1
        # cv2.imshow('Face Capturing', frame)
        if cv2.waitKey(1)&0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
