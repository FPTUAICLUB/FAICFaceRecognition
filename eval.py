from utils import *
from utils.metrics import *
import glob
from sklearn.metrics import accuracy_score
from main import preprocess
import onnxruntime as ort

if __name__ == '__main__':
    eval_dir = '/home/dungmaster/MLProjects/AI-Checkin/data/split_aligned_face_data_aic'
    ort_sess = ort.InferenceSession('/home/dungmaster/MLProjects/AI-Checkin/checkpoints/webface_r50.onnx',
                                    providers=['CUDAExecutionProvider'])

    emb_vecs = []
    labels = []

    for image_path in tqdm(glob.glob(eval_dir+'/train/*/*.jpg')):
        label = image_path.split('/')[-2]

        image = cv2.imread(image_path)
        image = preprocess(image)

        input_name = ort_sess.get_inputs()[0].name
        emb = ort_sess.run([], {input_name: image})[0]

        emb_vecs.append(emb)
        labels.append(label)

    save_pickle(emb_vecs, f'embedding_data/embed_faces.pkl')
    save_pickle(labels, f'embedding_data/labels.pkl')

    embed_faces = np.stack(emb_vecs)
    embed_faces = np.squeeze(embed_faces, axis=1)

    y_preds = []
    y_gt = []

    for image_path in tqdm(glob.glob(eval_dir+'/val/*/*.jpg')):
        label = image_path.split('/')[-2]

        image = cv2.imread(image_path)
        image = preprocess(image)

        input_name = ort_sess.get_inputs()[0].name
        emb = ort_sess.run([], {input_name: image})[0]

        y_pred = most_similarity(embed_faces, emb, labels)
        y_preds.append(y_pred)
        y_gt.append(label)

    acc = accuracy_score(y_preds, y_gt)
    print(f'Valid | acc: {acc:.4f}')
