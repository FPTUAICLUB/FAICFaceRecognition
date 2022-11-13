from utils import *
import glob
import random
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Split data into train and validation set')
    parser.add_argument('-d', '--data-dir', type=str, default='')
    parser.add_argument('-o', '--out-dir', type=str, default='')
    parser.add_argument('-vr', '--val-ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = get_args()
    data_dir = cfg.data_dir
    out_dir = cfg.out_dir
    ratio = cfg.val_ratio

    image_paths = glob.glob(data_dir + '/*/*.jpg')
    print('Total number of images:', len(image_paths))
    
    n_val = int(len(image_paths) * ratio)
    print('Number of validation images:', n_val)
    val_paths = random.sample(image_paths, n_val)

    os.mkdir(out_dir)
    os.mkdir(out_dir + '/train')
    os.mkdir(out_dir + '/val')

    for path in tqdm(image_paths):
        name, file = path.split('/')[-2:]
        if path in val_paths:
            val_dir = osp.join(out_dir, 'val')
            val_name = osp.join(val_dir, name)
            val_file = osp.join(val_name, file)
            if not osp.exists(val_name):
                os.mkdir(val_name)
            shutil.copy(path, val_file)
        else:
            train_dir = osp.join(out_dir, 'train')
            train_name = osp.join(train_dir, name)
            train_file = osp.join(train_name, file)
            if not osp.exists(train_name):
                os.mkdir(train_name)
            shutil.copy(path, train_file)
