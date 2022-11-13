from utils import *
import shutil

def get_args():
    parser = argparse.ArgumentParser(description='Merge Data')
    parser.add_argument('-ds', '--data-dirs', nargs='+', default=[])
    parser.add_argument('-o', '--out-dir', type=str, default='')
    parser.add_argument('-f', '--format', type=str, default='jpg')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    cfg = get_args()
    data_dirs = cfg.data_dirs
    ic(data_dirs)
    out_dir = cfg.out_dir
    format = cfg.format
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    for d in data_dirs:
        for root, dirs, files in tqdm(os.walk(d)):
            for f in files:
                name = root.split('/')[-1]
                img_path = osp.join(root, f)

                name_dir = osp.join(out_dir, name)
                if not osp.exists(name_dir):
                    os.mkdir(name_dir)

                out_img_path = osp.join(name_dir, f[:-3]+format)
                shutil.copy(img_path, out_img_path)



