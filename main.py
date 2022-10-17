import cv2, random, os, shutil, string
from collections import Counter
import albumentations as A
from os.path import isfile, join


def open_file(file_name: str):
    with open(file_name, "r") as file:
        obj = map(
            lambda x: (int(x[1]), x[0]),
            map(lambda x: x.strip().split(","), file.readlines()[1:]),
        )
    return list(obj)


def create_train_dir(name: str):
    if not os.path.isdir(f".\\{name}"):
        os.mkdir(f".\\{name}")
    else:
        pass
    return f".\\{name}"


def create_train_subdir(directory: str, obj):
    for i in sorted(obj):
        if not os.path.isdir(f".\\{directory}\\{i[0]}"):
            os.mkdir(f".\\{directory}\\{i[0]}")
        else:
            pass


def copy_to_train_subdir(source: str, destination: str, obj):
    for i in sorted(obj):
        file = f".\\{destination}\\{i[0]}\\{i[1]}"
        if not os.path.exists(file):
            shutil.copy2(f".\\{source}\\{i[1]}", file)
        else:
            pass


def subdir_lenofelement_pairs(obj):
    lst = (subdir[0] for subdir in obj)
    return sorted(dict(Counter(lst)).items())


def length_of_current_dir(directory):
    return len(
        [
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        ]
    )


def define_aumentation_pipeline(transform="light"):
    match transform:
        case "light":
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.7),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.CLAHE(p=1),
                ],
                p=1,
            )
        case "medium":
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.7),
                    A.CLAHE(p=1),
                    A.HueSaturationValue(
                        hue_shift_limit=20, sat_shift_limit=50, val_shift_limit=50, p=1
                    ),
                ],
                p=1,
            )
        case "strong":
            transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.7),
                    A.ChannelShuffle(p=1),
                ],
                p=1,
            )

    return transform


def convert_image(path: str):
    image = cv2.imread(path)
    return image


def generate_augmentation_image(path_image):
    transforms = define_aumentation_pipeline(
        random.choice(["light", "medium", "strong"])
    )
    augmented_image = transforms(image=convert_image(path_image))["image"]
    return augmented_image


#  def generate_augmentation_image_name(
#      size=32, chars=string.ascii_uppercase + string.digits
#  ):
#      return "".join(random.choice(chars) for _ in range(size))


def add_single_augmentation_image_to_dir(save_path: str, path_image: str):
    augmented_image = generate_augmentation_image(path_image)
    return cv2.imwrite(save_path, augmented_image)


def add_multiple_aumentation_images_to_dir(path_images, save_dir, maximum_picture=16):
    count = length_of_current_dir(save_dir)
    while count < maximum_picture:
        for path_image in path_images:
            count += 1
            print(count, path_image)
            add_single_augmentation_image_to_dir(
                f".\\{save_dir}\\{count}.png", path_image
            )
            if count == maximum_picture:
                break


#  def rename_file_train_subdir(path_images, save_dir):
#      count = 0
#      while count < length_of_current_dir(save_dir):
#          for path_image in path_images:
#              count += 1
#              if not os.path.exists(f".\\{save_dir}\\{count}.png"):
#                  os.rename(path_image, f".\\{save_dir}\\{count}.png")
#              else:
#                  pass
#              if count == length_of_current_dir(save_dir):
#                  break


def main():
    obj = open_file("train.csv")
    train_dir = create_train_dir("train_file")
    create_train_subdir("train_file", obj)
    copy_to_train_subdir("train", "train_file", obj)
    lst_dir = map(lambda x: f"{train_dir}\\{x}", os.listdir(".\\train_file"))
    for save_dir in lst_dir:
        # Create a list of image path each subdir
        path_images = [
            join(save_dir, f) for f in os.listdir(save_dir) if isfile(join(save_dir, f))
        ]
        add_multiple_aumentation_images_to_dir(path_images, save_dir)

    #  for save_dir in list(lst_dir):
    #      # Create a list of image path each subdir
    #      path_images = [
    #          join(save_dir, f) for f in os.listdir(save_dir) if isfile(join(save_dir, f))
    #      ]
    #      rename_file_train_subdir(path_images, save_dir)


if __name__ == "__main__":
    main()
