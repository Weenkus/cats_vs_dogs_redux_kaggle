import glob
import os
import re
import numpy as np
import PIL
from PIL import Image


class Preprocessor(object):
    TRAIN_DIR = '../dataset/train/'
    TEST_DIR = '../dataset/test/'


    @staticmethod
    def natural_key(string_):
        """
        Define sort key that is integer-aware
        """
        return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

    @staticmethod
    def norm_image(image):
        """
        Normalize PIL image

        Normalizes luminance to (mean,std)=(0,1), and applies a [1%, 99%] contrast stretch
        """
        img_y, img_b, img_r = image.convert('YCbCr').split()

        img_y_np = np.asarray(img_y).astype(float)

        img_y_np /= 255
        img_y_np -= img_y_np.mean()
        img_y_np /= img_y_np.std()
        scale = np.max([np.abs(np.percentile(img_y_np, 1)), np.abs(np.percentile(img_y_np, 99))])

        img_y_np /= scale
        img_y_np = np.clip(img_y_np, -1.0, 1.0)
        img_y_np = (img_y_np + 1.0) / 2.0

        img_y_np = (img_y_np * 255 + 0.5).astype(np.uint8)
        img_y = Image.fromarray(img_y_np)
        img_ybr = Image.merge('YCbCr', (img_y, img_b, img_r))

        img_nrm = img_ybr.convert('RGB')
        return img_nrm

    @staticmethod
    def resize_image(image, size):
        """
        Resize PIL image

        Resizes image to be square with sidelength size. Pads with black if needed.
        """
        # Resize
        n_x, n_y = image.size
        if n_y > n_x:
            n_y_new = size
            n_x_new = int(size * n_x / n_y + 0.5)
        else:
            n_x_new = size
            n_y_new = int(size * n_y / n_x + 0.5)

        img_res = image.resize((n_x_new, n_y_new), resample=PIL.Image.BICUBIC)

        # Pad the borders to create a square image
        img_pad = Image.new('RGB', (size, size), (128, 128, 128))
        ulc = ((size - n_x_new) // 2, (size - n_y_new) // 2)
        img_pad.paste(img_res, ulc)

        return img_pad

    @staticmethod
    def get_dataset_paths(train_dir=TRAIN_DIR, test_dir=TEST_DIR, key=natural_key):
        train_cats = sorted(glob.glob(os.path.join(train_dir, 'cat*.jpg')), key=key)
        train_dogs = sorted(glob.glob(os.path.join(train_dir, 'dog*.jpg')), key=key)
        train_all = train_cats + train_dogs

        test_all = sorted(glob.glob(os.path.join(test_dir, '*.jpg')), key=key)
        return train_cats, train_dogs, train_all, test_all

