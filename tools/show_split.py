import cv2
import numpy as np
import os
import yaml

from opencv_digit_recognize.utils import split_char

def get_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = config['Split']
    return config


if __name__ == '__main__':
    dataset = '../dataset/angle'
    config = '../config/angle.yml'

    with open(os.path.join(dataset, 'label.txt'), 'r', encoding='utf-8') as file:
        labels = (tuple(line.strip().split('\t')) for line in file.readlines())
    acc, total, used = 0, 0, 0
    for name, label in labels:
        image = cv2.imread(os.path.join(dataset, 'image', name))
        contours = split_char(image, get_config(config))
        cv2.imshow('show_split', np.hstack([cnt.image for cnt in contours]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


