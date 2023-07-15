import cv2
import numpy as np
import os
import pickle
import yaml

from pathlib import Path
from typing import Union
from collections import defaultdict, namedtuple

from opencv_digit_recognize.utils import split_char


CharInfo = namedtuple('CharInfo', 'image area')

class Learner:
    def __init__(self):
        self.dataset = None
        self.config = None
        self.labels = None
        self.data = defaultdict(list)
        self.model = {}

    def __repr__(self):
        return '\t'.join((key for key in self.model.keys()))

    def load(self, dataset: Union[Path, str], config_path: Union[Path, str]):
        self.dataset = dataset
        with open(os.path.join(dataset, 'label.txt'), 'r') as file:
            self.labels = (tuple(line.strip().split('\t')) for line in file.readlines())

        with open(config_path, 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        self.config = config['Split']

    def learn(self):
        for name, label in self.labels:
            image = cv2.imread(os.path.join(self.dataset, 'image', name))
            contours = split_char(image, self.config)
            for char, info in zip(label, contours):
                self.data[char].append(info)

        for char in self.data.keys():
            canvas = np.zeros((100, 100), np.uint64)
            total_area = 0
            count = len(self.data[char])
            for image, area in self.data[char]:
                canvas += image
                total_area += area

            canvas = np.floor_divide(canvas, count)
            canvas = np.uint8(canvas)
            mean_area = total_area / count

            self.model[char] = CharInfo(canvas, mean_area)

    def show(self):
        cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('show', np.hstack([value.image for value in self.model.values()]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save(self, path='./result.qmodel'):
        assert path.endswith('.qmodel')
        output = {
            'model': self.model,
            'config': self.config
        }
        with open(path, 'wb') as file:
            pickle.dump(output, file)

