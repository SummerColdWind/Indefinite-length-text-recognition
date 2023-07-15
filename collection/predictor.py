import numpy as np
import pickle
from operator import truediv
from opencv_digit_recognize.utils import split_char


class Predictor:
    def __init__(self):
        self.active = 'current'
        self.data_dict = {}
        self.model = None
        self.config = None

    def load_model(self, path, name='current'):
        assert path.endswith('.qmodel')
        with open(path, 'rb') as file:
            data = pickle.load(file)
        self.data_dict.update({name: data})
        self.model, self.config = data['model'], data['config']

    def use(self, name='current'):
        self.active = name
        self.model = self.data_dict[name]['model']
        self.config = self.data_dict[name]['config']

    def predict(self, image):
        contours = split_char(image, self.config)
        result = []
        for cnt in contours:
            score, predict_char = 0, None
            for key in self.model.keys():
                char_info = self.model[key]
                templ, mean_area = char_info.image, char_info.area
                delta = np.sum(np.where(templ == cnt.image, 0, 255))
                shape_delta = truediv(*(mean_area, cnt.area)[::1 if mean_area > cnt.area else -1])
                delta = 1 - (delta * shape_delta) / (255 * 100 * 100)
                if delta > score:
                    score, predict_char = delta, key
            result.append(predict_char)

        if len(result) > 0:
            result = ''.join(result)
            return result
