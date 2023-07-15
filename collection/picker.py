import cv2
import numpy as np

from collections import defaultdict
from collections.abc import Sequence
from typing import Optional

from opencv_digit_recognize.utils import split_char


class Picker:
    def __init__(self):
        self.image = None
        self.contours = defaultdict(list)

    def __repr__(self):
        return '\n'.join((f'{key}: {len(self.contours[key])}' for key in self.contours.keys()))

    def load(self, image):
        if isinstance(image, str):
            self.image = cv2.imread(image)
        else:
            self.image = image

    def detect(
            self,
            image_slice: Sequence = None,
            index: int = 0
    ) -> Optional[np.ndarray]:
        """
        检测轮廓并返回
        :param image_slice: 图像切片
        :param index: 如果产生了多个轮廓，选择轮廓的索引，默认为第一个
        :return: 轮廓数组
        """
        if image_slice is not None:
            y1, y2, x1, x2 = image_slice
            image = self.image[y1:y2, x1:x2]
        else:
            image = self.image

        return split_char(image)[index]

    def show(self):
        cv2.namedWindow('show', cv2.WINDOW_AUTOSIZE)
        for key in self.contours.keys():
            for cnt in self.contours[key]:
                print(f'Show contour: {key}')
                x, y, w, h = cv2.boundingRect(cnt)
                canvas = np.zeros((y * 2 + h, x * 2 + w, 3), dtype=np.uint8)
                cv2.drawContours(canvas, [cnt], 0, (0, 0, 255), 1)
                cv2.imshow('show', canvas)
                cv2.waitKey(0)
        cv2.destroyAllWindows()

    def update(self, key, contour):
        self.contours[key].append(contour)

    def loads(
            self,
            images: Sequence,
            label: Sequence,
            image_slice: Sequence = None
    ):
        """
        加载数据集，识别并保存轮廓信息
        :param images: 图像或图像路径
        :param label: 标注
        :param image_slice: 图像切片
        :return:
        """
        for i, image in enumerate(images):
            self.load(image)
            contour = self.detect(image_slice)
            self.update(label[i], contour)





if __name__ == '__main__':
    import os

    picker = Picker()
    sliced = (555, 576, 29, 73)
    image_dir = '../dataset'
    image_paths = [os.path.join(image_dir, name) for name in os.listdir(image_dir) if not name.startswith('-')]
    image_paths += [os.path.join(image_dir, '-1.png')]
    picker.loads(image_paths, '0123456789-', sliced)
    # picker.show()
    print(picker)
