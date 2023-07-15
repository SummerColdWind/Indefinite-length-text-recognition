import cv2
import numpy as np
from collections import namedtuple
from typing import List

CharInfo = namedtuple('CharInfo', 'image area')


def sort_char(contours):
    return sorted(contours, key=lambda x: cv2.boundingRect(x)[0])


def split_char(
        image,
        config: dict,
) -> List[CharInfo]:
    enlarge, thresh, is_adapted, bias, min_area, is_closed, kernel, is_clahe = \
        config['enlarge'], config['threshold'], config['adaptedThreshold'], config['threshold_bias'], \
        config['min_area'], config['close_morphology'], config['kernel_size'], config['enhance_contrast']
    image = cv2.resize(image, None, fx=enlarge, fy=enlarge)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if is_clahe:
        gray = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(gray)
    # gray = cv2.bilateralFilter(gray, 9, 10, 10)
    if not is_adapted:
        edges = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        edges = cv2.threshold(gray, thresh + bias, 255, cv2.THRESH_BINARY)[1]
    if is_closed:
        kernel = tuple(int(size) for size in kernel.strip('()').split(','))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones(kernel, np.uint8))
    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    contours = sort_char(contours)
    result = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        h0, w0 = image.shape[:2]
        if area > min_area and x > 10 and y > 10 and (w0 - x - w) > 10 and (h0 - y - h) > 10:
            cv2.drawContours(edges, [cnt], 0, 255, 1)
            canvas = edges[y:y + h, x:x + w]
            canvas = cv2.resize(canvas, (100, 100))

            result.append(CharInfo(canvas, area))

    return result
