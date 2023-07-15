import os
import time
import cv2

from opencv_digit_recognize.collection import Learner, Predictor

if __name__ == '__main__':
    result_path = './models/angle.qmodel'

    learner = Learner()
    learner.load(dataset='./dataset/angle', config_path='./config/angle.yml')
    learner.learn()
    learner.save(result_path)

    predictor = Predictor()
    predictor.load_model(result_path)

    with open('./dataset/angle/label.txt', 'r') as file:
        labels = (tuple(line.strip().split('\t')) for line in file.readlines())
    acc, total, used = 0, 0, 0
    for name, label in labels:
        image = cv2.imread(os.path.join('./dataset/angle/image', name))
        start = time.perf_counter()
        predict = predictor.predict(image)
        used += time.perf_counter() - start
        if label == predict:
            acc += 1
        total += 1
    print(f'acc: {(acc / total) * 100:.2f}%, per_used: {(used / total) * 1000:.2f}ms')






