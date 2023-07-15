import os
import time
import cv2

from opencv_digit_recognize.collection import Learner, Predictor

if __name__ == '__main__':
    result_path = './models/wind.qmodel'

    learner = Learner()
    learner.load(dataset='./dataset/wind', config_path='./config/wind.yml')
    learner.learn()
    print(learner)
    learner.show()
    learner.save(result_path)

    predictor = Predictor()
    predictor.load_model(result_path)

    with open('./dataset/valid/label.txt', 'r', encoding='utf-8') as file:
        labels = (tuple(line.strip().split('\t')) for line in file.readlines())
    acc, total, used = 0, 0, 0
    for name, label in labels:
        image = cv2.imread(os.path.join('./dataset/valid/image', name))
        start = time.perf_counter()
        predict = predictor.predict(image)
        used += time.perf_counter() - start
        if label == predict:
            acc += 1
        else:
            print(label, predict)
        total += 1
    print(f'acc: {(acc / total) * 100:.2f}%, per_used: {(used / total) * 1000:.2f}ms')





