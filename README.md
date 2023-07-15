# Simple indefinite length text recognition
## 基于Opencv的简单不定长文本识别

> Author: github.com/SummerColdWind

> 写在前面：
> 
> 本项目是一个基于Python-Opencv，针对简单的单行不定长文本的识别。
> 本项目旨在为简单场景下的文字识别提供一个轻量级的解决方案。
> 项目基于简单的矩阵运算来决定输出，因此针对复杂场景下的文字识别，
> 你应该使用深度学习的方法来解决任务。

---

## 1.准备数据集

数据集应具有以下结构：
```
---dir_name
 |---image
 |---label.txt
```
数据集文件夹本身的名字可以是任意的，其中应该包含储存图片的文件夹**image**和储存标注信息的文本文件**label.txt**。
标注文件**label.txt**应具有如下格式：
```
1.png	-18
3.png	-18
4.png	-19
5.png	-30
6.png	-29
7.png	-28
8.png	-27
9.png	-26
10.png	-26
11.png	-26
```
每一行为一条数据，用 **'\t'** 进行分割，左侧为**文件名**，右侧为图片中的文字。

## 2.分割测试
使用**tools**文件夹下的**show_split.py**进行字符分割测试。

你应该在**config**文件夹中复制**global.yml**，然后重命名为自己的名字，进行分割参数调整。
```
Split:
  enlarge: 1 
  enhance_contrast: False
  threshold: 100
  adaptedThreshold: False
  threshold_bias: 0
  min_area: 0
  close_morphology: False
  kernel_size: (3,3)
```
  - enlarge: 图片放大倍数
  - enhance_contrast: 是否进行直方图均衡以增强对比度
  - threshold: 二值化图像阈值
  - adaptedThreshold: 是否启用自适应阈值
  - threshold_bias: 自适应阈值基础上进行调整的数值
  - min_area: 轮廓最小面积，低于此面积的轮廓将不视为一个字符
  - close_morphology: 是否进行轮廓闭合操作
  - kernel_size: 闭运算卷积核大小

然后启动**show_split.py**，输入数据集和配置文件的路径，观察分割效果并调整。


## 3.启动学习

导入学习器**Learner**进行模型的学习。
```python
from opencv_digit_recognize.collection import Learner

result_path = './models/angle.qmodel'

learner = Learner()
learner.load(dataset='./dataset/angle', config_path='./config/angle.yml')
learner.learn()
learner.save(result_path)
```
实例化学习器后，指定数据集文件夹，然后使用**learn**方法进行学习，整个过程您无需其他操作。

最后使用**save**方法保存识别得到的模型，我们建议您将文件扩展名设置为“**.qmodel**”。

 - 对测试使用的300+张数据集进行学习所耗费的时间约为0.17s。

>Tips:
> 
> 打印learner可以输出所有学习到的字符列表。
> 
> 调用learner的show方法可以查看学习到的标准化图片。

## 4.进行预测
```python
from opencv_digit_recognize.collection import Predictor

result_path = './models/angle.qmodel'
predict_image = './dataset/angle/image/1.png'

predictor = Predictor()
predictor.load_model(result_path)
result = predictor.predict(predict_image)
print(result)
```
实例化预测器后，使用**load_model**方法导入模型，然后使用**predict**方法进行预测。

 - 对测试使用的300+张数据集进行预测所耗费的平均时间约为0.001s。
