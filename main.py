import tensorflow as tf
import cv2
from src.MTCNN import MTCNN
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建MTCNN对象
mtcnn = MTCNN(threshold=[0.5, 0.6, 0.7], # 置信度阈值
              nms_threshold=[0.7, 0.7, 0.7], # nms阈值
              weight_paths=['./weight_path/pnet.h5', # 权重文件路径
                            './weight_path/rnet.h5',
                            './weight_path/onet.h5'],
              max_face=False, # 是否检测最大人脸
              save_face=True, # 是否保存检测到的人脸
              save_dirt="./output", # 保存人脸的路径
              print_time=True, # 是否打印时间信息
              print_message=True, # 是否打印辅助信息
              detect_hat=True, # 是否检测安全帽佩戴
              resize_type='without_loss', # resize图片类型
              padding_type='mask' # 填充图片类型
              )

# 读取图片
file_path = 'test3.jpg'
image = cv2.imread(file_path)

# 传入网络
rects, landmarks, ret = mtcnn.detect_face(image)
print('rects: ', str(rects))
print('landmarks: ', str(landmarks))

if ret == False:
    # 未检测到人脸
    print("该图片未检测到人脸")

else:
    img_ = image.copy()
    # 人脸框
    for rect in rects:
        img_ = cv2.rectangle(img_,
                            (int(rect[0]), int(rect[1])),
                            (int(rect[2]), int(rect[3])),
                            (0, 255, 0), 4)
    # 人脸关键位置点
    for landmark in landmarks:
        for mark in landmark:
            mark = tuple(mark)
            # img_ = cv2.drawMarker(img_, mark, (0, 0, 255))
            img_ = cv2.circle(img_, (int(mark[0]), int(mark[1])), 2,
                              (0, 0, 255), cv2.FILLED)
    cv2.imshow("img_", img_)
    cv2.imwrite(os.path.splitext(file_path)[0] + '_prediction.jpg', img_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
