import tensorflow as tf
from src.Net import Net
import numpy as np
import cv2
import time

class Pnet(Net):
    """Pnet网络前向传播"""

    def __init__(self,
                 weight_path,
                 scales,
                 threshold,
                 nms_threshold):
        """初始化，构建网络架构，读取网络权重"""

        super().__init__()

        self.__threshold = threshold

        self.__nms_threshold = nms_threshold

        self.__model = self.__create_model()

        self.__model.load_weights(weight_path, by_name=True)

        self.__scales = scales


    #######################################
    # 主函数，前向传播函数，获得并且处理pnet得到的结果
    # 1.获取网络预测值
    # 2.处理预测结果
    #    1).找到大于阈值的位置
    #    2).找到边框的位置
    #    3).获取偏移量
    #    4).获取得分值
    #######################################
    def forward(self, pnet_need_imgs):
        """前向传播函数，主要接口
        
        这个函数将完成pnet的全过程，输入图片
        获得预测值，并对预测值进行处理
        
        Args:
            pnet_need_imgs:(np.array), shape=(x, h, w, 3)
            
        Returns:
            返回处理得到的预测人脸框. (np.array)
        
        """
        self.width = pnet_need_imgs[0].shape[0]
        self.height = pnet_need_imgs[0].shape[1]

        rectangles = []

        # 传入网络
        out = self.__model.predict(pnet_need_imgs)

        # 获取矩形框
        # shape为(bz, 5)，其中bz为通过pNet检测出来有多少人脸框，5个channel分别代表
        # (left_top_x, left_top_y, right_bottom_x, right_bottom_y, score)
        boundingbox = self.__get_boundingbox(out)
        print('boundingbox shape: ', str(boundingbox.shape))

        if len(boundingbox) == 0:

            self.print_messages("该张图像在pnet网络检测不到人脸")

            return []

        # 将矩形框调整成正方形
        boundingbox = self._rect2square(boundingbox)

        # 避免数值不合理
        boundingbox = self._trimming_frame(boundingbox,
                                           width = self.width,
                                           height = self.height)

        # nms
        boundingbox = self._nms(boundingbox, 0.3)

        self.print_messages("Pnet网络处理完毕")

        return self._nms(boundingbox, self.__nms_threshold)

    #######################################
    # 找到边框位置
    #######################################
    def __get_boundingbox(self, out):
        """这个方法主要用于判断大于阈值的坐标，并且转换成矩形框，为所有缩放的尺寸的图片生成
        bounding box。

        Args:
            out: pNet的输出，包含两个array，分别是[classifier, bbox_regress]。
                classifier shape为[bz, h, w, 2]，表示每个点映射到原始图像上的矩形框是
                否包含人脸的概率。
                bbox_regress shape为[bz, h, w, 4]，表示每个点映射到原始图像上的矩形框
                与真实人脸框的偏移量。

        """

        boundingbox = []

        #scores = []

        for i in range(len(self.__scales)):  # 所有缩放尺寸图片寻找矩形框

            scale = self.__scales[i]  # 当前缩放系数

            # 当前缩放图片pNet输出的包含人脸的概率，shape为[h, w]
            cls_prob = out[0][i, :, :, 1]

            # 获取概率大于阈值的坐标点以及当前层坐标点映射到原始图片的矩形框
            # 假设bbx shape为(w, h, 4), x shape为(h, )，y shape为(w, )
            (x, y), bbx = self.__boundingbox(cls_prob, scale)

            if bbx.shape[0] == 0:
                continue

            # scores shape为(w, h, 1)
            scores = np.array(out[0][i, x, y, 1][np.newaxis, :].T)

            # offset shape为(w, h, 4)
            offset = out[1][i, x, y]*12*(1/scale)

            # bbx shape为(w, h, 5)
            bbx = bbx + offset
            bbx = np.concatenate((bbx, scores), axis=1)

            # # 将矩形框调整成正方形
            # bbx = self._rect2square(bbx)
            #
            # # 避免数值不合理
            # bbx = self._trimming_frame(bbx)
            #
            # # nms
            # bbx = self._nms(bbx, 0.3)


            for b in bbx:
                boundingbox.append(b)

        return np.array(boundingbox)

    def __boundingbox(self, cls_prob, scale):
        """根据当前缩放系数与pNet输出的每个点映射回原始矩形框包含人脸的概率，返回概率大于阈
        值的矩形框数组。

        Args:
            cls_prob: pNet输出的每个点映射回原始矩形框包含人脸的概率。
            scale:当前缩放系数。

        Returns:
            概率大于阈值的点坐标，以及映射回原始图片上的矩形框。

        """

        # 假设cls_prob如下：
        # 例如：
        # [[0.32978287, 0.07975868, 0.31843527, 0.36670653, 0.70001274],
        #  [0.90282534, 0.71664857, 0.83183313, 0.42012622, 0.1784742 ],
        #  [0.32516478, 0.2473772 , 0.58397128, 0.46537295, 0.58577106],
        #  [0.93418164, 0.17123243, 0.58486384, 0.67952699, 0.36251685],
        #  [0.42392291, 0.29361146, 0.89902606, 0.38633181, 0.03290287]]

        # 假设阈值为0.5，x, y表示大于阈值的坐标，如下：
        # x: [0, 1, 1, 1, 2, 2, 3, 3, 3, 4]
        # y: [4, 0, 1, 2, 2, 4, 0, 2, 3, 2]
        x, y = np.where(cls_prob > self.__threshold)

        # bbx:
        # [[4, 0],
        #  [0, 1],
        #  [1, 1],
        #  [2, 1],
        #  [2, 2],
        #  [4, 2],
        #  [0, 3],
        #  [2, 3],
        #  [3, 3],
        #  [2, 4]]
        bbx = np.array((y, x)).T

        # 假设缩放系数是0.5，映射到之前的矩形框则需要除以0.5
        # left_top代表左上角坐标
        # [[16., 0.],
        #  [0., 4.],
        #  [4., 4.],
        #  [8., 4.],
        #  [8., 8.],
        #  [16., 8.],
        #  [0., 12.],
        #  [8., 12.],
        #  [12., 12.],
        #  [8., 16.]]
        left_top = np.fix(((bbx * 2) + 0) * (1/scale))

        # right_down代表右下角坐标
        # [[38., 22.],
        #  [22., 26.],
        #  [26., 26.],
        #  [30., 26.],
        #  [30., 30.],
        #  [38., 30.],
        #  [22., 34.],
        #  [30., 34.],
        #  [34., 34.],
        #  [30., 38.]]
        right_down = np.fix(((bbx * 2) + 11) * (1/scale))

        return (x, y), np.concatenate((left_top, right_down), axis=1)

    #######################################
    # 定义网络架构
    #######################################
    @classmethod
    def __create_model(cls):
        """定义PNet网络的架构"""

        input = tf.keras.Input(shape=[None, None, 3])
        x = tf.keras.layers.Conv2D(10, (3, 3),
                                   strides=1,
                                   padding='valid',
                                   name='conv1')(input)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU1')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(16, (3, 3),
                                   strides=1,
                                   padding='valid',
                                   name='conv2')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU2')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3),
                                   strides=1, padding='valid', name='conv3')(x)
        x = tf.keras.layers.PReLU(shared_axes=[1, 2],
                                  name='PReLU3')(x)


        # shape为[bz, h, w, 2]，表示每个点映射到原始图像上的矩形框是否包含人脸的概率。
        classifier = tf.keras.layers.Conv2D(2, (1, 1),
                                            activation='softmax',
                                            name='conv4-1')(x)

        # shape为[bz, h, w, 4]，表示每个点映射到原始图像上的矩形框与真实人脸框的偏移量。
        bbox_regress = tf.keras.layers.Conv2D(4, (1, 1),
                                              name='conv4-2')(x)

        model = tf.keras.models.Model([input], [classifier, bbox_regress])

        return model

