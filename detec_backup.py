import os
import config
import numpy as np
import tensorflow as tf
import random
import colorsys
from PIL import Image, ImageFont, ImageDraw
# 指定使用GPU的Index
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_index


def letterbox_image(image, size):
    """
    Introduction
    ------------
        对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
    Parameters
    ----------
        image: 输入图像
        size: 图像大小
    Returns
    -------
        boxed_image: 缩放后的图像
    """
    image_w, image_h = image.size
    w, h = size
    new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
    resized_image = image.resize((new_w,new_h), Image.BICUBIC)

    boxed_image = Image.new('RGB', size, (128, 128, 128))
    boxed_image.paste(resized_image, ((w-new_w)//2,(h-new_h)//2))
    return boxed_image


class yolo:
    def __init__(self):
        self.load_op = self.load_weights(tf.global_variables(scope='predict'), weights_file=config.yolo3_weights_path)
        self.norm_epsilon = config.norm_epsilon
        self.norm_decay = config.norm_decay
        self.anchors_path = config.anchors_path
        self.classes_path = config.classes_path
        self.pre_train = False
        self.anchors = self._get_anchors()
        self.classes = self._get_class()
        self.obj_threshold = config.obj_threshold
        self.nms_threshold = config.nms_threshold
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)
        self.input_image = tf.placeholder(shape=[None, config.input_shape, config.input_shape, 3], dtype=tf.float32)

        self.boxes, self.scores, self.classes = self.predict(self.input_image)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        # self.saver.restore(self.sess, 1)


    def load_weights(self, var_list, weights_file):
        """
        Introduction
        ------------
            加载预训练好的darknet53权重文件
        Parameters
        ----------
            var_list: 赋值变量名
            weights_file: 权重文件
        Returns
        -------
            assign_ops: 赋值更新操作
        """
        with open(weights_file, "rb") as fp:
            _ = np.fromfile(fp, dtype=np.int32, count=5)

            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        i = 0
        assign_ops = []
        while i < len(var_list) - 1:
            var1 = var_list[i]
            var2 = var_list[i + 1]
            # do something only if we process conv layer
            if 'conv2d' in var1.name.split('/')[-2]:
                # check type of next layer
                if 'batch_normalization' in var2.name.split('/')[-2]:
                    # load batch norm params
                    gamma, beta, mean, var = var_list[i + 1:i + 5]
                    batch_norm_vars = [beta, gamma, mean, var]
                    for var in batch_norm_vars:
                        shape = var.shape.as_list()
                        num_params = np.prod(shape)
                        var_weights = weights[ptr:ptr + num_params].reshape(shape)
                        ptr += num_params
                        assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

                    # we move the pointer by 4, because we loaded 4 variables
                    i += 4
                elif 'conv2d' in var2.name.split('/')[-2]:
                    # load biases
                    bias = var2
                    bias_shape = bias.shape.as_list()
                    bias_params = np.prod(bias_shape)
                    bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
                    ptr += bias_params
                    assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

                    # we loaded 1 variable
                    i += 1
                # we can load weights of conv layer
                shape = var1.shape.as_list()
                num_params = np.prod(shape)

                var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                # remember to transpose to column-major
                var_weights = np.transpose(var_weights, (2, 3, 1, 0))
                ptr += num_params
                assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
                i += 1

        return assign_ops


    def _get_class(self):
        """
        Introduction
        ------------
            获取类别名字
        Returns
        -------
            class_names: coco数据集类别对应的名字
        """
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def _get_anchors(self):
        """
        Introduction
        ------------
            获取anchors
        """
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def _batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bn_layer: batch normalization处理之后的feature map
        '''
        bn_layer = tf.layers.batch_normalization(inputs = input_layer,
            momentum = norm_decay, epsilon = norm_epsilon, center = True,
            scale = True, training = training, name = name)
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)


    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            strides: 卷积步长
            name: 卷积层名字
            trainging: 是否为训练过程
            use_bias: 是否使用偏置项
            kernel_size: 卷积核大小
        Returns
        -------
            conv: 卷积之后的feature map
        """
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4), use_bias = use_bias, name = name)
        return conv


    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            trainging: 是否为训练过程
            blocks_num: block的数量
            conv_index: 为了方便加载预训练权重，统一命名序号
            weights_dict: 加载预训练模型的权重
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            inputs: 经过残差网络处理后的结果
        """
        # 在输入feature map的长宽维度进行padding
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self._conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index


    def _darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs: 模型输入变量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            weights_dict: 预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv: 经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1: 返回第26层卷积计算结果52x52x256, 供后续使用
            route2: 返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        with tf.variable_scope('darknet53'):
            conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index


    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        return route, conv, conv_index


    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs: 模型的输入变量
            num_anchors: 每个grid cell负责检测的anchor数量
            num_classes: 类别数量
            training: 是否为训练模式
        """
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = 'route_0')
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        return [conv2d_59, conv2d_67, conv2d_75]


    def yolo_head(self, feats, anchors, num_classes, input_shape, training = True):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x1024, 26x26x512, 52x52x256
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype = tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)
        #将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        #将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs


    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        Introduction
        ------------
            该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
        Parameters
        ----------
            feats: 模型输出feature map
            anchors: 模型anchors
            num_classes: 数据集类别数
            input_shape: 训练输入图片大小
            image_shape: 原始图片的大小
        """
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape, training = False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis = -1
        )
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores


    def box_iou(self, box1, box2):
        """
        Introduction
        ------------
            计算box tensor之间的iou
        Parameters
        ----------
            box1: shape=[grid_size, grid_size, anchors, xywh]
            box2: shape=[box_num, xywh]
        Returns
        -------
            iou:
        """
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou


    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh = .5):
        """
        Introduction
        ------------
            yolo模型的损失函数
        Parameters
        ----------
            yolo_output: yolo模型的输出
            y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
            anchors: yolo模型对应的anchors
            num_classes: 类别数量
            ignore_thresh: 小于该阈值的box我们认为没有物体
        Returns
        -------
            loss: 每个batch的平均损失值
            accuracy
        """
        loss = 0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = [416.0, 416.0]
        grid_shapes = [tf.cast(tf.shape(yolo_output[l])[1:3], tf.float32) for l in range(3)]
        for index in range(3):
            # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
            # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]
            object_mask = y_true[index][..., 4:5]
            class_probs = y_true[index][..., 5:]
            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]], num_classes, input_shape, training = True)
            # pred_box的shape为[batch, box_num, 4]
            pred_box = tf.concat([pred_xy, pred_wh], axis = -1)
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
            object_mask_bool = tf.cast(object_mask, dtype = tf.bool)
            raw_true_wh = tf.log(tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0), tf.ones_like(y_true[index][..., 2:4]), y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
            # 该系数是用来调整box坐标loss的系数
            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
            ignore_mask = tf.TensorArray(dtype = tf.float32, size = 1, dynamic_size = True)
            def loop_body(internal_index, ignore_mask):
                # true_box的shape为[box_num, 4]
                true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4], object_mask_bool[internal_index, ..., 0])
                iou = self.box_iou(pred_box[internal_index], true_box)
                # 计算每个true_box对应的预测的iou最大的box
                best_iou = tf.reduce_max(iou, axis = -1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask
            _, ignore_mask = tf.while_loop(lambda internal_index, ignore_mask : internal_index < tf.shape(yolo_output[0])[0], loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis = -1)
            # 计算四个部分的loss
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels = raw_true_xy, logits = predictions[..., 0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) + (1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels = object_mask, logits = predictions[..., 4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels =  class_probs, logits = predictions[..., 5:])
            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss


    def eval(self, yolo_outputs, image_shape, max_boxes = 20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_outputs[0])[1 : 3] * 32
        # 对三个尺度的输出获取每个预测box坐标和box的分数，score计算为置信度x类别概率
        for i in range(len(yolo_outputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_outputs[i], self.anchors[anchor_mask[i]], len(self.class_names), input_shape, image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes, axis = 0)
        box_scores = tf.concat(box_scores, axis = 0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes, dtype = tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes, mask[:, c])
            class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold = self.nms_threshold)
            class_boxes = tf.gather(class_boxes, nms_index)
            class_box_scores = tf.gather(class_box_scores, nms_index)
            classes = tf.ones_like(class_box_scores, 'int32') * c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)
        boxes_ = tf.concat(boxes_, axis = 0)
        scores_ = tf.concat(scores_, axis = 0)
        classes_ = tf.concat(classes_, axis = 0)
        return boxes_, scores_, classes_


    def boxes_and_scores(self, feats, anchors, classes_num, input_shape, image_shape):
        """
        Introduction
        ------------
            将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
        Parameters
        ----------
            feats: yolo输出的feature map
            anchors: anchor的位置
            class_num: 类别数目
            input_shape: 输入大小
            image_shape: 图片大小
        Returns
        -------
            boxes: 物体框的位置
            boxes_scores: 物体框的分数，为置信度和类别概率的乘积
        """
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats, anchors, classes_num, input_shape)
        boxes = self.correct_boxes(box_xy, box_wh, input_shape, image_shape)
        boxes = tf.reshape(boxes, [-1, 4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores, [-1, classes_num])
        return boxes, box_scores


    def correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        """
        Introduction
        ------------
            计算物体框预测坐标在原图中的位置坐标
        Parameters
        ----------
            box_xy: 物体框左上角坐标
            box_wh: 物体框的宽高
            input_shape: 输入的大小
            image_shape: 图片的大小
        Returns
        -------
            boxes: 物体框的位置
        """
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = tf.cast(input_shape, dtype = tf.float32)
        image_shape = tf.cast(image_shape, dtype = tf.float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = tf.concat([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ], axis = -1)
        boxes *= tf.concat([image_shape, image_shape], axis = -1)
        return boxes


    def _get_feats(self, feats, anchors, num_classes, input_shape):
        """
        Introduction
        ------------
            根据yolo最后一层的输出确定bounding box
        Parameters
        ----------
            feats: yolo模型最后一层输出
            anchors: anchors的位置
            num_classes: 类别数量
            input_shape: 输入大小
        Returns
        -------
            box_xy, box_wh, box_confidence, box_class_probs
        """
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis = -1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.float32)
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        return box_xy, box_wh, box_confidence, box_class_probs


    def predict(self, inputs):
        """
        Introduction
        ------------
            构建预测模型
        Parameters
        ----------
            inputs: 处理之后的输入图片
            image_shape: 图像原始大小
        Returns
        -------
            boxes: 物体框坐标
            scores: 物体概率值
            classes: 物体类别
        """
        output = self.yolo_inference(inputs, config.num_anchors // 3, config.num_classes, training = False)
        boxes, scores, classes = self.eval(output, [416, 416], max_boxes = 20)
        return boxes, scores, classes


predictor = yolo()


def detect(image):
    resize_image = letterbox_image(image, (config.input_shape, config.input_shape))
    image_data = np.array(resize_image, dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data, axis=0)
    out_boxes, out_scores, out_classes = yolo.sess.run([yolo.boxes, yolo.scores, yolo.classes],feed_dict={yolo.input_image: image_data})
    return out_boxes, out_classes


if __name__ == '__main__':
    image_file = './dog.jpg'
    image = Image.open(image_file)
    detect(image)
