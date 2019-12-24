import tensorflow as tf

from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def DarknetConv2D(x, num_filters, kernel_size, strides=None):
    """
    定义darknet卷积层,使用模型batchnormalization和leakyrelu
    """
    if strides == (2, 2):
        x = Conv2D(num_filters, kernel_size, strides=strides, padding='valid', use_bias=False,
                   kernel_regularizer=l2(5e-4))(x)
    else:
        x = Conv2D(num_filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


def resblock_body(x, num_filters, num_blocks):
    """
    构建resblock模块
    """
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D(x, num_filters, (3, 3), strides=(2, 2))
    for _ in range(num_blocks):
        y = DarknetConv2D(x, num_filters // 2, (1, 1))
        y = DarknetConv2D(y, num_filters, (3, 3))
        x = Add()([x, y])
    return x


def darknet_body(x):
    # 定义darknet基础模型
    x = DarknetConv2D(x, 32, (3, 3))
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """
    定义darknet之后的连接
    """
    x = DarknetConv2D(x, num_filters, (1, 1))
    x = DarknetConv2D(x, num_filters * 2, (3, 3))
    x = DarknetConv2D(x, num_filters, (1, 1))
    x = DarknetConv2D(x, num_filters * 2, (3, 3))
    x = DarknetConv2D(x, num_filters, (1, 1))
    y = DarknetConv2D(x, num_filters * 2, (3, 3))
    y = Conv2D(out_filters, (1, 1), padding='same', kernel_regularizer=l2(5e-4))(y)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    # 构建darknet模型
    darknet = Model(inputs, darknet_body(inputs))

    # 获取第一个输出
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    # 获取第二个输出
    x = DarknetConv2D(x, 256, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[152].output])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))

    # 获取第三个输出
    x = DarknetConv2D(x, 128, (1, 1))
    x = UpSampling2D(2)(x)
    x = Concatenate()([x, darknet.layers[92].output])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    # 输出shape [13*13*3*(4+1+C),26*26*3*(4+1+C),52*52*3*(4+1+C)]
    # 13*13 为feature_map大小 3为每个cell预测的box数目,4+1+C为预测的box位置、前景背景、类别信息
    return Model(inputs, [y1, y2, y3])


def yolo_out(feats, anchors, num_classes, input_shape):
    """
    计算yolo的输出
    """

    # 设置anchor
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])
    grid_shape = K.shape(feats)[1:3]
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(
        feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])

    # 计算box位置信息 x,y使用simmod进行输出,  w,h为对数函数输出
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))

    # 计算box概率与box类别信息
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    计算yolo_box与score
    """

    def yolo_trans_boxes(box_xy, box_wh, input_shape, image_shape):
        # 获取到box的x,y,w,h信息
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        # 进行shape的转换
        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))
        new_shape = K.round(image_shape * K.min(input_shape / image_shape))

        # 获取box的实际位置信息
        offset = (input_shape - new_shape) / 2. / input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw *= scale

        # box转换
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)
        boxes = K.concatenate([
            box_mins[..., 0:1],
            box_mins[..., 1:2],
            box_maxes[..., 0:1],
            box_maxes[..., 1:2]
        ])
        boxes *= K.concatenate([image_shape, image_shape])
        return boxes

    # 获取yolo输出并进行box转换
    box_xy, box_wh, box_confidence, box_class_probs = yolo_out(feats,
                                                               anchors, num_classes, input_shape)
    boxes = yolo_trans_boxes(box_xy, box_wh, input_shape, image_shape)
    boxes = K.reshape(boxes, [-1, 4])

    # 计算box得分
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores


def yolo_test(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5):
    # 测试部分
    num_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    # 计算3层特征的检测结果
    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                    anchors[anchor_mask[l]], num_classes, input_shape, image_shape)

        boxes.append(_boxes)
        box_scores.append(_box_scores)
    # 融合所有的检测结果
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    # 进行筛选
    mask = box_scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    # 针对每一个类别进行score挑选
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=iou_threshold)
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = K.concatenate(boxes_, axis=0)
    scores_ = K.concatenate(scores_, axis=0)
    classes_ = K.concatenate(classes_, axis=0)
    return boxes_, scores_, classes_
