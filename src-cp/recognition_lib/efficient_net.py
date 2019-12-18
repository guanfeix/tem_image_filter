import os
import json
import math
import string
import collections
import numpy as np
from keras.layers import *
import keras.backend as K
from six.moves import xrange
from keras.models import Model
from keras.utils.generic_utils import get_custom_objects


def swish(x):
    return x * K.sigmoid(x)
get_custom_objects().update({'swish': Activation(swish )})

def softmax(x):
    ndim = K.ndim(x)
    if ndim == 1:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
    elif ndim == 2:
        import tensorflow as tf
        return tf.nn.softmax(x)
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D. '
                         'Received input: %s' % x)
get_custom_objects().update({'mySoftmax': Activation(softmax)})

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio'
])

DEFAULT_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=24,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=2, input_filters=24, output_filters=40,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=40, output_filters=80,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=3, input_filters=80, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25),
    BlockArgs(kernel_size=5, num_repeat=4, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25),
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=192, output_filters=320,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25)
]

CONV_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling','config': {'scale': 2.0,'mode': 'fan_out',\
                            'distribution': 'normal'}}

DENSE_KERNEL_INITIALIZER = {'class_name': 'VarianceScaling','config': \
                            {'scale': 1. / 3.,'mode': 'fan_out','distribution': 'uniform'}}

def round_filters(filters, width_coefficient, depth_divisor):
    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)


def round_repeats(repeats, depth_coefficient):
    return int(math.ceil(depth_coefficient * repeats))

def mb_conv_block(inputs, block_args, activation, drop_rate=None, prefix='', ):

    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = Conv2D(filters, 1,padding='same',use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'expand_conv')(inputs)
        x = BatchNormalization(axis=3, name=prefix + 'expand_bn')(x)
        x = Activation(activation, name=prefix + 'expand_activation')(x)
    else:
        x = inputs

    # Depthwise Convolution
    x = DepthwiseConv2D(block_args.kernel_size,strides=block_args.strides,padding='same',
                               use_bias=False,depthwise_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'dwconv')(x)
    x = BatchNormalization(axis=3, name=prefix + 'bn')(x)
    x = Activation(activation, name=prefix + 'activation')(x)

    # Squeeze and Excitation phase
    if has_se:
        num_reduced_filters = max(1, int(block_args.input_filters * block_args.se_ratio))
        se_tensor = GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters)
        se_tensor = Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = Conv2D(num_reduced_filters, 1,activation=activation,padding='same',use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'se_reduce')(se_tensor)
        se_tensor = Conv2D(filters, 1,activation='sigmoid',padding='same',use_bias=True,
                                    kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'se_expand')(se_tensor)
        x = multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = Conv2D(block_args.output_filters, 1,padding='same',use_bias=False,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,name=prefix + 'project_conv')(x)
    x = BatchNormalization(axis=3, name=prefix + 'project_bn')(x)
    if block_args.id_skip and all(s == 1 for s in block_args.strides) and \
            block_args.input_filters == block_args.output_filters:
        if drop_rate and (drop_rate > 0):
            x = Dropout(drop_rate,noise_shape=(None, 1, 1, 1),name=prefix + 'drop')(x)
        x = add([x, inputs], name=prefix + 'add')
    return x


def EfficientNet(width_coefficient,depth_coefficient,default_resolution,dropout_rate=0.2,drop_connect_rate=0.2,
                 depth_divisor=8,blocks_args=DEFAULT_BLOCKS_ARGS,model_name='efficientnet',include_top=True,
                 weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000,**kwargs):

    # Build stem
    x = input_tensor
#     print('x is :{}'.format(x))
    x = Conv2D(round_filters(32, width_coefficient, depth_divisor), 3,strides=(2, 2),
                      padding='same',use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name='stem_conv')(x)
    x = BatchNormalization(axis=3, name='stem_bn')(x)
    x = Activation('swish', name='stem_activation')(x)

    # Build blocks
    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0
    for idx, block_args in enumerate(blocks_args):
        assert block_args.num_repeat > 0
        # Update block input and output filters based on depth multiplier.
        block_args = block_args._replace(
            input_filters=round_filters(block_args.input_filters,
                                        width_coefficient, depth_divisor),
            output_filters=round_filters(block_args.output_filters,
                                         width_coefficient, depth_divisor),
            num_repeat=round_repeats(block_args.num_repeat, depth_coefficient))

        # The first block needs to take care of stride and filter size increase.
        drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
        x = mb_conv_block(x, block_args,activation='swish',drop_rate=drop_rate,prefix='block{}a_'.format(idx + 1))
        block_num += 1
        if block_args.num_repeat > 1:
            # pylint: disable=protected-access
            block_args = block_args._replace(
                input_filters=block_args.output_filters, strides=[1, 1])
            # pylint: enable=protected-access
            for bidx in xrange(block_args.num_repeat - 1):
                drop_rate = drop_connect_rate * float(block_num) / num_blocks_total
                block_prefix = 'block{}{}_'.format(idx + 1,string.ascii_lowercase[bidx + 1])
                x = mb_conv_block(x, block_args,activation='swish',drop_rate=drop_rate,prefix=block_prefix)
                block_num += 1
    x = Conv2D(round_filters(1280, width_coefficient, depth_divisor), 1,
                      padding='same',use_bias=False,kernel_initializer=CONV_KERNEL_INITIALIZER,name='top_conv')(x)
    x = BatchNormalization(axis=3, name='top_bn')(x)
    x = Activation('swish', name='top_activation')(x)

    if pooling == 'avg':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D(name='max_pool')(x)
    if include_top:
        if dropout_rate and dropout_rate > 0:
            x = Dropout(dropout_rate, name='top_dropout')(x)
        x = Dense(classes,activation='mySoftmax',kernel_initializer=DENSE_KERNEL_INITIALIZER,name='probs')(x)
    model = Model(input_tensor, x, name=model_name)
    return model


def EfficientNetB3(include_top=True,weights='imagenet',input_tensor=None,input_shape=None,pooling=None,classes=1000,**kwargs):
    return EfficientNet(1.2, 1.4, 300, 0.3,model_name='efficientnet-b3',include_top=include_top, weights=weights,\
                        input_tensor=input_tensor, input_shape=input_shape,pooling=pooling, classes=classes,**kwargs)


def EfficientNetB4(include_top=True,weights='imagenet',input_tensor=None,input_shape=None,pooling=None,\
                            classes=1000,**kwargs):
    return EfficientNet(1.4, 1.8, 380, 0.4,model_name='efficientnet-b4',include_top=include_top, weights=weights,\
                        input_tensor=input_tensor, input_shape=input_shape,pooling=pooling, classes=classes,**kwargs)


"""
# 模型地址:
#       B3:model_data/efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment.h5
#       B4:model_data/efficientnet-b4_weights_tf_dim_ordering_tf_kernels_autoaugment.h5
# 模型使用同正常的xception即可,只需要自己手动进行weights的设置,需要升级到keras==2.24
input_tensor = Input(shape=(300,300,3))
base_model = EfficientNetB3(input_tensor=input_tensor,weights=None,pooling='avg')
base_model.load_weights('efficientnet-b3_weights_tf_dim_ordering_tf_kernels_autoaugment.h5')

x = base_model.layers[-3].output
x = BatchNormalization()(x,training=False)
x = Dropout(0.3)(x)
classify1 = Dense(6, activation='mySoftmax',use_bias=False, name='classify1')(x)
classify2 = Dense(9, activation='mySoftmax',use_bias=False, name='classify2')(x)
classify3 = Dense(30, activation='mySoftmax',use_bias=False, name='classify3')(x)
model = Model(input_tensor, [classify1,classify2,classify3])
model.summary()
"""
