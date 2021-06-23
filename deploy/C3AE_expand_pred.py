#coding=utf-8
import tensorflow as tf
from tensorflow.keras.activations import softmax, sigmoid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Concatenate, Layer, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Lambda, multiply
from tensorflow.keras import regularizers
from tensorflow.keras.backend import l2_normalize
from tensorflow.keras.constraints import unit_norm


def model_refresh_without_nan(models):
    '''
        https://github.com/tensorflow/tensorflow/issues/38698
    '''
    import numpy as np
    valid_weights = []
    for l in models.get_weights():
        if np.isnan(l).any():
            print("!!!!!", l)
            valid_weights.append(np.nan_to_num(l))
        else:
            valid_weights.append(l)
    models.set_weights(valid_weights)


class MishActivation6(Layer):

    def __init__(self, **kwargs):
        super(MishActivation6, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MishActivation6, self).build(input_shape)

    def call(self, x):
        return tf.clip_by_value(x * tf.tanh(tf.math.log(1 + tf.exp(x))), -6., 6.)


Activation = MishActivation6  #ReLU

class GeM(Layer):

    def __init__(self, init_p=3., dynamic_p=False, **kwargs):
        # https://arxiv.org/pdf/1711.02512.pdf
        # add this behind relu
        if init_p <= 0:
             raise Exception("fatal p")
        super(GeM, self).__init__(**kwargs)
        self.init_p = init_p
        self.epsilon = 1e-8
        self.dynamic_p = dynamic_p

    def build(self, input_shape):
        super(GeM, self).build(input_shape)
        if self.dynamic_p:
            self.init_p = tf.Variable(self.init_p, dtype=tf.float32)

    def call(self, inputs):
        pool = tf.nn.avg_pool(tf.pow(tf.math.maximum(inputs, self.epsilon), self.init_p), inputs.shape[1:3], strides=(1, 1), padding="VALID")
        pool = tf.pow(pool, 1. / self.init_p)
        return pool

def BRA(input):
    bn = BatchNormalization()(input)
    activation = Activation()(bn)
    return AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(activation)

def BN_ReLU(input, name):
    return Activation()(BatchNormalization()(input))

def SE_BLOCK(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GeM(dynamic_p=True)(input)  # GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)
    scale = Dense(channel_nums, activation=sigmoid)(Activation()(fc1))
    return multiply([scale, input])

def SE_BLOCK1(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor / 2)(ga_pooling)

    gm_pooling = GlobalMaxPooling2D()(input)
    fc2 = Dense(channel_nums//r_factor / 2)(gm_pooling)

    fc = Concatenate()([fc1, fc2]) 

    scale = Dense(channel_nums, activation=sigmoid)(Activation()(fc))
    return multiply([scale, input])

def SE_BLOCK2(input, using_SE=True, r_factor=2):
    if not using_SE:
        return input
    channel_nums = input.get_shape()[-1]
    ga_pooling = GlobalAveragePooling2D()(input)
    fc1 = Dense(channel_nums//r_factor)(ga_pooling)

    gm_pooling = GlobalMaxPooling2D()(input)
    fc2 = Dense(channel_nums//r_factor)(gm_pooling)
   
    fc = Concatenate()([Activation()(fc1), Activation()(fc2)]) 

    scale = Dense(channel_nums, activation=sigmoid)(fc)
    return multiply([scale, input])


def SE_BLOCK_SAM(input, using_SE=True, r_factor=2):
    # SAM
    if not using_SE:
        return input

    a_pooling = AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(input)
    m_pooling = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same")(input)
    fc = Concatenate()([a_pooling, m_pooling]) 
    weight = Conv2D(1, (3, 3), padding="same", strides=1, use_bias=False, activation=sigmoid)(fc)
    return multiply([weight, input])
      

def SE_BLOCK_YOLO(input, using_SE=True, r_factor=2):
    # SAM yolo-v4
    if not using_SE:
        return input
    weight = Conv2D(1, (5, 5), padding="same", strides=1, use_bias=False, activation=sigmoid)(input)
    return multiply([weight, input])

def white_norm(input):
    return (input - tf.constant(127.5)) / 128.0

def build_shared_plain_network(height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    input_image = Input(shape=(height, width, channel))

    if using_white_norm:
        wn = Lambda(white_norm, name="white_norm")(input_image)
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(wn)  # output 62*62*32
    else:
        conv1 = Conv2D(32, (3, 3), padding="valid", strides=1, use_bias=False, name="conv1")(input_image)  # output 62*62*32
    block1 = BRA(conv1)
    block1 = SE_BLOCK(block1, using_SE)

    conv2 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv2")(block1)  # param 9248 = 32 * 32 * 3 * 3 + 32
    block2 = BRA(conv2)
    block2 = SE_BLOCK(block2, using_SE)  # put the se_net after BRA which achived better!!!!

    conv3 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv3")(block2)  # 9248
    block3 = BRA(conv3)
    block3 = SE_BLOCK(block3, using_SE)

    conv4 = Conv2D(32, (3, 3), padding="valid", strides=1, name="conv4")(block3)  # 9248
    block4 = BN_ReLU(conv4, name="BN_ReLu")  # 128
    block4 = SE_BLOCK(block4, using_SE)

    conv5 = Conv2D(32, (1, 1), padding="valid", strides=1, name="conv5")(block4)  # 1024 + 32
    conv5 = SE_BLOCK(conv5, using_SE)  # r=16效果不如conv5

    flat_conv = Flatten()(conv5)
    # cant find the detail how to change 4*4*32->12, you can try out all dims reduction
    # fc or pooling or any ohter operation
    #shape = map(int, conv5.get_shape()[1:])
    #shrinking_op = Lambda(lambda x: K.reshape(x, (-1, np.prod(shape))))(conv5)

    pmodel = Model(inputs=[input_image], outputs=[flat_conv])
    return pmodel

def build_net(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)

    cfeat = Concatenate(axis=-1)([y1, y2, y3])
    print("cates->", CATES, cfeat.shape)
    bulk_feat = Dense(CATES, use_bias=True, activity_regularizer=regularizers.l1(0.), activation=softmax, name="W1")(cfeat)
    age = Dense(1, name="age")(bulk_feat)
    gender = Dense(2, activation=softmax, activity_regularizer=regularizers.l2(0.), name="gender")(cfeat)

    #age = Lambda(lambda a: tf.reshape(tf.reduce_sum(a * tf.constant([[x * 10.0 for x in xrange(12)]]), axis=-1), shape=(-1, 1)), name="age")(bulk_feat)
    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat, gender])

def build_net3(CATES=12, height=64, width=64, channel=3, using_white_norm=True, using_SE=True):
    base_model = build_shared_plain_network(using_white_norm=using_white_norm, using_SE=using_SE)
    x1 = Input(shape=(height, width, channel))
    x2 = Input(shape=(height, width, channel))
    x3 = Input(shape=(height, width, channel))

    y1 = base_model(x1)
    y2 = base_model(x2)
    y3 = base_model(x3)
    cfeat = Concatenate(axis=-1)([l2_normalize(y1, axis=-1), l2_normalize(y2, axis=-1), l2_normalize(y3, axis=-1)])
    cfeat = BatchNormalization()(cfeat)
    cfeat = Dropout(0.5)(cfeat)
    cfeat = Dense(512, use_bias=False)(cfeat)
    cfeat = BatchNormalization()(cfeat)
    cfeat = l2_normalize(cfeat, axis=-1)
     
    # print("cates->", CATES, cfeat.shape)
    bulk_feat = Dense(CATES, use_bias=False, kernel_constraint=unit_norm(axis=0), activation=softmax, name="W1")(16 * cfeat)

    age = Dense(1, use_bias=False, name="age")(bulk_feat)

    gender = Dense(2, use_bias=False, kernel_constraint=unit_norm(axis=0), activation=softmax, name="gender")(cfeat)
    return Model(inputs=[x1, x2, x3], outputs=[age, bulk_feat, gender])
