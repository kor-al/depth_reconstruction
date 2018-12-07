#! /usr/bin/python
# -*- coding: utf8 -*-

import tensorflow as tf
import tensorlayer as tl

import time
from ops import *

# from tensorflow.python.ops import variable_scope as vs
# from tensorflow.python.ops import math_ops, init_ops, array_ops, nn
# from tensorflow.python.util import nest
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell

# https://github.com/david-gpu/srez/blob/master/srez_model.py


def SRGAN_g(d_input,t_interpolated, is_train=True, reuse=False):
    """ Generator
    """
    base_filters = 32
    p = 0.5
    net = {}
    net['gate'] = d_input

    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    
    '''
    with tf.variable_scope('G_Depth_SR', reuse=reuse):
        net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=3, stride=1, act=tf.nn.elu)
        net['conv2'] = projection_block(net['conv1'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu)
        net['drop1'] = drop(net['conv2'], p, is_training=is_train)
        net['conv3'] = residual_block(net['drop1'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu)
        net['conv4'] = residual_block(net['conv3'], base_filters * 2, 'convolution_4', kernel_sz=3, act=tf.nn.elu)
        net['drop2'] = drop(net['conv4'], p, is_training=is_train)
        net['conv5'] = residual_block(net['drop2'], base_filters * 2, 'convolution_5', kernel_sz=3, act=tf.nn.elu)
        net['conv6'] = residual_block(net['conv5'], base_filters * 2, 'convolution_6', kernel_sz=3, act=tf.nn.elu)
        net['drop3'] = drop(net['conv6'], p, is_training=is_train)
        net['conv7'] = conv2d(net['drop3'], base_filters * 4, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv1'] = tf.depth_to_space(net['conv7'], 2)
        net['conv8'] = conv2d(net['uconv1'], base_filters * 2, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv2'] = tf.depth_to_space(net['conv8'], 2)
        net['conv9'] = deconv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv3'] = tf.depth_to_space(net['conv9'], 2)
        net['conv_out'] = conv2d(net['uconv3'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False)
    G_output = net['conv_out'] + t_interpolated

    '''


    with tf.variable_scope('G_Depth_SR', reuse=reuse):
        net['conv0'] = conv2d(net['gate'], base_filters * 2, 'convolution_0', kernel_sz=3, stride=1, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)

        net['conv1'] = residual_block(net['conv0'], base_filters * 2, 'convolution_1', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv2'] = residual_block(net['conv1'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop1'] = drop(net['conv2'], p, is_training=is_train)

        net['conv3'] = residual_block(net['drop1'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv4'] = residual_block(net['conv3'], base_filters * 2, 'convolution_4', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop2'] = drop(net['conv4'], p, is_training=is_train)

        net['conv5'] = residual_block(net['drop2'], base_filters * 2, 'convolution_5', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv6'] = residual_block(net['conv5'], base_filters * 2, 'convolution_6', kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop3'] = drop(net['conv6'], p, is_training=is_train)

        net['conv7'] = conv2d(net['drop3'], base_filters * 2, 'convolution_end', stride=1, kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train, w_init=w_init, g_init=g_init)

        net['add'] =  net['conv0'] + net['conv7']

        net['conv8'] = conv2d(net['add'], base_filters * 4, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu, w_init=w_init)
        #net['uconv1'] = pixelShuffler(net['conv8'], scale=2)
        net['uconv1'] = tf.depth_to_space(net['conv8'], 2)

        net['conv9'] = conv2d(net['uconv1'], base_filters * 2, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu,w_init=w_init)
        #net['uconv2'] = pixelShuffler(net['conv9'], scale=2)
        net['uconv2'] = tf.depth_to_space(net['conv9'], 2)

        net['conv10'] = conv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu, w_init=w_init)
        #net['conv10'] = deconv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu, bn=True)
        #net['uconv3'] = pixelShuffler(net['conv10'], scale=2)
        net['uconv3'] = tf.depth_to_space(net['conv10'], 2)

        net['conv_out'] = conv2d(net['uconv3'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False, w_init=w_init)

        G_output = net['conv_out'] + t_interpolated
    #G_output.set_shape(t_interpolated.shape)


    print('>>>>>>>>>>>>>>>>>>>>>',G_output.shape)
    return tf.nn.tanh(G_output)



    """
    with tf.variable_scope('G_Depth_SR', reuse=reuse):
        net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=3, act=tf.nn.elu, bn=True,
                              is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv2'] = projection_block(net['conv1'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu,
                                        bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop1'] = drop(net['conv2'], p, is_training=is_train)

        net['conv3'] = residual_block(net['drop1'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu,
                                      bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv4'] = residual_block(net['conv3'], base_filters * 2, 'convolution_4', kernel_sz=3, act=tf.nn.elu,
                                      bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop2'] = drop(net['conv4'], p, is_training=is_train)

        net['conv5'] = residual_block(net['drop2'], base_filters * 2, 'convolution_5', kernel_sz=3, act=tf.nn.elu,
                                      bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        net['conv6'] = residual_block(net['conv5'], base_filters * 2, 'convolution_6', kernel_sz=3, act=tf.nn.elu,
                                      bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        net['drop3'] = drop(net['conv6'], p, is_training=is_train)

        net['conv7'] = conv2d(net['drop3'], base_filters * 4, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu, bn=True,
                              is_training=is_train, w_init=w_init, g_init=g_init)
        # net['uconv1'] = pixelShuffler(net['conv7'], scale=2)
        net['uconv1'] = tf.depth_to_space(net['conv7'], 2)

        net['conv8'] = conv2d(net['uconv1'], base_filters * 2, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu,
                              bn=True, is_training=is_train, w_init=w_init, g_init=g_init)
        # net['uconv2'] = pixelShuffler(net['conv8'], scale=2)
        net['uconv2'] = tf.depth_to_space(net['conv8'], 2)

        net['conv9'] = deconv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu, bn=True)
        # conv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train, w_init=w_init, g_init=g_init)
        # net['uconv3'] = pixelShuffler(net['conv9'], scale=2)
        net['uconv3'] = tf.depth_to_space(net['conv9'], 2)

        net['conv_out'] = conv2d(net['uconv3'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False, w_init=w_init)

        print(net['conv_out'].shape)

        G_output = net['conv_out'] + t_interpolated

    return tf.nn.tanh(G_output)
    """



    """

    with tf.variable_scope("G_Depth_SR", reuse=reuse) as vs:
        net['conv1'] = residual_block(net['gate'], base_filters * 2, 'convolution_1', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        net['conv2'] = residual_block(net['conv1'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        #net['drop1'] = drop(net['conv2'], p, is_training=is_train)
        net['drop1'] = net['conv2']
        net['conv3'] = residual_block(net['drop1'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        net['conv4'] = residual_block(net['conv3'], base_filters * 2, 'convolution_4', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        #net['drop2'] = drop(net['conv4'], p, is_training=is_train)
        net['drop2'] = net['conv4']
        net['conv5'] = residual_block(net['drop2'], base_filters * 2, 'convolution_5', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        net['conv6'] = residual_block(net['conv5'], base_filters * 2, 'convolution_6', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
        #net['drop3'] = drop(net['conv6'], p, is_training=is_train)
        net['drop3'] = net['conv6']
        net['conv7'] = conv2d(net['drop3'], base_filters * 4, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train)
        net['uconv1'] = tf.depth_to_space(net['conv7'], 2)
        net['conv8'] = conv2d(net['uconv1'], base_filters * 4, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train)
        net['uconv2'] = tf.depth_to_space(net['conv8'], 2)
        net['conv9'] = conv2d(net['uconv2'], base_filters * 4, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu, bn = False, is_training=is_train)
        net['uconv3'] = tf.depth_to_space(net['conv9'], 2)
        net['conv10'] = conv2d(net['uconv3'], 1, 'upconv_4', kernel_sz=3, act=None, bn=False, is_training=is_train)
        net['add'] = net['conv10'] + t_interpolated
        G_output = net['add']
    return tf.nn.tanh(G_output)
    """

def SRGAN_d(input_images, is_train=True, reuse=False):
    df_dim = 64
    lrelu = lambda x: tl.act.lrelu(x, 0.2)
    p = 0.4
    use_bn = False

    with tf.variable_scope("SRGAN_d", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        net = {}
        net['gate'] = input_images
        print('dicr>>>>>>>>>>>>>>>',input_images.shape)

        net['conv1'] = conv2d(net['gate'], df_dim,kernel_sz = 5, stride = 2, act=lrelu, name='convolution_1', bn = use_bn, is_training=is_train)
        net['drop1'] = drop(net['conv1'], p, is_training=is_train)
        net['conv2'] = conv2d(net['drop1'], df_dim * 2,kernel_sz = 5, stride = 1, act=lrelu, name='convolution_2', bn = use_bn, is_training=is_train)
        net['drop2'] = drop(net['conv2'], p, is_training=is_train)
        net['conv3'] = conv2d(net['drop2'], df_dim * 4,kernel_sz = 3, stride = 2, act=lrelu, name='convolution_3',bn = use_bn, is_training=is_train)
        net['drop3'] = drop(net['conv3'], p, is_training=is_train)
        net['conv4'] = conv2d(net['drop3'], df_dim * 8,kernel_sz = 3, stride = 2, act=lrelu, name='convolution_4', bn = use_bn, is_training=is_train)
        net['drop4'] = drop(net['conv4'], p, is_training=is_train)
        net['conv5'] = conv2d(net['drop4'], df_dim * 2,kernel_sz = 3, stride = 1, act=lrelu, name='convolution_5', bn = use_bn, is_training=is_train)

        net['flatten'] = l.flatten(net['conv5'],scope = 'flatten')
        net['out'] = l.fully_connected(net['flatten'], 1, activation_fn=None, scope='dense')

        logits = net['out']

    return  tf.nn.sigmoid(net['out']), logits


def Vgg19_simple_api(rgb, reuse):
    """
    Build the VGG 19 Model

    Parameters
    -----------
    rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
    """
    VGG_MEAN = [103.939, 116.779, 123.68]
    with tf.variable_scope("VGG19", reuse=reuse) as vs:
        start_time = time.time()
        print("build model started")
        rgb_scaled = rgb * 255.0
        # Convert RGB to BGR
        if tf.__version__ <= '0.11':
            red, green, blue = tf.split(3, 3, rgb_scaled)
        else:  # TF 1.0
            # print(rgb_scaled)
            red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        if tf.__version__ <= '0.11':
            bgr = tf.concat(3, [
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ])
        else:
            bgr = tf.concat(
                [
                    blue - VGG_MEAN[0],
                    green - VGG_MEAN[1],
                    red - VGG_MEAN[2],
                ], axis=3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
        """ input layer """
        net_in = tl.layers.InputLayer(bgr, name='input')
        """ conv1 """
        network = tl.layers.Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
        network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
        """ conv2 """
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
        network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
        """ conv3 """
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
        network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
        """ conv4 """
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
        conv = network
        """ conv5 """
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_3')
        network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
        network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
        """ fc 6~8 """
        network = tl.layers.FlattenLayer(network, name='flatten')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
        network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
        network = tl.layers.DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')
        print("build model finished: %fs" % (time.time() - start_time))
        return network, conv


# def vgg16_cnn_emb(t_image, reuse=False):
#     """ t_image = 244x244 [0~255] """
#     with tf.variable_scope("vgg16_cnn", reuse=reuse) as vs:
#         tl.layers.set_name_reuse(reuse)
#
#         mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
#         net_in = InputLayer(t_image - mean, name='vgg_input_im')
#         """ conv1 """
#         network = tl.layers.Conv2dLayer(net_in,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 3, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 64],  # 64 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv1_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool1')
#         """ conv2 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv2_2')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool2')
#         """ conv3 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv3_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool3')
#         """ conv4 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv4_3')
#
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool4')
#         conv4 = network
#
#         """ conv5 """
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_1')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_2')
#         network = tl.layers.Conv2dLayer(network,
#                         act = tf.nn.relu,
#                         shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
#                         strides = [1, 1, 1, 1],
#                         padding='SAME',
#                         name ='vgg_conv5_3')
#         network = tl.layers.PoolLayer(network,
#                         ksize=[1, 2, 2, 1],
#                         strides=[1, 2, 2, 1],
#                         padding='SAME',
#                         pool = tf.nn.max_pool,
#                         name ='vgg_pool5')
#
#         network = FlattenLayer(network, name='vgg_flatten')
#
#         # # network = DropoutLayer(network, keep=0.6, is_fix=True, is_train=is_train, name='vgg_out/drop1')
#         # new_network = tl.layers.DenseLayer(network, n_units=4096,
#         #                     act = tf.nn.relu,
#         #                     name = 'vgg_out/dense')
#         #
#         # # new_network = DropoutLayer(new_network, keep=0.8, is_fix=True, is_train=is_train, name='vgg_out/drop2')
#         # new_network = DenseLayer(new_network, z_dim, #num_lstm_units,
#         #             b_init=None, name='vgg_out/out')
#         return conv4, network
