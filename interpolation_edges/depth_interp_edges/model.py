from utils.ops import *
from tensorflow.contrib import layers as l


def network(t_input, t_input_edges, is_train=True):

    base_filters = 32
    p = 0.5
    net = {}
    net['gate'] = t_input
    net['guidance_gate'] = t_input_edges
    use_bn = False
    d_flg = is_train


    net['gate'] = t_input
    with tf.device('/gpu:0'):
        with tf.variable_scope('G_Depth'):
            net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=5, stride=1, act=tf.nn.elu,
                                  bn=use_bn)
            # net['conv1'] = dilated2d(net['gate'], base_filters, 'convolution_1', kernel_sz=7, dilation=2, act=tf.nn.elu, bn=use_bn)
            net['pool'] = l.avg_pool2d(net['conv1'], 3, 2, padding='SAME')

            # /2
            net['conv2'] = projection_block(net['pool'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu,
                                            bn=use_bn)

            # net['conv3'] = downscale_block(net['conv2'], base_filters*2, 'convolution_3', kernel_sz=3,   act=tf.nn.elu, bn=use_bn)
            net['conv3'] = downscale_pool(net['conv2'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu,
                                          bn=use_bn)
            net['drop2'] = drop(net['conv3'], p, is_training=d_flg)

            # /4
            net['conv4'] = projection_block(net['drop2'], base_filters * 4, 'convolution_4', kernel_sz=3, act=tf.nn.elu,
                                            bn=use_bn)
            # net['conv5'] = downscale_block(net['conv4'], base_filters*4, 'convolution_5', kernel_sz=3, act=tf.nn.elu, bn=use_bn)
            net['conv5'] = downscale_pool(net['conv4'], base_filters * 4, 'convolution_5', kernel_sz=3, act=tf.nn.elu,
                                          bn=use_bn)
            net['drop3'] = drop(net['conv5'], p, is_training=d_flg)
            # /8
            net['conv6'] = projection_block(net['drop3'], base_filters * 8, 'convolution_6', kernel_sz=3, act=tf.nn.elu,
                                            bn=use_bn)
            net['conv7'] = residual_block(net['conv6'], base_filters * 8, 'convolution_7', kernel_sz=3, act=tf.nn.elu,
                                          bn=use_bn)

            net['drop4'] = drop(net['conv7'], p, is_training=d_flg)

            # net['uconv1'] = deconv2d(net['drop4'], base_filters*4, 'upconv_1',  stride=2, kernel_sz=3,act=tf.nn.elu, bn=False)
            net['convup1'] = conv2d(net['drop4'], base_filters * 8, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu,
                                    bn=False)
            net['uconv1'] = tf.depth_to_space(net['convup1'], 2)
            net['skip1'] = tf.concat([net['conv4'], net['uconv1']], 3)

            # net['uconv2'] = deconv2d(net['skip1'], base_filters*2,'upconv_2',  stride=2, kernel_sz=3,act=tf.nn.elu, bn=False)
            net['convup2'] = conv2d(net['skip1'], base_filters * 8, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu,
                                    bn=False)
            net['uconv2'] = tf.depth_to_space(net['convup2'], 2)
            net['skip2'] = tf.concat([net['conv2'], net['uconv2']], 3)

            # net['uconv3'] = deconv2d(net['skip2'], base_filters,'upconv_3',  stride=2, kernel_sz=3, act=tf.nn.elu, bn=False)
            net['convup3'] = conv2d(net['skip2'], base_filters * 4, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu,
                                    bn=False)
            net['uconv3'] = tf.depth_to_space(net['convup3'], 2)
            net['skip3'] = tf.concat([net['conv1'], net['uconv3']], 3)

            # net['uconv4'] = conv2d(net['skip3'], base_filters,'upconv_4',  stride=1, kernel_sz=3, act=tf.nn.elu, bn=False)
            net['uconv4'] = conv2d(net['skip3'], base_filters, 'upconv_4', stride=1, kernel_sz=3, act=tf.nn.elu,
                                   bn=False)

            net['residuals'] = net['uconv4']
            net['conv_out'] = conv2d(net['residuals'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False)
        G_output = net['conv_out'] + net['gate']

    return G_output


def network2(t_input, t_input_edges, is_train=True):
    """ Generator
    """
    base_filters = 32
    #p = 0.5
    net = {}
    net['gate'] = t_input
    net['guidance_gate'] = t_input_edges


    return conv2d(t_input, 1, 'conv_1', stride=1, kernel_sz=3, act=tf.nn.elu)

def network3(t_input, t_input_edges, is_train=True):
    """ Generator
    """
    base_filters = 32
    p = 0.5
    p_g = 0.5
    net = {}
    net['gate'] = t_input
    net['guidance_gate'] = t_input_edges

    with tf.variable_scope('G_Depth'):

        with tf.variable_scope('Guide_Branch'):
            net['g_conv1'] = conv2d(net['guidance_gate'], base_filters, 'g_conv_1', stride=1, kernel_sz=5, act=tf.nn.elu)
            net['g_conv2'] = conv2d(net['g_conv1'], base_filters, 'g_conv_2', stride=1, kernel_sz=3, act=tf.nn.elu)
            net['g_conv3'] = conv2d(net['g_conv2'], base_filters, 'g_conv_3', stride=1, kernel_sz=3, act=tf.nn.elu)
            net['g_down1'] = l.avg_pool2d(net['g_conv2'], 3, 2, padding='SAME')
            net['g_drop1'] = drop(net['g_down1'],p_g, is_training=is_train)

            net['g_conv4'] = conv2d(net['g_drop1'], base_filters, 'g_conv_4', stride=1, kernel_sz=5, act=tf.nn.elu)
            net['g_conv5'] = conv2d(net['g_conv4'], base_filters, 'g_conv_5', stride=1, kernel_sz=3, act=tf.nn.elu)
            net['g_down2'] = l.avg_pool2d(net['g_conv5'], 3, 2, padding='SAME')
            net['g_drop2'] = drop(net['g_down2'],p_g, is_training=is_train)

            net['g_conv6'] = conv2d(net['g_drop2'], base_filters, 'g_conv_6', stride=1, kernel_sz=3, act=tf.nn.elu)




        with tf.variable_scope('Depth_Branch'):
            #print('gate', net['gate'].shape)
            net['conv1'] = conv2d(net['gate'], base_filters, 'conv_1', stride=1, kernel_sz=3, act=tf.nn.elu)
            net['skip1_g'] = tf.concat([net['conv1'], net['g_conv2']], 3)
            net['down1'] = l.avg_pool2d(net['skip1_g'], 3, 2, scope= 'down_1', padding= 'SAME')
            net['drop1'] = drop(net['down1'],p, is_training=is_train)
            #print('down1', net['down1'].shape)

            net['conv2'] = projection_block(net['drop1'], base_filters*2, 'conv_2', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
            net['skip2_g'] = tf.concat([net['conv2'], net['g_conv4']], 3)
            #net['down2'] = l.avg_pool2d(net['skip2_g'], 3, 2, scope= 'down_2', padding= 'SAME')
            net['down2'] = downscale_pool(net['skip2_g'], base_filters*2, 'down_2', kernel_sz=3,   act=tf.nn.elu, bn = True, is_training=is_train)
            net['drop2'] = drop(net['down2'],p, is_training=is_train)
            #print('down2',net['down2'].shape)

            net['conv3'] =  projection_block(net['drop2'], base_filters*4, 'conv_3', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
            net['skip3_g'] = tf.concat([net['conv3'], net['g_conv6']], 3)
            #net['down3'] = l.avg_pool2d(net['skip3_g'], 3, 2, scope= 'down_3', padding= 'SAME')
            net['down3'] = downscale_pool(net['skip3_g'], base_filters*4, 'down_3', kernel_sz=3,   act=tf.nn.elu, bn = True, is_training=is_train)
            net['drop3'] = drop(net['down3'],p, is_training=is_train)
            #print('down3',net['down3'].shape)

            net['conv4a'] =  projection_block(net['down3'], base_filters*8, 'conv_4a', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
            net['conv4'] = residual_block(net['conv4a'], base_filters *8, 'conv_4', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)
            net['conv5'] = residual_block(net['conv4'], base_filters*8, 'conv_5', kernel_sz=3, act=tf.nn.elu, bn = True, is_training=is_train)

            net['conv6'] = conv2d(net['conv5'], base_filters*16, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu)
            #net['uconv1'] = pixelShuffler(net['conv7'], scale=2)
            net['uconv1'] = tf.depth_to_space(net['conv6'], 2)
            net['skip1'] = tf.concat([net['conv3'], net['uconv1']], 3)

            net['conv7'] = conv2d(net['skip1'], base_filters*8, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu)
            #net['uconv2'] = pixelShuffler(net['conv8'], scale=2)
            net['uconv2'] = tf.depth_to_space(net['conv7'], 2)
            net['skip2'] = tf.concat([net['conv2'], net['uconv2']], 3)

            net['conv8'] = conv2d(net['skip2'], base_filters*4, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu)
            #net['uconv3'] = pixelShuffler(net['conv9'], scale=2)
            net['uconv3'] = tf.depth_to_space(net['conv8'], 2)
            net['skip3'] = tf.concat([net['conv1'], net['uconv3']], 3)

            net['conv_closing'] = conv2d(net['skip3'], base_filters, 'conv_close', stride=1, kernel_sz=3, act=tf.nn.elu)
            net['conv_out'] = conv2d(net['conv_closing'], 1, 'conv_out', kernel_sz=3, act=None, bn=False)

        G_output = net['conv_out'] + t_input
    #G_output.set_shape(t_interpolated.shape)

    return G_output