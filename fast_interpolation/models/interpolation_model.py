import tensorflow as tf
from tensorflow.contrib import layers as l

def get_non_zero_mean(masked_depth_tensor):
    non_zero_sum = tf.reduce_sum(masked_depth_tensor, axis=(1,2), keep_dims=True)
    non_zero_elements = tf.cast(tf.count_nonzero(masked_depth_tensor, axis=(1, 2), keep_dims=True), tf.float32)
    non_zero_mean = non_zero_sum / non_zero_elements
    return non_zero_mean

def preprocess_sparse_depth(d_tensor, size):
    #eps = tf.constant(1e-2)
    #maxi = tf.constant(1e4)

    #d_cl = tf.clip_by_value(d_tensor, eps, maxi)
    d_log = tf.log(d_tensor + 1.)
    d_res = tf.image.resize_images(d_log, size)
    d_norm = d_res
    nonzero = tf.cast(tf.count_nonzero(d_norm, axis=(1, 2), keep_dims=True), tf.float32)
    masked_sum = tf.reduce_sum(d_norm, axis=(1, 2), keep_dims=True)
    mean_d = masked_sum / nonzero

    d_masked_out = (d_norm - mean_d)

    return d_masked_out, mean_d



def preprocess_depth(d_tensor, mask_tensor, size):
    eps = tf.constant(1e-2)
    maxi = tf.constant(1e4)




    d_cl = tf.clip_by_value(d_tensor, eps, maxi)
    d_log = tf.log(d_cl + 1.)
    d_res = tf.image.resize_images(d_log, size)
    d_norm = d_res
    nonzero = tf.cast(tf.count_nonzero(d_norm * mask_tensor[:, :], axis=(1, 2), keep_dims=True), tf.float32)

    masked_sum = tf.reduce_sum(d_norm * mask_tensor[:, :], axis=(1, 2), keep_dims=True)
    mean_d = masked_sum / nonzero
    ground_truth = d_norm

    mean, std = tf.nn.moments(ground_truth * mask_tensor[:, :], axes=(1, 2), keep_dims=True)
    d_masked_out = (ground_truth * mask_tensor[:, :] - mean_d)

    """

    ################################################

    d_cl = tf.clip_by_value(d_tensor, eps, maxi)
    d_log = tf.log(d_cl + 1.)
    ground_truth = tf.image.resize_images(d_log, size)
    d_masked = ground_truth * mask_tensor
    #normalize masked depth
    masked_max = tf.reduce_max(d_masked, axis=(1, 2), keep_dims=True)
    dummy = tf.where(tf.equal(d_masked, 0), tf.ones_like(d_masked) * masked_max, d_masked)
    masked_min = tf.reduce_min(dummy, axis=(1, 2), keep_dims=True)
    masked_scale = tf.where(tf.equal(masked_max, masked_min), masked_max, masked_max - masked_min)
    d_masked = (d_masked - masked_min) / masked_scale
    d_masked = tf.clip_by_value(d_masked, 0, masked_max)
    mean = get_non_zero_mean(d_masked)


    """
    """

    # normalize
    d_log = tf.log(d_tensor + 1)
    # d_log = tf.check_numerics(d_log, message='Troubles after log', name='log')
    #tf.check_numerics(d_tensor, message='NaN in input 1', name='inputcheck')
    d_tensor = tf.image.resize_images(d_log, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    #tf.check_numerics(d_tensor, message='NaN in input', name='inputcheck')
    d_max = tf.reduce_max(d_tensor, axis=(1, 2), keep_dims=True)
    # max depth to unknown values
    d_normed = tf.where(tf.equal(d_tensor, 0), tf.ones_like(d_tensor) * d_max, d_tensor)
    d_min = tf.reduce_min(d_tensor, axis=(1, 2), keep_dims=True)
    scale = tf.where(tf.equal(d_max, d_min), d_max, d_max - d_min)
    d_normed = (d_normed - d_min) / scale
    d_normed = tf.check_numerics(d_normed, message='Troubles after norm', name='normal')

    # normalize masked tensor separately
    d_masked = d_tensor * mask_tensor
    masked_max = tf.reduce_max(d_masked, axis=(1, 2), keep_dims=True)
    dummy = tf.where(tf.equal(d_masked, 0), tf.ones_like(d_masked) * masked_max, d_masked)
    masked_min = tf.reduce_min(dummy, axis=(1, 2), keep_dims=True)
    masked_scale = tf.where(tf.equal(masked_max, masked_min), masked_max, masked_max - masked_min)
    d_masked = (d_masked - masked_min) / masked_scale
    d_masked = tf.clip_by_value(d_masked, 0, masked_max)
    mean = get_non_zero_mean(d_masked)
     return  d_normed, d_masked, mean
    """

    return ground_truth, d_masked_out, mean_d




def preprocess_rgb(rgb_tensor, size):
    rgb_res = tf.image.resize_images(rgb_tensor, size)
    greyscale = rgb_res[:, :, :, 0] * 0.2989 + rgb_res[:, :, :, 1] * 0.5870 + rgb_res[:, :, :, 2] * 0.1140
    greyscale = greyscale[:, :, :, None]
    g_min = tf.reduce_min(greyscale, axis=(1, 2), keep_dims=True)
    g_max = tf.reduce_max(greyscale, axis=(1, 2), keep_dims=True)
    g_normed = (greyscale - g_min) / (g_max - g_min)
    g_out = g_normed - tf.reduce_mean(g_normed, axis=(1, 2), keep_dims=True)

    return g_out

def get_placeholders():
    input_rgb = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    target = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    mask_t = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    d_flg = tf.placeholder(tf.bool)
    return input_rgb, target,mask_t, d_flg


def drop(x, p, is_training=True):
    noise_sh = [1, 1, 1, x.get_shape().as_list()[3]]
    return l.dropout(x, p, is_training=is_training, noise_shape=noise_sh)


def conv2d(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
    return out


def projection_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        proj = l.conv2d(in_tensor,
                        num_outputs=out_dim,
                        kernel_size=1,
                        stride=stride,
                        normalizer_fn=norm,
                        activation_fn=None,
                        padding='SAME',
                        normalizer_params={'activation_fn': None})
    return out + proj


def downscale_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=2, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=1,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        proj = l.conv2d(in_tensor,
                        num_outputs=out_dim,
                        kernel_size=2,
                        stride=stride,
                        normalizer_fn=norm,
                        activation_fn=None,
                        padding='SAME',
                        normalizer_params={'activation_fn': None})
    return out + proj


def downscale_pool(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=2, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=1,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})

        down = l.avg_pool2d(in_tensor, 3, 2, padding='SAME')
        proj = l.conv2d(down,
                        num_outputs=out_dim,
                        kernel_size=1,
                        stride=1,
                        normalizer_fn=norm,
                        activation_fn=None,
                        padding='SAME',
                        normalizer_params={'activation_fn': None})
    return out + proj


def residual_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
    return out + in_tensor


def deconv2d(in_tensor, out_dim, name='transposed_conv', kernel_sz=3, act=None, stride=1, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d_transpose(in_tensor,
                                 num_outputs=out_dim,
                                 kernel_size=kernel_sz,
                                 stride=stride,
                                 normalizer_fn=norm,
                                 activation_fn=l_act,
                                 padding='SAME',
                                 normalizer_params={'activation_fn': act})

    return out


def dilated2d(in_tensor, out_dim, name='dilated_conv', kernel_sz=3, act=None, dilation=1, bn=True):
    with tf.variable_scope(name):
        if bn:
            norm = l.batch_norm
            l_act = None
        else:
            norm = None
            l_act = act
        out = l.conv2d(in_tensor,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       rate=dilation,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act})
    return out


def upproject(in_tensor, out_dim, name='uproj_conv', kernel_sz=4, stride=2):
    return deconv2d(in_tensor, out_dim, name, kernel_sz, act=None, stride=stride, bn=True)


def prelu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def model(d_input, m, base_filters,p, d_flg = False, use_bn = False):
    net = {}
    input_with_mean = d_input + m
    net['gate'] = d_input
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
            net['convup1'] = conv2d(net['drop4'], base_filters * 16, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu,
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
        G_output = net['conv_out'] + net['gate'] + m

    return G_output

def predict(disp, mask):
    predicted = sess.run(G_output, feed_dict={target: np.exp(disp[None, ..., None]), mask_t: mask, d_flg: False})[0, :, :, 0]
    return predicted