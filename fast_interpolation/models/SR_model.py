import tensorflow as tf
from tensorflow.contrib import layers as l


def drop(x, p, is_training=True):
    noise_sh = [1, 1, 1, x.get_shape().as_list()[3]]
    return l.dropout(x, p, is_training=is_training, noise_shape=noise_sh)


def conv2d(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False):
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


def projection_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False):
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


def downscale_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=2, bn=False):
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


def downscale_pool(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=2, bn=False):
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


def residual_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False):
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


def deconv2d(in_tensor, out_dim, name='transposed_conv', kernel_sz=3, act=None, stride=1, bn=False):
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


def dilated2d(in_tensor, out_dim, name='dilated_conv', kernel_sz=3, act=None, dilation=1, bn=False):
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
    return deconv2d(in_tensor, out_dim, name, kernel_sz, act=None, stride=stride, bn=False)


def prelu(_x):
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.1),
                             dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def preprocess(d_small, size_big):
    d_small = (d_small - tf.reduce_min(d_small)) / (tf.reduce_max(d_small) -
                                                    tf.reduce_min(d_small))
    d_small = d_small + tf.constant(1.)
    d_upsampled = tf.image.resize_images(d_small, size_big, method=tf.image.ResizeMethod.BICUBIC)

    return d_small, d_upsampled


def preprocess_any(d_tensor, size_small, size_big):
    d_log = (d_tensor - tf.reduce_min(d_tensor)) / (tf.reduce_max(d_tensor) -
                                                    tf.reduce_min(d_tensor))
    d_log = d_log + tf.constant(1.)
    d_small = tf.image.resize_images(d_log, size_small, method=tf.image.ResizeMethod.AREA)
    d_upsampled = tf.image.resize_images(d_small, size_big, method=tf.image.ResizeMethod.BICUBIC)
    ground_truth = tf.image.resize_images(d_log, size_big, method=tf.image.ResizeMethod.BILINEAR)
    #ground_truth = d_log

    return ground_truth, d_small, d_upsampled

def get_placeholders():
    target = tf.placeholder(tf.float32, shape=(None, None, None, 1))
    d_flg = tf.placeholder(tf.bool)
    return target, d_flg

def model(d_input, interpolated, base_filters,p, d_flg = False):
    net = {}
    net['gate'] = d_input
    with tf.variable_scope('G_Depth_SR'):
        net['conv1'] = conv2d(net['gate'], base_filters, 'convolution_1', kernel_sz=3, stride=1, act=tf.nn.elu)
        net['conv2'] = projection_block(net['conv1'], base_filters * 2, 'convolution_2', kernel_sz=3, act=tf.nn.elu)
        net['drop1'] = drop(net['conv2'], p, is_training=d_flg)
        net['conv3'] = residual_block(net['drop1'], base_filters * 2, 'convolution_3', kernel_sz=3, act=tf.nn.elu)
        net['conv4'] = residual_block(net['conv3'], base_filters * 2, 'convolution_4', kernel_sz=3, act=tf.nn.elu)
        net['drop2'] = drop(net['conv4'], p, is_training=d_flg)
        net['conv5'] = residual_block(net['drop2'], base_filters * 2, 'convolution_5', kernel_sz=3, act=tf.nn.elu)
        net['conv6'] = residual_block(net['conv5'], base_filters * 2, 'convolution_6', kernel_sz=3, act=tf.nn.elu)
        net['drop3'] = drop(net['conv6'], p, is_training=d_flg)
        net['conv7'] = conv2d(net['drop3'], base_filters * 4, 'upconv_1', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv1'] = tf.depth_to_space(net['conv7'], 2)
        net['conv8'] = conv2d(net['uconv1'], base_filters * 2, 'upconv_2', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv2'] = tf.depth_to_space(net['conv8'], 2)
        net['conv9'] = deconv2d(net['uconv2'], base_filters, 'upconv_3', stride=1, kernel_sz=3, act=tf.nn.elu)
        net['uconv3'] = tf.depth_to_space(net['conv9'], 2)
        net['conv_out'] = conv2d(net['uconv3'], 1, 'convolution_out', kernel_sz=3, act=None, bn=False)
    G_output = net['conv_out'] + interpolated
    return G_output



def predict(disp_small):
    result = sess.run(G_output, feed_dict={d_small:disp_small[None,:,:,None], d_flg:False})[0,:,:,0].astype(float)
    return result