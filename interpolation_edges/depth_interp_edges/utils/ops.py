import tensorflow as tf
from tensorflow.contrib import layers as l
from tensorflow import layers as tfl


def drop(x, p, is_training=True):
    #P is keep_prob
    noise_sh = [1, 1, 1, x.get_shape().as_list()[3]]
    return l.dropout(x, p, is_training=is_training, noise_shape=noise_sh)
    
# The implementation of PixelShuffler
#https://github.com/brade31919/SRGAN-tensorflow/blob/master/lib/ops.py
def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)


def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output

def conv2d(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False, is_training = True):
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
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
    return out


def projection_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False, is_training = True):
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
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
        proj = l.conv2d(in_tensor,
                        num_outputs=out_dim,
                        kernel_size=1,
                        stride=stride,
                        normalizer_fn=norm,
                        activation_fn=None,
                        padding='SAME',
                        normalizer_params={'activation_fn': act, 'is_training': is_training})
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


def downscale_pool(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=2, bn=False, is_training = True):
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
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=1,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act, 'is_training': is_training})

        down = l.avg_pool2d(in_tensor, 3, 2, padding='SAME')
        proj = l.conv2d(down,
                        num_outputs=out_dim,
                        kernel_size=1,
                        stride=1,
                        normalizer_fn=norm,
                        activation_fn=None,
                        padding='SAME',
                        normalizer_params={'activation_fn': None, 'is_training': is_training})
    return out + proj


def residual_block(in_tensor, out_dim, name='conv', kernel_sz=3, act=None, stride=1, bn=False, is_training=True):
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
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
        out = l.conv2d(out,
                       num_outputs=out_dim,
                       kernel_size=kernel_sz,
                       stride=stride,
                       normalizer_fn=norm,
                       activation_fn=l_act,
                       padding='SAME',
                       normalizer_params={'activation_fn': act, 'is_training': is_training})
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
