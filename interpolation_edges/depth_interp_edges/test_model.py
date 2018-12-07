
import numpy as np

import tensorflow as tf
from model import network



class TrainTest(tf.test.TestCase):

  def test_input(self):
    batch_size = 1

    # Mock input pipeline.
    mock_imgs = np.ones([batch_size,  32,32, 1], dtype=np.float32)
    mock_edges = np.ones([batch_size, 32,32, 1], dtype=np.float32)
    mock_interp = np.ones([batch_size,  32,32, 1], dtype=np.float32)

    t_image = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_input')
    t_interpolated = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_input')
    t_edges = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_input')
    d_flg = tf.placeholder(tf.bool, name = 'is_train')

    net_g = network(t_image,t_edges,is_train=d_flg)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    result = sess.run(net_g,feed_dict={t_image: mock_imgs, t_interpolated:mock_interp, t_edges:mock_edges, d_flg: False})
    print(result.shape)


  def test_build_graph(self):
    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder(tf.float32, shape=(None, 32,32, 1), name='t_input')
    t_interpolated = tf.placeholder(tf.float32, shape=(None, 32,32, 1), name='t_input')
    t_edges = tf.placeholder(tf.float32, shape=(None, 32,32, 1), name='t_input')
    d_flg = tf.placeholder(tf.bool, name = 'is_train')

    net_g = network(t_image,t_edges, t_interpolated, is_train=d_flg)
    print('out',net_g.shape)

if __name__ == '__main__':
  tf.test.main()