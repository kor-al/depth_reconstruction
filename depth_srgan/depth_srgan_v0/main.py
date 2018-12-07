#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy

from iterators import *

import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config
from tqdm import tqdm
import time
from vgg.vgg16 import VGG16
import random

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

small_size = config.TRAIN.small_size
big_size = config.TRAIN.big_size
vgg16_npy_path = "D:/depth_dev/YandexDisk/korinevskaya-gtx1080/depth_srgan/vgg/vgg16.npy"

ni = int(np.sqrt(batch_size))

# init_g_model_dir = 'D:/depth_dev/srres_ps/model.ckpt'
#init_g_model_dir = 'D:/depth_dev/YandexDisk/korinevskaya-gtx1080/depth_srgan/checkpoint/gan/model16.ckpt'

do_init_g = True


def preprocess(d_tensor_frame, to_crop=True, bs=batch_size):
    if to_crop:
        d_tensor = tf.random_crop(d_tensor_frame, [bs, *big_size, 1])
    else:
        d_tensor = d_tensor_frame
    d_norm = (d_tensor - tf.reduce_min(d_tensor)) / (tf.reduce_max(d_tensor) - tf.reduce_min(d_tensor))
    d_norm = tf.cast(d_norm, tf.float32) * (2.) - tf.constant(1.)
    d_small = tf.image.resize_images(d_norm, small_size, method=tf.image.ResizeMethod.AREA)
    if to_crop:
        d_big = d_norm
    else:
        d_big = tf.image.resize_images(d_norm, big_size, method=tf.image.ResizeMethod.BILINEAR)
    d_interpolated = tf.image.resize_images(d_small, big_size, method=tf.image.ResizeMethod.BILINEAR)

    return d_small, d_big, d_interpolated

def preprocess_any(d_tensor, ss, bs):

    d_norm = (d_tensor - tf.reduce_min(d_tensor)) / (tf.reduce_max(d_tensor)- tf.reduce_min(d_tensor))
    d_norm = tf.cast(d_norm, tf.float32) * (2.) - tf.constant(1.)
    d_small = tf.image.resize_images(d_norm, ss, method=tf.image.ResizeMethod.AREA)
    d_big = tf.image.resize_images(d_norm, bs, method=tf.image.ResizeMethod.BILINEAR)
    d_interpolated = tf.image.resize_images(d_small, bs, method=tf.image.ResizeMethod.BILINEAR)
    return d_small, d_big,  d_interpolated


def train():
    n_epoch_init = 12
    ## create folders to save result images and trained model
    # save_dir_ginit = "samples/{}_ginit".format(tl.global_flag['mode'])
    # save_dir_gan = "samples/{}_gan".format(tl.global_flag['mode'])
    # tl.files.exists_or_mkdir(save_dir_ginit)
    # tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    log_dir = "logs"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    ###====================== PRE-LOAD DATA ===========================###

    train_hr_img_list = sorted(
        get_synthia_imgs_list(config.VALID.hr_img_path, is_train=True, synthia_dataset=config.TRAIN.hr_img_path))
    valid_hr_img_list = sorted(
        get_synthia_imgs_list(config.VALID.hr_img_path, is_train=False, synthia_dataset=config.TRAIN.hr_img_path))
    print(len(train_hr_img_list))
    print(len(valid_hr_img_list))

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_input = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_input')
    # try with log?
    t_input = tf.log(t_input)

    d_flg = tf.placeholder(tf.bool, name='is_train')

    t_image, t_target_image, t_interpolated = preprocess(t_input)

    net_g_outputs = SRGAN_g(t_image, t_interpolated, is_train=d_flg, reuse=False)

    net_d, logits_real = SRGAN_d(t_target_image, is_train=d_flg, reuse=False)
    _, logits_fake = SRGAN_d(net_g_outputs, is_train=d_flg, reuse=True)

    vgg_model_true = VGG16(vgg16_npy_path)
    vgg_model_gen = VGG16(vgg16_npy_path)

    ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
    # to 3 channels
    y_true_normalized = (t_target_image - tf.reduce_min(t_target_image)) / (tf.reduce_max(t_target_image)
                                                                            - tf.reduce_min(t_target_image))
    gen_normalized = (net_g_outputs - tf.reduce_min(net_g_outputs)) / (tf.reduce_max(net_g_outputs)
                                                                       - tf.reduce_min(net_g_outputs))

    t_target_image_3ch = tf.concat([y_true_normalized] * 3, 3)
    t_predict_image_3ch = tf.concat([gen_normalized] * 3, 3)

    vgg_model_true.build(t_target_image_3ch)
    true_features = vgg_model_true.conv3_1
    vgg_model_gen.build(t_predict_image_3ch)
    gen_features = vgg_model_gen.conv3_1

    ## test inference
    net_g_test = SRGAN_g(t_image, t_interpolated, is_train=d_flg, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real), name='d1')
    d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake), name='d2')
    # d_vgg_loss =  2e-6*tl.cost.mean_squared_error(true_features, gen_features, is_mean=True)

    d_loss = d_loss1 + d_loss2

    g_gan_loss = 1e-2 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake), name='g')  # 1e-3 *
    mse_loss = tl.cost.mean_squared_error(net_g_outputs, t_target_image, is_mean=True)
    vgg_loss = 2e-6 * tl.cost.mean_squared_error(true_features, gen_features, is_mean=True)  # 2e-6 *
    tv_loss = 2e-6 * tf.reduce_mean(tf.square(net_g_outputs[:, :-1, :, :] - net_g_outputs[:, 1:, :, :])) + \
              tf.reduce_mean(tf.square(net_g_outputs[:, :, :-1, :] - net_g_outputs[:, :, 1:, :]))  # 2e-6*

    g_init_loss = mse_loss + vgg_loss# mse_loss # + vgg_loss + tv_loss
    g_loss =  g_gan_loss  + mse_loss +vgg_loss# + mse_loss

    g_vars = tl.layers.get_variables_with_name('G_Depth_SR', True, True)
    d_vars = tl.layers.get_variables_with_name('SRGAN_d', True, True)

    glob_step_t = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
        ## Pretrain
        g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_init_loss, var_list=g_vars,
                                                                          global_step=glob_step_t)
        ## SRGAN
        g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars, global_step=glob_step_t)
        d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)

    ###========================== RESTORE MODEL =============================###

    saver = tf.train.Saver(max_to_keep=5)
    saver_d = tf.train.Saver(d_vars, max_to_keep=5)
    saver_g = tf.train.Saver(g_vars, max_to_keep=5)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    with tf.variable_scope('summaries'):
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.scalar('mse_loss', mse_loss)
        tf.summary.scalar('vgg_loss', vgg_loss)
        tf.summary.scalar('tv_loss', tv_loss)
        tf.summary.scalar('g_gan_loss', g_gan_loss)
        mae = tf.reduce_mean(tf.abs(net_g_outputs - t_target_image) / (t_target_image + tf.constant(1e-8)))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(net_g_outputs - t_target_image)))
        tf.summary.scalar('MAE', mae)
        tf.summary.scalar('RMSE', rmse)
        tf.summary.scalar('learning_rate', lr_v)
        # tf.summary.image('input', t_input , max_outputs=1)
        tf.summary.image('GT', t_target_image, max_outputs=1)
        tf.summary.image('input_small_size', t_image, max_outputs=1)
        tf.summary.image('interpolated', t_interpolated, max_outputs=1)
        tf.summary.image('result', net_g_outputs, max_outputs=1)
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(log_dir + '/test')

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    # sample_imgs = train_hr_imgs[0:batch_size]
    # sample_imgs = tl.vis.read_images(train_hr_img_list[0:batch_size], path=config.TRAIN.hr_img_path, n_threads=32) # if no pre-load train set

    # sample_imgs = tl.prepro.threading_data(train_hr_img_list[0:batch_size], fn=get_imgs_fn)  # if no pre-load train set
    # print('sample images:', sample_imgs.shape, sample_imgs.min(), sample_imgs.max())

    n_batches = int(len(train_hr_img_list) / batch_size)
    n_batches_valid = int(len(valid_hr_img_list) / batch_size)

    ###========================= initialize G ====================###

    if not do_init_g:
        n_epoch_init = -1
        try:
            saver_g.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/g_init'))
        except Exception as e:
            print(' ** You need to initialize generator: put do_init_g to True or provide a valid restore path')
            raise e

    else:
        try:
            #saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir+'/gan')) # 2 round
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir + '/g_init'))
        except:
            print(' ** Creating new g_init model')
            pass

    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    train_iter, test_iter = 0, 0
    for epoch in range(0, n_epoch_init + 1):
        try:
            epoch_time = time.time()

            val_mae, val_mse, val_g_loss = 0, 0, 0
            batch_it = tqdm(SynthiaIterator(valid_hr_img_list, batchsize=batch_size, shuffle=True, buffer_size=70),
                            total=n_batches_valid, leave=False)
            for b in batch_it:
                xb = b[0]
                errM, errG, mae_score = sess.run([mse_loss, g_loss, mae], feed_dict={t_input: xb, d_flg: False})
                val_mae += mae_score
                val_mse += errM
                val_g_loss += errG

            print("Validation: Epoch {0} val mae {1} val mse {2}".format(epoch - 1, val_mae / n_batches_valid,
                                                                         val_mse / n_batches_valid))

            total_mse_loss, total_g_loss = 0, 0
            batch_it = tqdm(SynthiaIterator(train_hr_img_list, batchsize=batch_size, shuffle=True, buffer_size=70),
                            total=n_batches, leave=False)
            for b in batch_it:
                xb = b[0]
                xb = augment_imgs(xb)
                glob_step, errM, errG, _ = sess.run([glob_step_t, mse_loss, g_loss, g_optim_init],
                                                    feed_dict={t_input: xb, d_flg: True})

                total_mse_loss += errM
                total_g_loss += errG
                if (train_iter + 1) % 200 == 0:
                    summary = sess.run(summary_op, feed_dict={t_input: xb, d_flg: False})
                    train_writer.add_summary(summary, train_iter + 1)

                train_iter += 1

            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f" % (
            epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_batches)

            val_mse_summary = tf.Summary.Value(tag='g_init/val_mse_loss', simple_value=val_mse / n_batches_valid)
            val_g_loss_summary = tf.Summary.Value(tag='g_init/val_loss', simple_value=val_g_loss / n_batches_valid)

            train_mse_loss_summary = tf.Summary.Value(tag='g_init/train_mse_loss',
                                                      simple_value=total_mse_loss / n_batches)
            train_g_loss_summary = tf.Summary.Value(tag='g_init/train_loss', simple_value=total_g_loss / n_batches)

            epoch_summary = tf.Summary(
                value=[val_mse_summary, val_g_loss_summary, train_mse_loss_summary, train_g_loss_summary])

            train_writer.add_summary(epoch_summary, glob_step)

            print(log)
            saver.save(sess, os.path.join(checkpoint_dir + '/g_init', 'model' + str(epoch) + '.ckpt'))

        except Exception as e:
            batch_it.iterable.stop()
            raise e


    ###========================= train GAN (SRGAN) =========================###
    try:
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir+'/g_init'))
        # saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir+'/gan'))
        pass
    except:
        print(' ** Creating new GAN model')
        pass

    train_iter, test_iter = 0, 0
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        try:
            epoch_time = time.time()

            val_mae, val_mse, val_g_loss, val_d_loss = 0, 0, 0, 0
            batch_it = tqdm(SynthiaIterator(valid_hr_img_list, batchsize=batch_size, shuffle=True, buffer_size=70),
                            total=n_batches_valid, leave=False)
            for b in batch_it:
                xb = b[0]
                errM, mae_score, errG, errD = sess.run([mse_loss, mae, g_loss, d_loss],
                                                       feed_dict={t_input: xb, d_flg: False})
                val_mae += mae_score
                val_mse += errM
                val_g_loss += errG
                val_d_loss += errD

            print("Validation (GAN): Epoch {0} val mae {1} val mse {2}".format(epoch - 1, val_mae / n_batches_valid,
                                                                               val_mse / n_batches_valid))

            total_d_loss, total_g_loss, total_mse_loss = 0, 0, 0
            batch_it = tqdm(SynthiaIterator(train_hr_img_list, batchsize=batch_size, shuffle=True, buffer_size=70),
                            total=n_batches, leave=False)
            for b in batch_it:
                xb = b[0]
                xb = augment_imgs(xb)
                ## update D
                errD, _ = sess.run([d_loss, d_optim], {t_input: xb, d_flg: True})
                ## update G
                glob_step, errG, errM, _, summary = sess.run([glob_step_t, g_loss, mse_loss, g_optim, summary_op],
                                                             {t_input: xb, d_flg: True})
                total_mse_loss += errM
                total_d_loss += errD
                total_g_loss += errG
                if (train_iter + 1) % 10 == 0:
                    train_writer.add_summary(summary, train_iter + 1)

                train_iter += 1

        except Exception as e:
            batch_it.iterable.stop()
            raise e
            break

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f mse_loss: %.8f" % (
            epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_batches, total_g_loss / n_batches,
            total_mse_loss / n_batches)

        val_mse_summary = tf.Summary.Value(tag='gan/val_mse_loss', simple_value=val_mse / n_batches_valid)
        val_g_loss_summary = tf.Summary.Value(tag='gan/val_g_loss', simple_value=val_g_loss / n_batches_valid)
        val_d_loss_summary = tf.Summary.Value(tag='gan/val_d_loss', simple_value=val_d_loss / n_batches_valid)

        train_mse_loss_summary = tf.Summary.Value(tag='gan/train_mse_loss', simple_value=total_mse_loss / n_batches)
        train_g_loss_summary = tf.Summary.Value(tag='gan/train_g_loss', simple_value=total_g_loss / n_batches)
        train_d_loss_summary = tf.Summary.Value(tag='gan/train_d_loss', simple_value=total_d_loss / n_batches)

        epoch_summary = tf.Summary(
            value=[val_mse_summary, val_g_loss_summary, val_d_loss_summary, train_mse_loss_summary,
                   train_g_loss_summary, train_d_loss_summary])

        train_writer.add_summary(epoch_summary, glob_step)

        print(log)
        saver.save(sess, os.path.join(checkpoint_dir + '/gan', 'model' + str(n_epoch_init + epoch) + '.ckpt'))


'''

def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)


    valid_hr_img_list = sorted(get_synthia_imgs_list(config.VALID.hr_img_path, is_train=False))
    valid_hr_imgs = tl.prepro.threading_data(valid_hr_img_list, thread_count=32, fn=get_imgs_fn)

    for im in valid_hr_imgs:
        print(im.shape)
    # exit()


    ###========================== RESTORE G =============================###

    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    times = []
    totalim = len(valid_lr_imgs)
    for imid in range(0, totalim):
        ###========================== DEFINE MODEL ============================###
        # imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        valid_hr_img = preprocess_fn(valid_hr_imgs[imid], is_small=False, normalize=True)
        valid_lr_img = preprocess_fn(valid_hr_img, is_small=True, normalize=False)
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        # valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size


        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))
        times.append((time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (
        size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/gen/valid_gen' + str(imid) + '.png')
        tl.vis.save_image(out[0], save_dir + '/valid_gen' + str(imid) + '.png')

        tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr' + str(imid) + '.png')
        tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr' + str(imid) + '.png')

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic' + str(imid) + '.png')

    print('Average time per SR operation = ', np.mean(times))


def evaluate2path():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    for im in valid_lr_imgs:
        print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    for im in valid_hr_imgs:
        print(im.shape)
    # exit()


    ###========================== RESTORE G =============================###

    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    times = []
    totalim = len(valid_lr_imgs)
    for imid in range(0, totalim):
        ###========================== DEFINE MODEL ============================###
        # imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
        valid_lr_img = valid_lr_imgs[imid]
        valid_hr_img = valid_hr_imgs[imid]
        # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
        valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
        # print(valid_lr_img.min(), valid_lr_img.max())

        size = valid_lr_img.shape
        # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size


        ###======================= EVALUATION =============================###
        start_time = time.time()
        out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
        print("took: %4.4fs" % (time.time() - start_time))
        times.append((time.time() - start_time))

        print("LR size: %s /  generated HR size: %s" % (
        size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
        print("[*] save images")
        tl.vis.save_image(out[0], save_dir + '/gen/valid_gen' + str(imid) + '.png')
        tl.vis.save_image(out[0], save_dir + '/valid_gen' + str(imid) + '.png')

        tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr' + str(imid) + '.png')
        tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr' + str(imid) + '.png')

        out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
        tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic' + str(imid) + '.png')

    print('Average time per SR operation = ', np.mean(times))
    '''


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
