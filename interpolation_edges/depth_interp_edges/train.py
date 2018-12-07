import os

from iterators import *

import tensorlayer as tl
from matplotlib import pyplot as plt
from model import network
from tqdm import tqdm
import time
from vgg.vgg16 import VGG16

from config import *
from preprocessing import *
from utils.utils import *

batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init

n_epoch = config.TRAIN.n_epoch

lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

t_size = config.TRAIN.model_size

mask_type = config.TRAIN.mask_type

vgg16_npy_path = config.TRAIN.vgg_path

ni = int(np.sqrt(batch_size))


def _get_loss(net_g_outputs, t_target_image):
    with tf.variable_scope('loss'):
        vgg_model_true = VGG16(vgg16_npy_path)
        vgg_model_gen = VGG16(vgg16_npy_path)

        ## vgg inference. 0, 1, 2, 3 BILINEAR NEAREST BICUBIC AREA
        # to 3 channels
        y_true_normalized = (t_target_image - tf.reduce_min(t_target_image)) / (tf.reduce_max(t_target_image)- tf.reduce_min(t_target_image))
        gen_normalized = (net_g_outputs - tf.reduce_min(net_g_outputs)) / (tf.reduce_max(net_g_outputs)- tf.reduce_min(net_g_outputs))

        t_target_image_3ch = tf.concat([y_true_normalized] * 3, 3)
        t_predict_image_3ch = tf.concat([gen_normalized] * 3, 3)

        vgg_model_true.build(t_target_image_3ch)
        true_features = vgg_model_true.conv3_1
        vgg_model_gen.build(t_predict_image_3ch)
        gen_features = vgg_model_gen.conv3_1


        mse_loss = tf.reduce_mean(tf.square(t_target_image - net_g_outputs))
        vgg_loss =  5e-5**tf.reduce_mean(tf.square(true_features - gen_features))
        tv_loss = 2e-6*tf.reduce_mean(tf.square(net_g_outputs[:, :-1, :, :] - net_g_outputs[:, 1:, :, :])) + \
                  tf.reduce_mean(tf.square(net_g_outputs[:, :, :-1, :] - net_g_outputs[:, :, 1:, :]))  #2e-6*

        g_loss =  mse_loss + vgg_loss + tv_loss

    return g_loss, mse_loss, vgg_loss, tv_loss


def train():
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    log_dir = "logs"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)
    tl.files.exists_or_mkdir(log_dir)

    ###====================== PRE-LOAD DATA ===========================###

    train_d_img_list, train_rgb_img_list = sorted(get_synthia_imgs_list(config.VALID.hr_img_path, is_train=True, synthia_dataset=config.TRAIN.hr_img_path, return_rgb = True))
    valid_d_img_list, valid_rgb_img_list = sorted(get_synthia_imgs_list(config.VALID.hr_img_path, is_train=False, synthia_dataset=config.TRAIN.hr_img_path, return_rgb = True))
    print('Total Train Examples = ', len(train_d_img_list))
    print('Total Val Examples = ', len(valid_d_img_list))

    ##---sample images -----------

    val_image_s = open_rgb(valid_rgb_img_list[0])[None, :, :, :]
    val_gt_s = open_depth_synthia(valid_d_img_list[0], debug=True)[None]
    #val_image_d = open_rgb(valid_rgb_img_list[-1])[None, :, :, :]
    #val_gt_d = open_depth_synthia(valid_d_img_list[-1])[None, :, :, :]
    grad_ms = get_grad_mask(val_image_s, t_size)
    #grad_md = get_grad_mask(val_image_d, t_size)
    edge_s = get_edges_from_rgb(val_image_s, t_size)
    #edge_d = get_edges_from_rgb(val_image_d, t_size)

    ###========================== DEFINE MODEL ============================###
    ## train inference
    t_input_d = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_input')
    #t_input_rgb = tf.placeholder(tf.float32, shape=(None, None, None, 3), name='t_input_rgb')
    t_edges = tf.placeholder(tf.bool, shape=(None, None, None, 1), name='t_edges')
    t_mask = tf.placeholder(tf.float32, shape=(None, None, None, 1), name='t_mask')

    d_flg = tf.placeholder(tf.bool, name = 'is_train')

    t_depth, mean_d, t_target_depth, t_edges = preprocess(t_input_d, t_mask, t_edges, t_size)
    tf.summary.image('masked_depth',t_depth, max_outputs=1)
    tf.summary.image('edges',t_edges, max_outputs=1)

    net_g_outputs = network(t_depth,t_edges,is_train=d_flg)
    net_g_outputs = return2original_scale(net_g_outputs, mean_d)

    g_loss, mse_loss, vgg_loss, tv_loss = _get_loss(net_g_outputs, t_target_depth)

    glob_step_t = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
        optim = tf.train.AdamOptimizer(lr_v).minimize(g_loss, global_step=glob_step_t)

    with tf.variable_scope('summaries'):
        e_summary = tf.summary.image('edges',t_edges, max_outputs=1)
        lr_summary = tf.summary.scalar('learning_rate', lr_v)
        gt_summary = tf.summary.image('gt_depth', t_target_depth, max_outputs=1)
        in_summary = tf.summary.image('input', t_depth, max_outputs=1)
        out_summary = tf.summary.image('output', net_g_outputs, max_outputs=1)

        loss_summary = tf.summary.scalar('g_loss', g_loss)
        mse_loss_summary = tf.summary.scalar('mse_loss', mse_loss)
        mae = tf.reduce_mean(tf.abs(net_g_outputs - t_target_depth) / (net_g_outputs + eps))
        rmse = tf.sqrt(tf.reduce_mean(tf.square(net_g_outputs - t_target_depth)))
        mae_summary = tf.summary.scalar('MAE', mae)
        rmse_summary = tf.summary.scalar('RMSE', rmse)

        all_summaries = tf.summary.merge([mae_summary,rmse_summary, in_summary, out_summary, loss_summary, mse_loss_summary])
        metrics_summaries = tf.summary.merge([mae_summary,rmse_summary, loss_summary, mse_loss_summary])

        train_writer = tf.summary.FileWriter(log_dir)

        writer_ground_truths = tf.summary.FileWriter(logdir=os.path.join(log_dir, 'img_gt', 's'))
        writer_ground_truthd = tf.summary.FileWriter(logdir=os.path.join(log_dir, 'img_gt', 'd'))
        writer_grads = tf.summary.FileWriter(logdir=os.path.join(log_dir, 'grad', 's'))
        writer_gradd = tf.summary.FileWriter(logdir=os.path.join(log_dir, 'grad', 'd'))
    ###========================== RESTORE MODEL =============================###

    saver = tf.train.Saver(max_to_keep=5)
    #sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=cfg)

    sess.run(tf.global_variables_initializer())

    ###============================= TRAINING ===============================###
    writer_ground_truths.add_summary(gt_summary.eval({t_input_d: val_gt_s}))
    #writer_ground_truthd.add_summary(gt_summary.eval({t_input_d: val_gt_d}))



    n_batches = int(len(train_d_img_list) / batch_size)
    n_batches_valid = int(len(valid_d_img_list) / batch_size)

    try:
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
    except:
        print(' ** Creating a new model')

    glob_step, train_iter, test_iter = 0,0,0
    cur_lr = lr_init

    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f" % (lr_init * new_lr_decay)
            print(log)
            cur_lr = lr_init * new_lr_decay
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f " % (lr_init, decay_every, lr_decay)
            print(log)

        try:
            #First: Validation
            epoch_time = time.time()
            val_mse, val_g_loss = 0,0
            batch_it = tqdm(SynthiaIterator(valid_d_img_list,valid_rgb_img_list, batchsize=batch_size, shuffle=True, buffer_size=70),total=n_batches_valid, leave=False)

            for xb, yb in batch_it:
                maskb = get_mask(yb, t_size, 'grad')
                edgeb = get_edges_from_rgb(yb, t_size)

                errM, errG = sess.run([mse_loss, g_loss],feed_dict={t_input_d: xb,t_mask: maskb, t_edges: edgeb, d_flg:False})

                val_mse+=errM
                val_g_loss+=errG

            print("Validation: Epoch {0}  val mse {1}".format(epoch - 1, val_mse/n_batches_valid))

            #Second: Training
            total_g_loss, total_mse_loss = 0,0
            batch_it = tqdm(SynthiaIterator(valid_d_img_list,valid_rgb_img_list,batchsize=batch_size, shuffle=True, buffer_size=70),total=n_batches, leave=False)

            for xb,yb in batch_it:

                #flip images
                xb,yb = augment_imgs(xb,yb)
                maskb = get_mask(yb, t_size, 'grad')
                edgeb = get_edges_from_rgb(yb, t_size)

                ## update G
                glob_step, errG, errM,_ = sess.run([glob_step_t, g_loss, mse_loss, optim],
                                                   {t_input_d: xb,t_edges: edgeb, t_mask:maskb, d_flg:True})

                total_mse_loss +=errM
                total_g_loss += errG

                if (train_iter + 1)%10 == 0:
                    #mask_uni = np.random.choice([0, 1], size=t_size, p=[1 - 0.1, 0.1])[None, :, :, None]
                    
                    sum_grad = all_summaries.eval( {t_edges: edge_s, t_mask: grad_ms, t_input_d: val_gt_s, d_flg: False})
                    writer_grads.add_summary(sum_grad, train_iter)
                    #sum_grad = all_summaries.eval({t_edges: edge_d, t_mask: grad_md, t_input_d: val_gt_d, d_flg: False})
                    #writer_gradd.add_summary(sum_grad, train_iter)

                train_iter += 1

        except Exception as e:
            batch_it.iterable.stop()
            raise e
            break


        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f mse_loss: %.8f" % (
        epoch, n_epoch, time.time() - epoch_time, total_g_loss / n_batches, total_mse_loss/n_batches)


        val_mse_summary = tf.Summary.Value(tag='val/mse_loss', simple_value= val_mse/n_batches_valid)
        val_g_loss_summary = tf.Summary.Value(tag='val/g_loss', simple_value= val_g_loss/n_batches_valid)

        train_mse_loss_summary = tf.Summary.Value(tag='train/mse_loss', simple_value=total_mse_loss / n_batches)
        train_g_loss_summary = tf.Summary.Value(tag='train/g_loss', simple_value=total_g_loss / n_batches)

        epoch_summary = tf.Summary(value=[val_mse_summary,val_g_loss_summary, train_mse_loss_summary, train_g_loss_summary])

        train_writer.add_summary(epoch_summary, glob_step)
        train_writer.add_summary(lr_summary.eval(), glob_step)

        print(log)
        saver.save(sess, os.path.join(checkpoint_dir,'model' + str(epoch)+'.ckpt'))





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train', help='train, valid, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'train':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        raise NotImplementedError("to do")
        #evaluate()
    elif tl.global_flag['mode'] == 'valid':
        raise NotImplementedError("to do")
        #run_model_on_test_set()
    else:
        raise Exception("Unknow --mode")
