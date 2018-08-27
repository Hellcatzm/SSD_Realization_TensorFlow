# Author : hellcat
# Time   : 18-8-24

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
np.set_printoptions(threshold=np.inf)

import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
"""
import tensorflow as tf


import util_tf
import tfr_data_process
import preprocess_img_tf
from ssd_vgg300_tf import SSDNet


slim = tf.contrib.slim


# =========================================================================== #
# Main training routine.
# =========================================================================== #
def main():

    max_steps = 1500
    batch_size = 32
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    opt_epsilon = 1.0
    num_epochs_per_decay = 2.0
    num_samples_per_epoch = 17125
    moving_average_decay = None

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():

        # Create global_step.
        with tf.device("/device:CPU:0"):
            global_step = tf.train.create_global_step()

        ssd = SSDNet()
        ssd_anchors = ssd.anchors

        # tfr解析操作放在GPU下有加速，效果不稳定
        dataset = \
            tfr_data_process.get_split('./TFR_Data',
                                       'voc2012_*.tfrecord',
                                       num_classes=21,
                                       num_samples=num_samples_per_epoch)

        with tf.device("/device:CPU:0"):  # 仅CPU支持队列操作
            image, glabels, gbboxes = \
                tfr_data_process.tfr_read(dataset)

            image, glabels, gbboxes = \
                preprocess_img_tf.preprocess_image(image, glabels, gbboxes, out_shape=(300, 300))

            gclasses, glocalisations, gscores = \
                ssd.bboxes_encode(glabels, gbboxes, ssd_anchors)

            batch_shape = [1] + [len(ssd_anchors)] * 3  # (1,f层,f层,f层)
            # Training batches and queue.
            r = tf.train.batch(  # 图片，中心点类别，真实框坐标，得分
                util_tf.reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=batch_size,
                num_threads=4,
                capacity=5 * batch_size)
            batch_queue = slim.prefetch_queue.prefetch_queue(
                r,  # <-----输入格式实际上并不需要调整
                capacity=2 * 1)

        # Dequeue batch.
        b_image, b_gclasses, b_glocalisations, b_gscores = \
            util_tf.reshape_list(batch_queue.dequeue(), batch_shape)  # 重整list

        predictions, localisations, logits, end_points = \
            ssd.net(b_image, is_training=True, weight_decay=0.00004)

        ssd.losses(logits, localisations,
                   b_gclasses, b_glocalisations, b_gscores,
                   match_threshold=.5,
                   negative_ratio=3,
                   alpha=1,
                   label_smoothing=.0)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # =================================================================== #
        # Configure the moving averages.
        # =================================================================== #
        if moving_average_decay:
            moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                moving_average_decay, global_step)
        else:
            moving_average_variables, variable_averages = None, None

        # =================================================================== #
        # Configure the optimization procedure.
        # =================================================================== #
        with tf.device("/device:CPU:0"):  # learning_rate节点使用CPU（不明）
            decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
            learning_rate = tf.train.exponential_decay(0.01,
                                                       global_step,
                                                       decay_steps,
                                                       0.94,  # learning_rate_decay_factor,
                                                       staircase=True,
                                                       name='exponential_decay_learning_rate')
            optimizer = tf.train.AdamOptimizer(
                learning_rate,
                beta1=adam_beta1,
                beta2=adam_beta2,
                epsilon=opt_epsilon)
            tf.summary.scalar('learning_rate', learning_rate)

        if moving_average_decay:
            # Update ops executed locally by trainer.
            update_ops.append(variable_averages.apply(moving_average_variables))

        # Variables to train.
        trainable_scopes = None
        if trainable_scopes is None:
            variables_to_train = tf.trainable_variables()
        else:
            scopes = [scope.strip() for scope in trainable_scopes.split(',')]
            variables_to_train = []
            for scope in scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                variables_to_train.extend(variables)

        losses = tf.get_collection(tf.GraphKeys.LOSSES)
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        regularization_loss = tf.add_n(regularization_losses)
        loss = tf.add_n(losses)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("regularization_loss", regularization_loss)

        grad = optimizer.compute_gradients(loss, var_list=variables_to_train)
        grad_updates = optimizer.apply_gradients(grad,
                                                 global_step=global_step)
        update_ops.append(grad_updates)
        # update_op = tf.group(*update_ops)

        with tf.control_dependencies(update_ops):
            total_loss = tf.add_n([loss, regularization_loss])
        tf.summary.scalar("total_loss", total_loss)

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config = tf.ConfigProto(log_device_placement=False,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=5,
                               keep_checkpoint_every_n_hours=1.0,
                               write_version=2,
                               pad_step_number=False)

        if True:
            import os
            import time

            print('start......')
            model_path = './logs'
            batch_size = batch_size
            with tf.Session(config=config) as sess:
                summary = tf.summary.merge_all()
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(sess=sess, coord=coord)
                writer = tf.summary.FileWriter(model_path, sess.graph)

                init_op = tf.group(tf.global_variables_initializer(),
                                   tf.local_variables_initializer())
                init_op.run()
                for step in range(max_steps):
                    start_time = time.time()
                    loss_value = sess.run(total_loss)
                    # loss_value, summary_str = sess.run([train_tensor, summary_op])
                    # writer.add_summary(summary_str, step)

                    duration = time.time() - start_time
                    if step % 10 == 0:
                        summary_str = sess.run(summary)
                        writer.add_summary(summary_str, step)

                        examples_per_sec = batch_size / duration
                        sec_per_batch = float(duration)
                        format_str = "[*] step %d,  loss=%.2f (%.1f examples/sec; %.3f sec/batch)"
                        print(format_str % (step, loss_value, examples_per_sec, sec_per_batch))
                    # if step % 100 == 0:
                    #     accuracy_step = test_cifar10(sess, training=False)
                    #     acc.append('{:.3f}'.format(accuracy_step))
                    #     print(acc)
                    if step % 500 == 0 and step != 0:
                        saver.save(sess, os.path.join(model_path, "ssd_tf.model"), global_step=step)

                coord.request_stop()
                coord.join(threads)


if __name__ == '__main__':
    main()

