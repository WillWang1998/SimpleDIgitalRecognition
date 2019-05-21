import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_cnn as mnist_interence
import mnist_train
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_TATE = 0.0001
MOVING_AVERAGE_DECAY = 0.99
TRAIN_STEP = 30000
MODEL_PATH = 'model'
MODEL_NAME = 'model'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(mnist):
    x = tf.placeholder(tf.float32, shape=[None,
                                          mnist_interence.IMAGE_SIZE,
                                          mnist_interence.IMAGE_SIZE,
                                          mnist_interence.NUM_CHANNEL], name='x-input')
    y_ = tf.placeholder(tf.float32, shape=[None, mnist_interence.OUTPUT_NODE], name='y-input')
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_TATE)
    y = mnist_interence.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_ops = variable_average.apply(tf.trainable_variables())
    cross_entroy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entroy_mean = tf.reduce_mean(cross_entroy)
    loss = cross_entroy_mean + tf.add_n(tf.get_collection('loss'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                               mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss, global_step=global_step)
    train_op = tf.group(train_step, variable_average_ops)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            for i in range(eval(global_step), TRAIN_STEP):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                reshape_xs = np.reshape(xs, (BATCH_SIZE, mnist_interence.IMAGE_SIZE,
                                             mnist_interence.IMAGE_SIZE,
                                             mnist_interence.NUM_CHANNEL))
                _, loss_value, step, learn_rate = sess.run([train_op, loss, global_step, learning_rate],
                                                           feed_dict={x: reshape_xs, y_: ys})
                if (i + 1) % 3000 == 0:
                    print('After %d step, loss on train is %g,and learn rate is %g' % (step, loss_value, learn_rate))
                    saver.save(sess, os.path.join(MODEL_PATH, MODEL_NAME), global_step=global_step)
        else:
            print('No Checkpoint file find')


def main():
    mnist = input_data.read_data_sets('./mni_data', one_hot=True)
    train(mnist)


if __name__ == "__main__":
    main()