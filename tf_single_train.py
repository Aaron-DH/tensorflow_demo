# -*- coding:utf-8 -*-

import os

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

MOVING_AVG_DECAY = 0.99

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 200,
                     'The number of pictures feed to Neural Network once. '
                     'Should be > 0 ,default is 200')
flags.DEFINE_float('learning_rate_base', 0.005,
                   'The base learning rate for Neural Network training '
                   'Default is 0.005')
flags.DEFINE_float('learning_rate_decay', 0.99,
                   'The decay of learning rate. '
                   'Default is 0.99')
flags.DEFINE_float('regularizer', 0.0001,
                   'The regularizer for Neural Network training. '
                   'Default is 0.0001')
flags.DEFINE_integer('steps', 10000,
                     'The total traning times. '
                     'Default is 10000')
flags.DEFINE_integer('eval_step', 1000,
                     'How many steps to eval train result. '
                     'Default is 1000')
flags.DEFINE_string('model_name', 'mnist_cnn',
                    'The name of train model. '
                    'Default is mnist_cnn')
flags.DEFINE_string('model_dir', '/root/tensorflow/mnist/model',
                     'The models trained save path. '
                     'Default is /root/tensorflow//mnist/model')
flags.DEFINE_string('data_dir', '/root/tensorflow/mnist/data',
                     'The path of training dataset path. '
                     'Default is /root/tensorflow/mnist/data')
flags.DEFINE_string('log_dir', '/root/tensorflow/log',
                     'The path of trained logs path. '
                     'Default is /root/tensorflow/mnist/log')
FLAGS = flags.FLAGS

# mnsit数据集的边长为28像素
MNIST_IMAGE_SIZE = 28
# mnsit数据集为灰度单通道的图像，因此固定为1
MNIST_IMAGE_CHANNELS = 1

# 第一层卷积使用的是5 × 5的图像
CONV1_KERNEL_SIZE = 5
# 总共使用了32个卷积核
CONV1_KERNEL_NUM = 32

# 第二层卷积使用的是 5 × 5 的图像
CONV2_KERNEL_SIZE = 5
# 总共使用了64个卷积核
CONV2_KERNEL_NUM = 64

# 设定卷积核的滑动步长，行和列的步长一致
CONV_STEP_SIZE = 1

# 设定池化核为2 × 2的图像
POOL_KERNEL_SIZE = 2
# 设定池化核的滑动步长，行和列的步长一致
POOL_STEP_SIZE = 2

# 设定第一个全连接网络为512个节点（神经元）
FC_CONNECTED_SIZE = 512

# 表示输出10个数，表示0-9出现的概率
OUTPUT_NODE = 10

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def forward(x, train=True, regularizer=None):

    # shape表示张量的维度，为一个列表.[1,2]以及[1,2,3,4]等模式
    # 本例当中，表示的是一个4维张量
    def get_weight(shape, regularizer, name):
        # 随机生成参数，去掉偏离过大的正态分布
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.1, name=name))
        # 加入正则化
        if regularizer:
            tf.add_to_collection(
                'losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
        return w

    def get_bias(shape, name):
        b = tf.Variable(tf.zeros(shape, name=name))
        return b

    def conv2d(x, w, name=None):
        # 进行卷积计算
        # x表示输入的描述表示4阶张量，类似[1,2,2,3]，
        # 第一维参数表示一次性读取的图片数量
        # 第2、3维表示图片的分辨率，最后一维表示图片的通道数
        # w表示卷积核的表述，同样是4维张量
        # 第1、2维表示卷积核的分辨率，第3维表示图片的通道数
        # 第4维表示卷积核的个数
        return tf.nn.conv2d(
            x, w, strides=[1, CONV_STEP_SIZE, CONV_STEP_SIZE, 1],
            padding='SAME', name=name)

    def max_pool_conv(x, name=None):
        # x表示4阶张量，类似[1,2,2,3]，第一维参数表示一次性读取的图片数量
        # 第2、3维表示图片的分辨率，最后一维表示输入的通道数
        return tf.nn.max_pool(
            x, ksize=[1, POOL_KERNEL_SIZE, POOL_KERNEL_SIZE, 1],
            strides=[1,POOL_STEP_SIZE, POOL_STEP_SIZE, 1],
            padding='SAME', name=name)

    with tf.name_scope('forward_layar'):
        with tf.name_scope('conv1_layer'):
            with tf.name_scope('conv1_w'):
                # 第1层卷积的权重w
                conv1_w = get_weight([CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE,
                                      MNIST_IMAGE_CHANNELS, CONV1_KERNEL_NUM],
                                     regularizer, name='conv1_w')
                variable_summaries(conv1_w)
            with tf.name_scope('conv1_b'):
                # 第一层卷积的偏置b
                conv1_b = get_bias([CONV1_KERNEL_NUM], name='conv1_b')
                variable_summaries(conv1_b)
            with tf.name_scope('conv1'):
                # 进行第1层卷积运算
                conv1 = conv2d(x, conv1_w)
                # 使用偏置项，进行去噪，使用relu函数进行激活操作
                relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
        with tf.name_scope('pool1_layar'):
            # 对激活结果进行第1层最大池化操作
            pool1 = max_pool_conv(relu1)

        with tf.name_scope('conv2_layer'):
            # 第2层卷积的权重定义。矩阵当中的第3个参数表示
            # 第n层卷积核的通道数（深度），等于第n-1次卷积操作的卷积核个数
            with tf.name_scope('conv2_w'):
                conv2_w = get_weight([CONV2_KERNEL_SIZE, CONV2_KERNEL_SIZE,
                                    CONV1_KERNEL_NUM, CONV2_KERNEL_NUM],
                                    regularizer, name='conv2_w')
                variable_summaries(conv2_w)
            with tf.name_scope('conv2_b'):
                # 定义第2层卷积的偏置b
                conv2_b = get_bias([CONV2_KERNEL_NUM], name='conv2_b')
                variable_summaries(conv2_b)
            with tf.name_scope('conv2'):
                # 进行第2层卷积运算，第n层的卷积输入，其第1个参数，应当是
                # 第n-1次卷积操作（可以是池化之后的）的输出
                conv2 = conv2d(pool1, conv2_w)
                # 使用偏置项去噪，同时用relu激活函数进行激活操作
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))

        with tf.name_scope('pool2_layar'):
            # 对第2次卷积操作的结果进行最大池化操作，得到一个3维的张量
            # pool2便是卷积操作提取的所有特征值
            pool2 = max_pool_conv(relu2)

        with tf.name_scope('fc1_layer'):
            # 将3维张量pool2进行转置，变更为2维张量
            # 将pool2转变为列表
            pool_shape = pool2.get_shape().as_list()
            # 分别使用提取特征的长度，宽度和深度，
            # 计算全连接网络的神经元个数（所有特征点）
            # pool_shape[0]是batch_size
            # pool_shape[1]是特征长度
            # pool_shape[2]是特征宽度
            # pool_shape[3]是特征深度
            nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
            # 将pool2表示为batch_size行 × 特征点个数列的二维形状
            reshaped = tf.reshape(pool2, [-1, nodes])
            # 定义第1层全连接网络
            with tf.name_scope('fc1_w'):
                fc1_w = get_weight([nodes, FC_CONNECTED_SIZE],
                                   regularizer, name='fc1_w')
                variable_summaries(fc1_w)
            with tf.name_scope('fc1_b'):
                fc1_b = get_bias([FC_CONNECTED_SIZE], name='fc1_b')
                variable_summaries(fc1_b)

            fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
            # 如果是训练阶段，丢弃50%的特征点
            if train:
                with tf.name_scope('dropout'):
                    tf.summary.scalar('dropout_keep_probability', 0.5)
                    fc1 = tf.nn.dropout(fc1, 0.5)

        with tf.name_scope('fc2_layer'):
            with tf.name_scope('fc2_w'):
                fc2_w = get_weight([FC_CONNECTED_SIZE, OUTPUT_NODE],
                                   regularizer, name='fc2_w')
                variable_summaries(fc2_w)
            with tf.name_scope('fc2_b'):
                fc2_b = get_bias([OUTPUT_NODE], name='fc2_b')
                variable_summaries(fc2_b)

        y = tf.matmul(fc1, fc2_w) + fc2_b
    return y

def backward(mnist):
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 28 * 28], name='x_input')
        # 卷积的输入要求必须是4维张量
        x_image = tf.reshape(x, [-1,
            MNIST_IMAGE_SIZE,
            MNIST_IMAGE_SIZE,
            MNIST_IMAGE_CHANNELS])
        y_ = tf.placeholder(tf.float32, [
            None, OUTPUT_NODE],
            name='y_input')
        tf.summary.image(
            'input', x_image, OUTPUT_NODE)

    y = forward(
        x_image, train=True, regularizer=FLAGS.regularizer)

    global_step = tf.Variable(0, trainable=False)

    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)

    with tf.name_scope('learning_rate'):
        learning_rate = tf.train.exponential_decay(
            FLAGS.learning_rate_base,
            global_step,
            mnist.train.num_examples/FLAGS.batch_size,
            FLAGS.learning_rate_decay,
            staircase=True, name='learning_rate'
        )
        tf.summary.scalar('learning_rate', learning_rate)

    with tf.name_scope('train_step'):
        train_step = tf.train.GradientDescentOptimizer(
            learning_rate).minimize(loss, global_step=global_step)

    with tf.name_scope('exponential_moving_average'):
        ema = tf.train.ExponentialMovingAverage(
            MOVING_AVG_DECAY, global_step)
        ema_op = ema.apply(tf.trainable_variables())

    with tf.name_scope('training'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        for i in range(FLAGS.steps):
            xs, ys = mnist.train.next_batch(FLAGS.batch_size)
            _, loss_value, learning_rate_val, step, summary = sess.run(
                [train_op, loss, learning_rate, global_step, merged],
                feed_dict={x: xs, y_: ys})
            writer.add_summary(summary, i)

            if 0 == i % FLAGS.eval_step:
                fmt = 'After {:05d} steps, loss is {:.09f}, learning rate is {:.09f}'
                print(fmt.format(step, loss_value, learning_rate_val))
                saver.save(sess, os.path.join(FLAGS.model_dir, FLAGS.model_name),
                           global_step=global_step)

                fmt = 'And test accuracy rate is {:.09f}'
                accuracy_rate = sess.run(
                    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print(fmt.format(accuracy_rate))

        writer.close()


def main(unused_args):
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    tf.app.run()
