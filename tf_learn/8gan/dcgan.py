"""
1.Data provider
   a.Image data
   b.random vector  a.b.一一对应
2. Build compute graph
      a.generator 从条件，生成图像
      b.discriminator 判断生成的图像是否是真实的
      c.DCGAN 汇总，定义损失函数，
          1.connect g and d
          2.define loss
          3.define train_op
3.training process
"""
import os
import tensorflow as tf
from tensorflow import logging
from tensorflow import gfile
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

output_dir = './local_run'
if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

# 参数设置
def get_default_params():
    return tf.contrib.training.HParams(
        z_dim = 100,                     # 随机向量长度
        init_conv_size = 4,              # 特征图初始大小
        g_channels = [128, 64, 32, 1],   # 反卷积，各层通道数（使用的统一卷积核大小和步长）
        d_channels = [32, 64, 128, 256], # 卷积核通道数，步长为2，图片尺寸减半，特征图通道数增加一倍
        batch_size = 128,
        learning_rate = 0.002,
        beta1 = 0.5,                      # adamoptimization 参数
        img_size = 32,                    # 生成目标图片尺寸，正方形的图片4->8->16->32
    )

hps = get_default_params()

# 数据生成器实现
class MnistData(object):
    def __init__(self, mnist_train, z_dim, img_size):
        """
        mnist_train: all data of train sets
        z_dim: the length of random vector
        img_size: the length of goal img_size
        """
        self._data = mnist_train
        self._example_num = len(self._data)
        # print(self._example_num)
        # 用正态分布生成随机向量
        self._z_data = np.random.standard_normal((self._example_num, z_dim))
        self._indicator = 0  # represent next_batch place
        self._resize_mnist_img(img_size)  # transform function
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(self._example_num)  # random shuffle arrange
        self._z_data = self._z_data[p]
        self._data = self._data[p]

    def _resize_mnist_img(self, img_size):
        """Resize mnist image to goal img_size
        how?
        1.numpy -> PIL img
        2.PIL img -> resize
        3.PIL img -> numpy
        """
        data = np.asarray(self._data * 255, np.uint8)  # mnist自动做了归一化
        # [example_num, 784] -> [example_num, 28, 28]
        data = data.reshape((self._example_num, 28, 28))
        new_data = []
        for i in range(self._example_num):
            img = data[i]
            img = Image.fromarray(img)
            img = img.resize((img_size, img_size))
            img = np.asarray(img)
            # 加上通道信息，卷积神经网络需要通道信息。
            img = img.reshape((img_size, img_size, 1))
            new_data.append(img)
        new_data = np.asarray(new_data, dtype=np.float32)  # transform list to vector
        new_data = new_data / 127.5 - 1  # normal
        # self._data: [num_example, img_size, img_size, 1]
        self._data = new_data  # 32 x 32

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > self._example_num:
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        assert end_indicator < self._example_num
        batch_data = self._data[self._indicator: end_indicator]
        batch_z = self._z_data[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_z

mnist_data = MnistData(mnist.train.images, hps.z_dim, hps.img_size)
# batch_data, batch_z = mnist_data.next_batch(5)
# print(batch_data)
# print(batch_data[0][16])
# print(batch_z)

# 生成器反卷积封装
def conv2d_transpose(inputs, out_channel, name, training, with_bn_relu=True):
    """wrapper of conv2d transpose.
    Arg:
    - inputs: 输入 [batch_size, img_size, img_size, channels]
    - out_channel: 输出通道
    - name: 名字
    - training: 是否train，用于bn层
    - with_bn_relu: 反卷积最后一层不需要做，bn和relu，只需要经过tanh生成-1到1之间的数即可
    """
    with tf.variable_scope(name):
        conv2d_trans = tf.layers.conv2d_transpose(inputs,
                                                  out_channel,
                                                  kernel_size = (5, 5),
                                                  strides=(2, 2),#一次反卷积，长宽增加一倍
                                                  padding='SAME')
        if with_bn_relu:
            bn = tf.layers.batch_normalization(conv2d_trans, training =training)
            return tf.nn.relu(bn)
        else:
            return conv2d_trans

# 判别器卷积封装
def conv2d(inputs, out_channel, name, training):
    """ wrapper of conv2d. """
    def leaky_relu(x, leak=0.2, name=""):  # slope is 0.2
        return tf.maximum(x, x * leak, name=name)

    with tf.variable_scope(name):
        conv2d_output = tf.layers.conv2d(inputs,
                                         out_channel,
                                         [5, 5],
                                         strides=(2, 2),
                                         activation=None,
                                         padding='SAME')
        bn = tf.layers.batch_normalization(conv2d_output,
                                           training=training
                                           )
        return leaky_relu(bn, name='outputs')

class Generator(object):
    """Generator of GAN."""
    def __init__(self, channels, init_conv_size):
        #私有成员变量
        self._channels = channels
        self._init_conv_size = init_conv_size
        self._reuse = False  # 是否重用,第二次调用时重用

    def __call__(self, inputs, training):  # 魔法方法，类的对象可以当作函数使用
        inputs = tf.convert_to_tensor(inputs)  # numpy 数组输入变换成tensor
        with tf.variable_scope('generator', reuse=self._reuse):
            """
            Random vector -> fc -> self._channels[0] * init_conv_size **2
            -> reshape -> [init_conv_size, init_conv_size, channels[0]]
            """
            with tf.variable_scope('inputs_conv'):
                fc = tf.layers.dense(inputs,
                                     self._channels[0] * self._init_conv_size * self._init_conv_size)
                conv0 = tf.reshape(fc,
                                   [-1,  # batch_size
                                    self._init_conv_size,
                                    self._init_conv_size,
                                    self._channels[0]])
                bn0 = tf.layers.batch_normalization(conv0, training=training)
                relu0 = tf.nn.relu(bn0)
            deconv_input_list = []
            deconv_input_list.append(relu0)
            for i in range(1, len(self._channels)):  # 做反卷积
                # 判断是不是最后一层
                with_bn_relu = (i != (len(self._channels) - 1))
                deconv_inputs = conv2d_transpose(deconv_input_list[-1],
                                                 self._channels[i],
                                                 "deconv-%d" % i,
                                                 training,
                                                 with_bn_relu)
                deconv_input_list.append(deconv_inputs)
                # print("deconv_inputs:",deconv_input)
            img_inputs = deconv_input_list[-1]
            # print("img_inputs:",img_inputs)
            with tf.variable_scope('generator_imgs'):
                # imgs value range: [-1, 1]
                imgs = tf.tanh(img_inputs, name='imgs')
        self._reuse = True
        # 保存生成器的所有参数,生成器和判别其是分别训练的
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                           scope='generator')
        return imgs

# 判别器
class Discriminator(object):
    """discriminator of gan"""
    def __init__(self, channels):
        self._channels = channels
        self._reuse = False  # 是否重用

    def __call__(self, inputs, training):
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        conv_inputs_list = []
        conv_inputs_list.append(inputs)
        with tf.variable_scope('discriminator', reuse=self._reuse):
            for i in range(len(self._channels)):
                conv_inputs = conv2d(conv_inputs_list[-1],
                                     self._channels[i],
                                     'conv-%d' % i,
                                     training)
                conv_inputs_list.append(conv_inputs)
                # print("conv_inputs:",conv_inputs)
            fc_inputs = conv_inputs_list[-1]
            # print("fc_inputs:",fc_inputs)
            with tf.variable_scope('fc'):
                flatten = tf.layers.flatten(fc_inputs)
                # 输出是不是真实的图片
                logits = tf.layers.dense(flatten, 2, name='logits')
        self._reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        return logits

# 连接D和G，定义loss train_op
class DCGAN(object):
    def __init__(self, hps):
        g_channels = hps.g_channels
        d_channels = hps.d_channels

        self._batch_size = hps.batch_size
        self._init_conv_size = hps.init_conv_size
        self._z_dim = hps.z_dim
        self._img_size = hps.img_size

        self._generator = Generator(g_channels, self._init_conv_size)
        self._discriminator = Discriminator(d_channels)

    def build(self):
        """builes the whole compute graph."""
        self._z_placeholder = tf.placeholder(tf.float32, (self._batch_size, self._z_dim)) # 随机向量
        self._img_placeholder = tf.placeholder(tf.float32, (self._batch_size,  # 真实图像的placeholder，训练判别器
                                                            self._img_size,
                                                            self._img_size,
                                                            1))
        generated_imgs = self._generator(self._z_placeholder, training=True)
        # 假图像的logits
        fake_img_logits = self._discriminator(generated_imgs, training=True)
        # 真图像的logits
        real_img_logits = self._discriminator(self._img_placeholder, training=True)

        # 生成器loss，假图片都判别为真
        loss_on_fake_to_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([self._batch_size], dtype=tf.int64),
            logits=fake_img_logits))

        # 判别器loss，假的判别成假的
        loss_on_fake_to_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros([self._batch_size], dtype=tf.int64),
            logits=fake_img_logits))
        # 判别器loss，真实的判别成真的
        loss_on_real_to_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.ones([self._batch_size], dtype=tf.int64),
            logits=real_img_logits))

        tf.add_to_collection('g_losses', loss_on_fake_to_real)  # 字典实现，add_to_collection
        tf.add_to_collection('d_losses', loss_on_fake_to_fake)
        tf.add_to_collection('d_losses', loss_on_real_to_real)

        loss = {
            'g': tf.add_n(tf.get_collection('g_losses'),  # 生成器损失函数
                          name='total_g_loss'),
            'd': tf.add_n(tf.get_collection('d_losses'),  # 判别其损失函数
                          name='total_d_loss')
        }
        return (self._z_placeholder,
                self._img_placeholder,
                generated_imgs,  # 生成的图像，打印中间结果用的到
                loss)

    # DCGAN训练算子实现
    def build_train_op(self, losses, learning_rate, beta1):
        """builds train op, should be called after build is called."""
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1,)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1)

        g_opt_op = g_opt.minimize(losses['g'], var_list=self._generator.variables)
        d_opt_op = d_opt.minimize(losses['d'], var_list=self._discriminator.variables)
        with tf.control_dependencies([g_opt_op, d_opt_op]):  # 交叉训练,依次执行
            return tf.no_op(name='train')

dcgan = DCGAN(hps)
z_placeholder, img_placeholder, generated_imgs, losses = dcgan.build()
train_op = dcgan.build_train_op(losses, hps.learning_rate, hps.beta1)

# 训练流程, 打印中间过程， 拼接生成的图片
def combine_imgs(batch_imgs, img_size, rows=8, cols=16):
    """combines small images in a batch into a big pic."""
    # batch_imgs: [batch_size, img_size, img_size, 1]
    result_big_img = []
    for i in range(rows):
        row_imgs = []
        for j in range(cols):
            # [img_size, img_size, 1]
            img = batch_imgs[cols * i + j]
            img = img.reshape((img_size, img_size))
            img = (img + 1) * 127.5
            row_imgs.append(img)
        row_imgs = np.hstack(row_imgs)  # 横向合并
        result_big_img.append(row_imgs)
    # result_big_img： [8*32, 16*32]
    result_big_img = np.vstack(result_big_img)  # 按列合并
    result_big_img = np.asarray(result_big_img, np.uint8)  # 变换数据类型
    result_big_img = Image.fromarray(result_big_img)  # 矩阵变成图像
    return result_big_img

init_op = tf.global_variables_initializer()
train_steps = 100000

with tf.Session() as sess:
    sess.run(init_op)
    for step in range(train_steps):
        batch_imgs, batch_z = mnist_data.next_batch(hps.batch_size)  # 获取数据
        fetches = [train_op, losses['g'], losses['d']]  # evaluate 变量
        should_sample = (step + 1) % 50 == 0  #
        if should_sample:
            fetches += [generated_imgs]
        output_values = sess.run(fetches,
                                 feed_dict={
                                     z_placeholder: batch_z,
                                     img_placeholder: batch_imgs})
        _, g_loss_val, d_loss_val = output_values[0:3]
        print('step: %4d, g_loss: %4.3f, d_loss: %4.3f' % (step+1, g_loss_val, d_loss_val))
        if should_sample:
            gen_imgs_val = output_values[3]
            gen_img_path = os.path.join(output_dir, '%05d-gen.jpg' % (step + 1))
            gt_img_path = os.path.join(output_dir, '%05d-gt.jpg' % (step + 1))
            gen_img = combine_imgs(gen_imgs_val, hps.img_size)
            gt_img = combine_imgs(batch_imgs, hps.img_size)
            gen_img.save(gen_img_path)
            gt_img.save(gt_img_path)
