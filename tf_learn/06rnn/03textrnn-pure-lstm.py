"""
谋篇布局
1.构建计算图 -----LSTM模型
embedding
LSTM
fc
train_op
2.训练流程代码
3.数据集封装
api: next_batch(batch_size)
词表封装：
api： sentence2id（text_sentence）：句子转换id
类别的封装：
api： category2id（text_category）.
"""
import tensorflow as tf
import os
import sys
import numpy as np
import math

# 申明打印日志
tf.logging.set_verbosity(tf.logging.INFO)

def get_default_params():
    return tf.contrib.training.HParams( # 返回一个对象
               num_embedding_size = 16,  #  32
               num_timesteps = 50,       # lstm的步长，一个sentence中有多少个词语（训练数据会以50做截断，测试数据支持变长） 600
               num_lstm_nodes = [32, 32],# 两层，每层32个lstm神经单元 [64, 64]
               num_lstm_layers = 2,      # 两层
               num_fc_nodes = 32,        # 训练快，值设小；结果好，值设大； 64
               batch_size = 100,
               clip_lstm_grads = 1.0,    # 防止梯度爆炸
               learning_rate = 0.001,
               num_word_threshold = 10   # 门限，超过该值纳入词表
    )

hps = get_default_params()

train_file = './datas/output_file/cnews.train.seg.txt'
val_file = './datas/output_file/cnews.val.seg.txt'
test_file = './datas/output_file/cnews.test.seg.txt'

vocab_file = './datas/output_file/cnews.vocab.txt'
category_file = './datas/output_file/cnews.category.txt'

output_folder = './datas/output_file/run_text_rnn'

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

# 词表封装
class Vocab:
    def __init__(self, filename, num_word_threshold):
        self._word_to_id = {}
        self._unk = -1
        self._num_word_threshold = num_word_threshold
        self._read_dict(filename)

    def _read_dict(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            word, frequency = line.strip('\r\n').split('\t')
            frequency = int(frequency)
            if frequency < self._num_word_threshold:
                continue
            idx = len(self._word_to_id)
            if word == '<UNK>':
                self._unk = idx
            self._word_to_id[word] = idx  # 因无重复词语，故直接插入

    def word_to_id(self, word):
        return self._word_to_id.get(word, self._unk)

    @property
    def unk(self):
        return self._unk

    def size(self):
        return len(self._word_to_id)

    def sentence_to_id(self, sentence):
        word_ids = [self.word_to_id(cur_word) for cur_word in sentence.split()]
        return word_ids

# 类别封装
class CategoryDict:
    def __init__(self, filename):
        self._category_to_id = {}
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            category = line.strip('\r\n')
            idx = len(self._category_to_id)
            self._category_to_id[category] = idx

    def size(self):
        return len(self._category_to_id)

    def category_to_id(self, category):
        if not category in self._category_to_id:
            raise Exception("%s is not our category list" % category)
        return self._category_to_id[category]

# 测试类实现
vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()

category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()

# test code
category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()

# 数据集封装
class TextDataSet:
    def __init__(self, filename, vocab, category_vocab, num_timesteps):
        self._vocab = vocab
        self._category_vocab = category_vocab
        self._num_timesteps = num_timesteps
        # matrix
        self._inputs = []
        # vector
        self._outputs = []
        self._indicator = 0
        self._parse_file(filename)

    def _parse_file(self, filename):
        tf.logging.info('Loading data from %s', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            id_label = self._category_vocab.category_to_id(label)
            id_words = self._vocab.sentence_to_id(content)
            id_words = id_words[0: self._num_timesteps]
            padding_num = self._num_timesteps - len(id_words)
            id_words = id_words + [self._vocab.unk for i in range(padding_num)]
            self._inputs.append(id_words)
            self._outputs.append(id_label)
        self._inputs = np.asarray(self._inputs, dtype=np.int32)
        self._outputs = np.asarray(self._outputs, dtype=np.int32)
        self._random_shuffle()

    def _random_shuffle(self):
        p = np.random.permutation(len(self._inputs))
        self._inputs = self._inputs[p]
        self._outputs = self._outputs[p]

    def next_batch(self, batch_size):
        end_indicator = self._indicator + batch_size
        if end_indicator > len(self._inputs):
            self._random_shuffle()
            self._indicator = 0
            end_indicator = batch_size
        if end_indicator > len(self._inputs):
            raise Exception("batch_size: %d is to large" % batch_size)

        batch_inputs = self._inputs[self._indicator: end_indicator]
        batch_outputs = self._outputs[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_inputs, batch_outputs

train_dataset = TextDataSet(train_file, vocab, category_vocab, hps.num_timesteps)
val_dataset = TextDataSet(val_file, vocab, category_vocab, hps.num_timesteps)
test_dataset = TextDataSet(test_file, vocab, category_vocab, hps.num_timesteps)

# 计算图构建
def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')  # dropout中用，避免全连接层过拟合，丢掉：1-keepprob 的值

    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)  # 记录训练到了哪一步

    # embedding层
    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)  # 初始化，均匀分布-1到1之间
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)
        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    # lstm层
    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_lstm_nodes[-1]) / 3.0
    lstm_init = tf.random_uniform_initializer(-scale, scale)

    def _generate_params_for_lstm_cell(x_size, h_size, bias_size):
        """generates parameters for pure lstm implementation."""
        x_w = tf.get_variable('x_weights', x_size)
        h_w = tf.get_variable('h_weights', h_size)
        b = tf.get_variable('biases', bias_size, initializer=tf.constant_initializer(0.0))
        return x_w, h_w, b

    with tf.variable_scope('lstm_nn', initializer=lstm_init):
        """
        cells = []
        for i in range(hps.num_lstm_layers):
            cell = tf.contrib.rnn.BasicLSTMCell(hps.num_fc_nodes[i], state_is_tuple=True)
            cell = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob=keep_prob)
            cells.append(cell)
        cell = tf.contrib.rnn.MultiRNNCell(cells) # 多层lstm的封装，多层当作单层操作

        initial_state = cell.zero_state(batch_size,tf.float32) # 中间状态的初始化
        # rnn_outputs: [batch_size, num_timesteps, lstm_outputs[-1]]
        rnn_outputs,_ = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state = initial_state) # _ 表示中间状态
        last = rnn_outputs[:,-1,:]
        """
        with tf.variable_scope('inputs'):
            ix, ih, ib = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]])

        with tf.variable_scope('outputs'):
            ox, oh, ob = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]])

        with tf.variable_scope('forget'):
            fx, fh, fb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]])

        with tf.variable_scope('memory'):
            cx, ch, cb = _generate_params_for_lstm_cell(
                x_size=[hps.num_embedding_size, hps.num_lstm_nodes[0]],
                h_size=[hps.num_lstm_nodes[0], hps.num_lstm_nodes[0]],
                bias_size=[1, hps.num_lstm_nodes[0]])

        state = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes[0]]),  # 中间隐含状态，
                            trainable=False)

        h = tf.Variable(tf.zeros([batch_size, hps.num_lstm_nodes[0]]),
                        trainable=False)

        for i in range(num_timesteps):
            # [batch_size, 1, embed_size]
            embed_input = embed_inputs[:, i, :]
            embed_input = tf.reshape(embed_input, [batch_size, hps.num_embedding_size])
            forget_gate = tf.sigmoid(
                tf.matmul(embed_input, fx) + tf.matmul(h, fh) + fb)
            input_gate = tf.sigmoid(
                tf.matmul(embed_input, ix) + tf.matmul(h, ih) + ib)
            output_gate = tf.sigmoid(
                tf.matmul(embed_input, ox) + tf.matmul(h, oh) + ob)
            mid_state = tf.tanh(
                tf.matmul(embed_input, cx) + tf.matmul(h, ch) + cb)
            state = mid_state * input_gate + state * forget_gate
            h = output_gate * tf.tanh(state)
        last = h

    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(last,
                              hps.num_fc_nodes,
                              activation=tf.nn.relu,
                              name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout,
                                 num_classes,
                                 name='fc2')

    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        # [0, 1, 5, 4, 2] -> argmax: 2
        y_pred = tf.argmax(tf.nn.softmax(logits),
                           1,
                           output_type=tf.int32)
        #         y_pred = tf.cast(y_pred,tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        tvars = tf.trainable_variables()
        for var in tvars:
            tf.logging.info('variable name: %s' % (var.name))
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), hps.clip_lstm_grads)  # 截断梯度
        optimizer = tf.train.AdamOptimizer(hps.learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)

    return ((inputs, outputs, keep_prob),
            (loss, accuracy),
            (train_op, global_step))


# 训练流程实现
placeholders, metrics, others = create_model(hps, vocab_size, num_classes)

inputs, outputs, keep_prob = placeholders
loss, accuracy = metrics
train_op, global_step = others
# 训练流程实现
init_op = tf.global_variables_initializer()
train_keep_prob_value = 0.8
test_keep_prob_value = 1.0

num_train_steps = 10000

# Train: 99.7%
# Train: 92.7%
# Val: 93.2%
with tf.Session() as sess:
    sess.run(init_op)
    for i in range(num_train_steps):
        batch_inputs, batch_labels = train_dataset.next_batch(hps.batch_size)
        outputs_val = sess.run([loss, accuracy, train_op, global_step],
                               feed_dict={
                                 inputs: batch_inputs,
                                 outputs: batch_labels,
                                 keep_prob: train_keep_prob_value,
        })
        loss_val, accuracy_val, _, global_step_val = outputs_val
        if global_step_val % 20 == 0:
            tf.logging.info('Step: %5d, loss: %3.3f, accuracy: %3.3f' % (global_step_val, loss_val, accuracy_val))


