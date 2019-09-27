import tensorflow as tf
import os
import sys
import numpy as np
import math

# 申明打印日志
tf.logging.set_verbosity(tf.logging.INFO)

def get_default_params():
    return tf.contrib.training.HParams(
               num_embedding_size = 16,
               num_timesteps = 50, # 600
               num_filters = 128,  # 256
               num_kernel_size = 3,
               num_fc_nodes = 32,
               batch_size = 100,
               learning_rate = 0.001,
               num_word_threshold = 10
    )

hps = get_default_params()

train_file = './datas/output_file/cnews.train.seg.txt'
val_file = './datas/output_file/cnews.test.seg.txt'
test_file = './datas/output_file/cnews.test.seg.txt'
vocab_file = './datas/output_file/cnews.vocab.txt'
category_file = './datas/output_file/cnews.category.txt'
output_file = './datas/output_file/run_text_rnn'

if not os.path.exists(output_file):
    os.mkdir(output_file)


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
            self._word_to_id[word] = idx

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
            raise Execption("%s is not our category list" % category_name)
        return self._category_to_id[category]


vocab = Vocab(vocab_file, hps.num_word_threshold)
vocab_size = vocab.size()
# vocab_size = str(vocab_size)
tf.logging.info('vocab_size: %d' % vocab_size)
# print(type(vocab_size))
category_vocab = CategoryDict(category_file)
num_classes = category_vocab.size()
tf.logging.info('num_classes: %d' % num_classes)
'''test code
test_str = '的 在 了 是'
print(vocab.sentence_to_id(test_str))
'''
# test code
category_vocab = CategoryDict(category_file)
tf.logging.info('num_classes: %d' % category_vocab.size())
test_str = '时尚'
tf.logging.info('label: %s, id: %d' % (test_str, category_vocab.category_to_id(test_str)))


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

print(train_dataset.next_batch(2))
print(val_dataset.next_batch(2))
print(test_dataset.next_batch(2))


# 计算图构建
def create_model(hps, vocab_size, num_classes):
    num_timesteps = hps.num_timesteps
    batch_size = hps.batch_size

    inputs = tf.placeholder(tf.int32, (batch_size, num_timesteps))
    outputs = tf.placeholder(tf.int32, (batch_size,))
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    global_step = tf.Variable(tf.zeros([], tf.int64), name='global_step', trainable=False)

    embedding_initializer = tf.random_uniform_initializer(-1.0, 1.0)
    with tf.variable_scope('embedding', initializer=embedding_initializer):
        embeddings = tf.get_variable('embedding', [vocab_size, hps.num_embedding_size], tf.float32)

        # [1, 10, 7] -> [embeddings[1], embeddings[10], embeddings[7]]
        embed_inputs = tf.nn.embedding_lookup(embeddings, inputs)

    scale = 1.0 / math.sqrt(hps.num_embedding_size + hps.num_filters) / 3.0
    cnn_init = tf.random_uniform_initializer(-scale, scale)
    with tf.variable_scope('cnn', initializer=cnn_init):
        # embed_inputs: [batch_size, timesteps, enbed_size]
        # conv1d: [batch_size, timesteps, num_filters]
        conv1d = tf.layers.conv1d(
            embed_inputs,
            hps.num_filters,
            hps.num_kernel_size,
            activation=tf.nn.relu,
        )
        global_maxpooling = tf.reduce_max(conv1d, axis=[1])

    fc_init = tf.uniform_unit_scaling_initializer(factor=1.0)
    with tf.variable_scope('fc', initializer=fc_init):
        fc1 = tf.layers.dense(global_maxpooling, hps.num_fc_nodes, name='fc1')
        fc1_dropout = tf.contrib.layers.dropout(fc1, keep_prob)
        logits = tf.layers.dense(fc1_dropout, num_classes, name='fc2')

    with tf.name_scope('metrics'):
        softmax_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=outputs)
        loss = tf.reduce_mean(softmax_loss)
        y_pred = tf.argmax(tf.nn.softmax(logits), 1)
        y_pred = tf.cast(y_pred, tf.int32)
        correct_pred = tf.equal(outputs, y_pred)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(hps.learning_rate).minimize(loss, global_step=global_step)
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
        outputs_val = sess.run([loss, accuracy, train_op, global_step],feed_dict={
                                 inputs: batch_inputs,
                                 outputs: batch_labels,
                                 keep_prob: train_keep_prob_value
        })
        loss_val, accuracy_val, _, global_step_val = outputs_val
        if global_step_val % 20 == 0:
            tf.logging.info('Step: %5d, loss: %3.3f, accuracy: %3.3f' % (global_step_val, loss_val, accuracy_val))