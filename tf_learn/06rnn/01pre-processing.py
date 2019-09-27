"""
分词
词语 -> id
 * matrix -> [|v|, embed_size]
 * 词语A -> id[5]
 * 词表
label -> id
"""
import os
import jieba

# input files
train_file = './datas/cnews.train.txt'
val_file = './datas/cnews.val.txt'
test_file = './datas/cnews.test.txt'

# output files
DIR = './datas'
OUTPUT = 'output_file'
output_dir = os.path.join(DIR, OUTPUT)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
seg_train_file = output_dir + './cnews.train.seg.txt'
seg_val_file = output_dir + './cnews.val.seg.txt'
seg_test_file = output_dir + './cnews.test.seg.txt'

vocab_file = output_dir + './cnews.vocab.txt'
category_file = output_dir + './cnews.category.txt'

# 分词工具使用
with open(val_file,'r',encoding='utf-8') as f:
    lines = f.readlines()

label, content = lines[0].strip('\r\n').split('\t') # 解析第一行数据
word_iter = jieba.cut(content)
print(content)
print('/'.join(word_iter))


def generate_seg_file(input_file, output_seg_file):
    """Segment the sentences in each line in input_file"""
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # 读所有行
    with open(output_seg_file, 'w', encoding='utf-8') as f:
        for line in lines:
            label, content = line.strip('\r\n').split('\t')
            word_iter = jieba.cut(content)
            word_content = ''
            for word in word_iter:
                word = word.strip(' ')
                if word != '':
                    word_content += word + ' '
            out_line = '%s\t%s\n' % (label, word_content.strip(' '))
            f.write(out_line)


generate_seg_file(train_file, seg_train_file)
generate_seg_file(val_file, seg_val_file)
generate_seg_file(test_file, seg_test_file)

# 构建词表   词语A -> id[5]
def generate_vocab_file(input_seg_file, output_vocab_file):
    with open(input_seg_file,'r', encoding='utf-8') as f:
        lines = f.readlines()
    word_dict = {} # 每个词语的频次信息
    # 统计词频
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        for word in content.split():
            word_dict.setdefault(word, 0)
            word_dict[word] += 1
        # 逆排序
        # [(word, frequency),......,()]
        sorted_word_dict = sorted(word_dict.items(),key=lambda d:d[1], reverse=True)
        with open(output_vocab_file,'w',encoding='utf-8') as f:
            f.write('<UNK>\t10000000\n') # 找不到词语的处理
            for item in sorted_word_dict:
                f.write('%s\t%d\n' % (item[0],item[1]))
generate_vocab_file(seg_train_file,vocab_file)


def generate_category_dict(input_file, category_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    category_dict = {}
    # 统计词频
    for line in lines:
        label, content = line.strip('\r\n').split('\t')
        category_dict.setdefault(label, 0)
        category_dict[label] += 1
    category_number = len(category_dict)
    with open(category_file, 'w', encoding='utf-8') as f:
        for category in category_dict:
            line = "%s\n" % category
            print('%s\t%d' % (category, category_dict[category]))
            f.write(line)


generate_category_dict(train_file, category_file)