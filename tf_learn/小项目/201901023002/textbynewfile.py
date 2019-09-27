import tensorflow
import tensorflow as tf
import numpy as np
from manage import models
import matplotlib.pyplot as plt


radar_dict = {0: '一', 1: '丁', 2: '七', 3: '万', 4: '丈', 5: '三', 6: '上', 7: '下', 8: '不', 9: '与', 10: '专',
                      11: '两', 12: '义'}

# 从训练集中选取一张图片
def get_one_image(train):
    image_raw_data = tf.gfile.FastGFile(train, 'rb').read()
    with tf.Session() as sess:
        img_data = tf.image.decode_png(image_raw_data)
        image = tf.image.adjust_contrast(img_data, -255)
        image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)
        # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
        image = tf.image.per_image_standardization(image)
        # plt.imshow(image.eval())
        # plt.show()
        image = np.array(image.eval())
        return image


def numeric_return(oldarr):
    prediction = list(oldarr)
    arrs=[]
    num_arrss = []
    for x in range(10):
        max_index = np.argmax(prediction[0])
        # print(radar_dict[max_index])
        arrs.append(radar_dict[max_index])
        y = '{:.10f}'.format(prediction[0][max_index])
        num_arrss.append(y)
        prediction[0][max_index] = 0
    return np.array(arrs),np.array(num_arrss)


def evaluate_one_image(train):
    # 获取图片路径集和标签集
    image_array = get_one_image(train)
    with tf.Graph().as_default():
        BATCH_SIZE = 1  # 因为只读取一副图片 所以batch 设置为1
        N_CLASSES = 13  # 2个输出神经元，［1，0］ 或者 ［0，1］猫和狗的概率

        # 转化图片格式
        image = tf.cast(image_array, tf.float32)
        # 图片标准化
        image = tf.image.per_image_standardization(image)
        # 图片原来是三维的 [208, 208, 3] 重新定义图片形状 改为一个4D  四维的 tensor
        image = tf.reshape(image, [1, 64, 64, 3])
        logit = models.inference(image, BATCH_SIZE, N_CLASSES, 1.0)
        # 因为 inference 的返回没有用激活函数，所以在这里对结果用softmax 激活
        logit = tf.nn.softmax(logit)
        # 用最原始的输入数据的方式向模型输入数据 placeholder
        x = tf.placeholder(tf.float32, shape=[64, 64, 3])
        # 我门存放模型的路径
        logs_train_dir = 'manage/train/'
        # 定义saver
        saver = tf.train.Saver()
        with tf.Session() as sess:
            # print("从指定的路径中加载模型。。。。")
            # 将模型加载到sess 中
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                # print('模型加载成功, 训练的步数为 %s' % global_step)
            # else:
            # print('模型加载失败，，，文件没有找到')
            # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_array})
            return numeric_return(prediction)
            # 获取输出结果中最大概率的索引

            # print(prediction)
            # max_index = np.argmax(prediction)
            # print(radar_dict[max_index])
            # print('测试结果：%s 。。。。/n', radar_dict[max_index])
            # pathnew = Dpath + '\\' + radar_dict[max_index] + '\\'
            # if not (os.path.isdir(pathnew)):
            #     os.makedirs(pathnew);
            # newps = pathnew + os.path.basename(train);
            # shutil.copy(train, newps)  # 复制图片到指定位置

            # if max_index == 0:
            #     print('猫的概率 %.6f' % prediction[:, 0])
            # else:
            #     print('狗的概率 %.6f' % prediction[:, 1])
            # 测试


#传入图片路径获取概率
def textbyimage(path):
    return evaluate_one_image(path)


# strarr,numarr=textbyimage(r'C:\Users\admin\Desktop/20190123101344.png')
# print(strarr)
# print(numarr)




