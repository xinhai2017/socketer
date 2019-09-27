import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image
import PIL.ImageOps

image_raw_data = tf.gfile.FastGFile(r'../image/2222.png','rgb').read()

with tf.Session() as sess:
      img_data = tf.image.decode_png(image_raw_data)
      # image = tf.image.resize_images(
      #     img_data,
      #     [64,64],
      #     method=1,
      #     align_corners=True
      # )
      # resized = np.asarray(image.eval(), dtype='uint8')  # 变为uint8才能显示
      # plt.imshow(resized)
      image = tf.image.adjust_contrast(img_data, -255)
      image = tf.image.resize_image_with_crop_or_pad(image, 64, 64)

      # 对数据进行标准化,标准化，就是减去它的均值，除以他的方差
      image = tf.image.per_image_standardization(image)
      # image = tf.image.adjust_contrast(image, 20)
      #
      plt.imshow(image.eval())
      plt.show()

      # adjusted = tf.image.per_image_standardization(image)
      # plt.imshow(adjusted.eval())
      # plt.show()
     # plt.imshow(img_data.eval())
     # plt.show()
     #
     # # 将图片的亮度-0.5。
     # adjusted = tf.image.adjust_brightness(img_data, -0.5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     #
     # # 将图片的亮度0.5
     # adjusted = tf.image.adjust_brightness(img_data, 0.5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     # # 在[-max_delta, max_delta)的范围随机调整图片的亮度。
     # adjusted = tf.image.random_brightness(img_data, max_delta=0.5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     # # 将图片的对比度-5
     # adjusted = tf.image.adjust_contrast(img_data, -5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     # # 将图片的对比度+5
     # adjusted = tf.image.adjust_contrast(img_data, 5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     # # 在[lower, upper]的范围随机调整图的对比度。
     # adjusted = tf.image.random_contrast(img_data, 0.1, 0.6)
     # plt.imshow(adjusted.eval())
     # plt.show()
     #
     # #调整图片的色相
     # adjusted = tf.image.adjust_hue(img_data, 0.1)
     # plt.imshow(adjusted.eval())
     # plt.show()
     #
     # # 在[-max_delta, max_delta]的范围随机调整图片的色相。max_delta的取值在[0, 0.5]之间。
#      # adjusted = tf.image.random_hue(img_data, 0.5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     #
     #
     # # 将图片的饱和度-5。
     # adjusted = tf.image.adjust_saturation(img_data, -5)
     # plt.imshow(adjusted.eval())
     # plt.show()
     #
     #
     # # 在[lower, upper]的范围随机调整图的饱和度。
     # adjusted = tf.image.random_saturation(img_data, 0, 5)
     #
     # # 将代表一张图片的三维矩阵中的数字均值变为0，方差变为1。
     # adjusted = tf.image.per_image_standardization(img_data)

     # image = tf.image.decode_jpeg(image_contents, channels=3)
     # image = tf.image.decode_png(img_data, channels=3)
     # 将图片以图片中心进行裁剪或者扩充为 指定的image_W，image_H


    # 拉伸压缩图片
    # image = tf.image.resize_images(
      #     img_data,
      #     [64,64],
      #     method=1,
      #     align_corners=True
      # )
      # resized = np.asarray(image.eval(), dtype='uint8')  # 变为uint8才能显示
      # plt.imshow(resized)
# ---------------------
# 作者：chaibubble
# 来源：CSDN
# 原文：https://blog.csdn.net/chaipp0607/article/details/73089910
# 版权声明：本文为博主原创文章，转载请附上博文链接！