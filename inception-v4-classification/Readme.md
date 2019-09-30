# 这部分代码块是基于inception v4网络结构，预加载训练好参数，进行自己数据集的分类任务：


## 环境：
   * dlib 19.16.0 
   * Keras 2.2.4
   * tenforflow-gpu 1.12.0
   * opencv 4.1.1
   
## 原始数据是mp4格式的，需要按每一帧的画面切下照片用于训练，执行capture_images_from_video文件夹下的multipool_capture_images_from_vedio.py：

'''python
   python multipool_capture_images_from_vedio.py
'''
    
