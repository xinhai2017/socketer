这部分代码块是基于inception v4网络结构，预加载训练好参数，进行自己数据集的分类任务：


*  环境：
   * dlib 19.16.0 
   * Keras 2.2.4
   * tenforflow-gpu 1.12.0
   * opencv 4.1.1
   
原始数据是mp4格式的，需要按每一帧的画面切下照片用于训练，执行capture_images_from_video文件夹下的multipool_capture_images_from_vedio.py：

         ```python multipool_capture_images_from_vedio.py```

然后运行文件夹extract_face_from_images下的py文件,切除人脸，图片的大小可以自己调整，调整这一行代码即可
```cropped = img[face.rect.top()-100:face.rect.bottom()+100, face.rect.left()-100:face.rect.right()+100]```：

         ```python multipool_crop_faces_save.py ```
    
接下就是训练模型部分了，

训练模型运行inception-v4-classification文件夹下的py文件，这个阶段需要在对应文件夹下补充一些预训练好的参数文件，文末附件：
         ```Python main.py```

测试模型运行test文件夹下的py文件：
         ```Python test_interface.py```

测试完模型之后运行ckpt_pb.py文件将模型转为.pb文件：
          ```Python ckpt_pb.py```

然后运行pb_to_tflite.py文件，将pb文件转为tflite文件，用于平台部署（安卓或者ios）：
        ```python pb_to_flite.py```
如果模型不需要量化注释掉这行代码就行了：```converter.post_training_quantize = True```

然后运行call_tflite_test_model.py文件在数据集上测试.tflite文件的性能：
       ```python call_tflite_test_model.py```

总结：
* 1.量化的模型大小是未量化模型大小的1/4，如果后期需要在移动端不是需要在类似Mobile Net等轻量级的模型上预测；
* 2.量化会对模型的预测精度有一定的影响，整体来看量化的预测精度没有未量化的进度高；

*  附件：
   * 预加载模型的参数：https://github.com/xinhai2017/models/tree/master/research/slim
   * 人脸定位预加载模型：http://dlib.net/files/
   
