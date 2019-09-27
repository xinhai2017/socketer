from flask import Flask, jsonify, render_template, request,json

import cv2
from PIL import Image as ImagePIL, ImageFont, ImageDraw
from PIL import Image

im = ImagePIL.open('../image/text.png')  #读取图片bgr 格式<class 'PIL.JpegImagePlugin.JpegImageFile'>
print(im)
print(type(im))
im = cv2.imread('../image/text.png')   #读取图片rgb 格式<class 'numpy.ndarray'>
image = Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))  #格式转换，bgr转rgb
image.save('../image/text.png',dpi=(300.0,300.0))    #调整图像的分辨率为300,dpi可以更改
