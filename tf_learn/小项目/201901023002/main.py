"""
https://download.csdn.net/download/qq_35554617/10938281
"""

from flask import Flask, jsonify, render_template, request,json
from PIL import Image
import cv2

import textbynewfile

app = Flask(__name__)


# FLASK_ENV=development

# @app.route('/')
# def index():
#     return '<h1>Hello World!</h1>'

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/api/mnist', methods=['POST'])
def mnist():
    # array=np.array(request.json, dtype=np.uint8).reshape(3, [64,64])
    # data = np.reshape(np.array(request.json, dtype=np.uint8), (64, 64))
    # new_im = Image.fromarray(data)
    # new_im.show()
    # json_Data = request.get_json()
    # num1=request.data
    # num2=request.form
    # # base64Data=json_Data['base64Data']
    # # fp = open("test.png", 'w')
    # # fp.writelines(base64Data)
    # # fp.close()
    # # nums1 = request.form.get("value", type=str, default=None)
    # # nums2 = request.values.get("value")
    # rs=request.json
    filesimage= request.files.get('image')
    if filesimage is None:
        return jsonify(results=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    else:
        filesimage.save("image/text.png")
        im = cv2.imread('image/text.png')  # 读取图片rgb 格式<class 'numpy.ndarray'>
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # 格式转换，bgr转rgb
        image.save('image/text.png', dpi=(300.0, 300.0))  # 调整图像的分辨率为300,dpi可以更改
        strarr, numarr = textbynewfile.textbyimage(r'image/text.png')
        # ss = data['value']
        return jsonify(results=[list(strarr), list(numarr)])

if __name__ == '__main__':
    app.run(debug=True, port='8000', host='127.0.0.1')
