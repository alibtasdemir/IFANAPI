from flask import Flask, request, jsonify
import base64
from flask_cors import CORS
import os
import cv2
from codecs import encode
from single_test import runTest

app = Flask(__name__)
cors = CORS(app)


def encode_image(path, ext='jpg'):
    response_img = cv2.imread(path)
    _, response_img = cv2.imencode('.'+ext, response_img)
    response_img = base64.b64encode(response_img)
    return response_img.decode()


@app.route('/image', methods=['GET', 'POST'])
def image():
    bytes_img = encode(request.json['imageEnc'], 'utf-8')
    img = base64.decodebytes(bytes_img)
    user_image = os.path.join('../input_images', 'image.jpg')
    with open(user_image, 'wb') as out:
        out.write(img)
    if request.method == 'POST':
        runtime = runTest()
        response_img_path = os.path.join('../result_images', 'image.jpg')
        response_img = encode_image(response_img_path, ext='jpg')
        response_json = {
            'response': str(response_img),
            'runtime': runtime
        }
        return jsonify(response_json)
    else:
        return "Get the image"


@app.route('/data')
def send_dummy():
    return {
        'Name': "Ali Baran Tasdemir",
        'Age': "24"
    }


@app.route('/process', methods=['POST', 'GET'])
def process_click():
    proc = request.json["count"] ** 3
    retVal = {
        'proc': proc,
        'count': request.json["count"]
    }

    if request.method == 'GET':
        return "The result is " + str(proc)
    else:
        return jsonify(retVal)


@app.route('/', methods=['GET'])
def handle_call():  # put application's code here
    return 'Successfully Connected'


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
