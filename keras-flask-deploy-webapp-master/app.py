import os
import sys
import json
import io

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

# Some utilites
import numpy as np
from PIL import Image


# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications

model_vgg16 = models.vgg16(pretrained=True)
for param in model_vgg16.parameters():
    param.requires_grad=True
model_vgg16.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(512, 4))

MODEL_PATH = 'C:/Users/DELL/keras-flask-deploy-webapp-master/models/model_vgg16 (3).pt'
model_vgg16.load_state_dict(torch.load(MODEL_PATH))

print('Model loaded. Check http://127.0.0.1:5000/')

class_index={0:'BrownSpot', 1: 'Healthy', 2:'Hispa', 3: 'LeafBlast'}

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    tensor = transform_image(image_bytes=image_bytes)
    outputs = model_vgg16.forward(tensor)
    _, y_hat = torch.max(outputs,1)
    y_hat=y_hat.numpy()
    #print(outputs)
    predicted_idx = y_hat.item()
    return predicted_idx, class_index[predicted_idx]

    #probs = torch.nn.functional.softmax(outputs, dim=1)
    #conf, classes =  torch.max(probs, 1)
    #print(probs)
    #return predicted_idx ,class_index[predicted_idx]
    #return conf.item(), class_index[classes.item()]





@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        img = base64_to_pil(request.json)

        imgByteArr = io.BytesIO()
        # image.save expects a file as a argument, passing a bytes io ins
        img.save(imgByteArr, format= img.format)
        # Turn the BytesIO object back into a bytes object
        imgByteArr = imgByteArr.getvalue()
        print(imgByteArr)
        #file = request.files['file']
        #img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=imgByteArr)
        return jsonify(result=class_name, probability=1)


if __name__ == '__main__':
    app.run()

    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
