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





# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications
print('running')
model_vgg16 = models.vgg16(pretrained=True)
for param in model_vgg16.parameters():
    param.requires_grad=True
print('running')
model_vgg16.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 512),
                      nn.ReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(512, 4))

MODEL_PATH = 'C:/Users/DELL/keras-flask-deploy-webapp-master/models/model_vgg16 (3).pt'
model_vgg16.load_state_dict(torch.load(MODEL_PATH))
print('running')
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
    return predicted_idx ,class_index[predicted_idx]

    #probs = torch.nn.functional.softmax(outputs, dim=1)
    #conf, classes =  torch.max(probs, 1)
    #print(probs)
    #return predicted_idx ,class_index[predicted_idx]
    #return conf.item(), class_index[classes.item()]






def predict(path):
    with open(path, 'rb') as fd:
        img_bytes = fd.read()
        print(img_bytes)
        class_name, class_id = get_prediction(image_bytes=img_bytes)
        # print(class_id,class_name)
        return class_name, class_id
        #return jsonify({'class_name': class_name, 'class_id':class_id})





print('running')


path='C:/Users/DELL/Downloads/IMG_20190419_135912.jpg'
print(predict(path))


