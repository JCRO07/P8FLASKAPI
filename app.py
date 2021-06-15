
from flask import Flask, request, Response
from flask import jsonify
from skimage.transform import resize
import jsonpickle
import numpy as np
from numpy import asarray
import cv2
import PIL
from io import BytesIO
from PIL import Image


# manipulate files
import io
import os 
import json
import requests
import numpy as np
import os, glob, re, tempfile
import requests, json
import cv2
import albumentations as A
import tensorflow as tf
from tensorflow import keras
from albumentations import Compose
from albumentations import ToFloat
import segmentation_models as sm
sm.set_framework('tf.keras')
from azureml.core.model import Model
from azureml.core import Workspace, Datastore, Dataset

from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from azureml.core import Workspace
from azureml.core.model import Model

#map category to its color
def cat2color(arr_to_convert):
    prediction_color = {0:[0, 0, 0],            # void
                        1:[128, 64,128],       # flat
                        2:[150,100,100],         # construction
                        3:[220,220,  0],      # object
                        4:[107,142, 35],       # nature
                        5:[70, 130, 180],       # sky
                        6:[220, 20, 60],        # human
                        7:[119, 11, 32]}          # vehicule
    arr = np.zeros((*arr_to_convert.shape, 3))
    for k, v in prediction_color.items():
        arr[arr_to_convert==k] = v
    arr = arr.astype('uint8')
    return arr




ws = Workspace.from_config(path="config.json")
model = Model(ws, 'model8lp3', version=1)
# AzureML stuff to consider, checks for the registered models.
from azureml.core.model import Model 

app = Flask(__name__)

#model = tf.keras.models.load_model("model8lp3s.h5")
# Model
ARCHITECTURE = 'Unet'
BACKBONE = 'resnet34'
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 5
DIM = (256, 256)
N_CLASSES = 8
N_CHANNELS = 3
ACTIVATION = 'softmax'
LOSS = 'DICE' 
CLASS_WEIGHTS = None


#metrics = [sm.metrics.IOUScore(class_weights=CLASS_WEIGHTS), sm.metrics.FScore(class_weights=CLASS_WEIGHTS)]
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]
opt = keras.optimizers.Adam(learning_rate=LR)
    
# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
#loss = sm.losses.DiceLoss(class_weights=CLASS_WEIGHTS)
loss = tf.keras.losses.CategoricalCrossentropy()
#model = sm.Unet(BACKBONE, encoder_weights='imagenet', encoder_freeze=False, input_shape=(256, 256, 3), classes=N_CLASSES, activation=ACTIVATION)
#model.compile(optimizer=opt, loss=loss, metrics=metrics)   
model = sm.Unet(input_shape=(256,256,3), classes=N_CLASSES, activation=ACTIVATION)
model.compile(optimizer=opt, loss=loss, metrics=metrics)


# route http posts to this method
@app.route("/", methods=['POST'])
def test():
          # preprocessing function
    def get_preprocessing(preprocessing_fn):   
        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)     
    
    
    data = request.json
    image = data['image'] 
    data = asarray(image)  
    #temp_raw = resize(data, (256, 256))
          # Predict
    aug = Compose([
            #A.HorizontalFlip(p=0.5),              
            #A.OpticalDistortion(distort_limit=2, shift_limit=0.9, p=0.5),
        A.RandomContrast(limit=0.5, p=1)
        ]
    )
    
    augmented = aug(image=data)
    preprocess_input = sm.get_preprocessing(BACKBONE)
    preprocessing=get_preprocessing(preprocess_input)
    sample = preprocessing(image=augmented['image'])


    print('>>>>>>>>>>>>>>>>>>>>>> 5')

    prediction = model.predict(sample['image'].reshape(1, *sample['image'].shape))
        #prediction = model.predict(temp_raw)

    print('>>>>>>>>>>>>>>>>>>>>>> 6')

    prediction_array = prediction[0].argmax(2)
    
        
    # convert string of image data to uint8
    #nparr = np.fromstring(r.data, np.uint8)
    # decode image
    #img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
    # do some fancy processing here....
    #img = cv2.imencode('.png', img)
    
    ##response_pickled = jsonpickle.encode(prediction)

    ##return Response(response=response_pickled, status=200, mimetype="application/json")
    #response = {'status': 'ok', 'image': image}
    #return jsonify(response(json.dumps(prediction_array.tolist()), 200)
    
    return AMLResponse(json.dumps(prediction_array.tolist()), 200)

# start flask app
app.run(host="0.0.0.0", port=5000)
