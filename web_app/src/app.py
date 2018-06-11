import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from keras.models import load_model
from src.predict import model
from src.get_image_features import extract_features
from src.predict_utils import encoding_answer, encoding_question
from src.load import init
from keras.applications.vgg19 import VGG19
from keras.models import Model
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))



# load the models
model, vgg_model, graph = init(os.path.join(APP_ROOT, 'model\model_weights.h5'))
# model = load_model(os.path.join(APP_ROOT, 'model/my_model.h5'))
# model.load_weights(os.path.join(APP_ROOT, 'model/model_weights.h5'))
# print(model)
# base_model = VGG19(weights='imagenet')
# vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)


@app.route("/")
def index():
    return render_template("upload.html")


@app.route("/upload", methods = ['POST'])
def upload():
    with graph.as_default():
        target = os.path.join(APP_ROOT, 'images/')
        #print(target)

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("file"):
            #print(file)
            filename = file.filename
            destination = "/".join([target, filename])
            #print(filename)
            #print(destination)
            file.save(destination)

        # VQA System
        question = request.form['question']
        pred = model.predict([encoding_question(question), extract_features(destination, vgg_model)], batch_size=1)

        ##

        ##

        #ans = encoding_answer(pred.argmax(axis=1))
        ans = encoding_answer(question, pred)

        #return send_from_directory("images", filename, as_attachment=True)
        return render_template("complete.html", image_name = filename, question = question, answer = ans)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)



if __name__ == "__main__":
    app.run(port=4555, debug=True)

