
from werkzeug.utils import secure_filename
from flask import Flask, request
from flask import  send_file
import numpy as np
from predict import model
from predict_Utils import encoding_question , encoding_answer
from Extract_Features_of_Images import ExtractFeatures
model.load_weights("C:\\Users\\Lenovo-PC\\Desktop\\Visual-Question-Answering\\data\\model_weights.h5")

app = Flask(__name__)

@app.route('/upload_img',methods=['GET','POST'])
def upload_img():
    if request.method == 'POST':

        question=request.form['ques']
        # save the image then process on it
        img=request.files['pic']
        img.save(secure_filename(img.filename))
        #####
        # YOUR Processing Here
        pred = model.predict([encoding_question(question),ExtractFeatures(img.filename)],batch_size=1)
        ans = encoding_answer(pred.argmax(axis=1))
        #####

        # return your result
        Ans=ans
        return Ans


    else:
        return "Y U NO USE POST?"

if __name__ == "__main__":
    app.run(host='192.168.43.36')
