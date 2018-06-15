from keras.models import load_model
import h5py
from src.predict import model
from keras.applications.vgg19 import VGG19
from keras.models import Model
import tensorflow as tf


def init(path):
    # load vqa model
    #model.load_weights("C:\\Users\\UPDATE\PycharmProjects\web_app\src\model\model_weights.h5")
    model.load_weights(path)
    print(path)
    print("vqa model loaded successfully")

    # load vgg model
    base_model = VGG19(weights='imagenet')
    vgg_model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    print("vgg model loaded successfully")

    graph = tf.get_default_graph()

    return model, vgg_model, graph

