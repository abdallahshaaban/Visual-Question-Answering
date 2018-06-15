import numpy as np
import keras
import pandas as pd
from keras.models import Sequential,Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Concatenate,Reshape , Input ,LSTM, Dense, Dropout ,concatenate , Flatten ,GlobalMaxPooling1D
from keras.layers.convolutional import Conv2D, MaxPooling2D , Conv1D
from keras.layers.embeddings import Embedding
from keras.layers.core import Activation,Lambda
from keras.models import model_from_json
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from src.Coattention import *
from src.Question_Hierarichy import *
import pandas as pd
from tkinter import *
import pandas as pd
from tkinter import messagebox

Q = Input(shape=(55,))
V = Input(shape=(512,49))

w_level ,p_level, q_level = Ques_Hierarchy(Q,V)

qw,vw = Co_attention([w_level,V])
qp,vp = Co_attention([w_level,V])
qs,vs = Co_attention([q_level,V])

w_att = keras.layers.Add()([qw,vw])
hw = Dense(512,activation='tanh')(w_att)
hw = Dropout(0.5)(hw)

p_att = keras.layers.Add()([qp,vp])
hp = keras.layers.concatenate([p_att,hw],axis=-1)
hp = Dense(512,activation='tanh')(p_att)
hp = Dropout(0.5)(hp)

s_att = keras.layers.Add()([qs,vs])
hs = keras.layers.concatenate([s_att,hp],axis=-1)

hs = Dense(512,activation='tanh')(s_att)
hs = Reshape((512,),input_shape=(1,512))(hs)
hs = Dropout(0.5)(hs)
p =  Dense(430,activation='softmax')(hs)


Rms = keras.optimizers.RMSprop(lr=0.0004, rho=0.9, epsilon=None, decay=0.00000001)
model = Model(inputs=[Q,V], outputs=p)
