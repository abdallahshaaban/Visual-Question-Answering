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
import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from Coattention import *
from Question_Hierarichy import *
import pandas as pd
from Reading_Images_Features  import fun
from Reading_Testing_Images_Features  import Testfun
from tkinter import *
import pandas as pd
from tkinter import messagebox


"""train"""
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\y_true_train.csv' , header=None)
y_true_train = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\Q_train.csv' , header=None)
Q_train = dataset.iloc[:, :].values
V_train=fun()

"""test"""
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\y_pred_test.csv' , header=None)
y_true_test = dataset.iloc[:, :].values
dataset=pd.read_csv('E:\\Prototype Dataset\\PreProcessed Text\\Q_test.csv' , header=None)
Q_test = dataset.iloc[:, :].values
V_test=Testfun() 


Q = Input(shape=(55,))
V = Input(shape=(512,49))

w_level , q_level = Ques_Hierarchy(Q,V)

qw,vw = Co_attention([w_level,V])
qs,vs = Co_attention([q_level,V])

w_att = keras.layers.Add()([qw,vw])
hw = Dense(512,activation='tanh')(w_att)
#hs = Reshape((512,),input_shape=(1,512))(hs)
hw = Dropout(0.5)(hw)

s_att = keras.layers.Add()([qs,vs])
hs = keras.layers.concatenate([s_att,hw],axis=-1)

hs = Dense(512,activation='tanh')(s_att)
hs = Reshape((512,),input_shape=(1,512))(hs)
hs = Dropout(0.5)(hs)
p =  Dense(10,activation='softmax')(hs)

print(p.shape)



model = Model(inputs=[Q,V], outputs=p)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit([Q_train,V_train], [y_true_train],epochs=100, batch_size=300,validation_data = ([Q_test,V_test],y_true_test ))

