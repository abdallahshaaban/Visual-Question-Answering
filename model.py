import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Reshape
from keras.layers.convolutional import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D
from keras.layers.embeddings import Embedding
from prepare_data import QuestionsData
from train import Train
 
vocab_size = 1000
max_length = 15
batchsize = 300
d = 512
N = 14
k = 1
 
 
Q = tf.placeholder(tf.float32, [batchsize, 15])
V = tf.placeholder(tf.float32, [batchsize, d, N * N])
 
 
def question_hierarchy(Q):
    # Word Level
    word_level = Embedding(vocab_size, 512, input_length=max_length)(Q)
    word = Reshape((15, 512, 1), input_shape=(15, 512))(word_level) # (300, 15, 512, 1)
 
    # Phrase Level
    uni_gram = Conv2D(1, (1, 512), activation='tanh', input_shape=(15, 512, 1), padding='same')(word)
    bi_gram = Conv2D(1, (2, 512), activation='tanh', input_shape=(15, 512, 1), padding='same')(word)
    tri_gram = Conv2D(1, (3, 512), activation='tanh', input_shape=(15, 512, 1), padding='same')(word)
    phrase = keras.layers.concatenate([uni_gram, bi_gram, tri_gram]) # (300, 15, 512, 3)
    phrase_level = MaxPooling2D(pool_size=(1, 3), data_format='channels_first')(phrase) # (300, 15, 512, 1)
    phrase_level = Reshape((15, 512), input_shape=(15, 512, 1))(phrase_level)
 
    # Question Level
    question_level = LSTM(512, input_shape=(15, 512))(phrase_level) # (300, 512)
    question_level = tf.reshape(question_level, [batchsize, 1, 512])
 
    return word_level, phrase_level, question_level
 
 
def parallel_co_attention(Qw, V):
    # Step 1
    Wb = tf.Variable(tf.random_uniform((batchsize,d, d), 0, 1))
    C = tf.matmul(Qw, Wb)
    C = tf.matmul(C, V)
    C = tf.nn.tanh(C)  # shape(batchs, T, N) = (300, 15, 196)
 
    # Step2
    Wv = tf.Variable(tf.random_uniform((batchsize, k, d), 0, 1))
    Wq = tf.Variable(tf.random_uniform((batchsize, k, d), 0, 1))
 
    QwT = tf.transpose(Qw, perm=[0, 2, 1])
    Hv = tf.add(tf.matmul(Wv, V), tf.matmul(tf.matmul(Wq, QwT), C))
    Hv = tf.nn.tanh(Hv)  # shape(K, N)
 
    CT = tf.transpose(C, perm=[0, 2, 1])
    Hq = tf.add(tf.matmul(Wq, QwT), tf.matmul(tf.matmul(Wv, V), CT))
    Hq = tf.nn.tanh(Hq)  # shape(K, T)
 
    #Step 3
    Whv = tf.Variable(tf.random_uniform((batchsize, 1, k), 0, 1))
    Whq = tf.Variable(tf.random_uniform((batchsize, 1, k), 0, 1))
 
    av = tf.matmul(Whv, Hv)
    av = tf.nn.softmax(av)  # shape(K, N)
    aq = tf.matmul(Whq, Hq)
    aq = tf.nn.softmax(aq)  # shape(K, T)
 
    #Step 4
    v = tf.zeros([batchsize, k, d])
    Vt = tf.transpose(V, perm=[0, 2, 1])
    for i in range(N):
        v = tf.add(tf.matmul(av[:batchsize, :k, i:i+1], Vt[:batchsize, i:i+1, :d]), v)
 
    q = tf.zeros([batchsize, k, d])
    seq_len = Qw.shape[1]
    for i in range(seq_len):
        q = tf.add(tf.matmul(aq[:batchsize, :k, i:i+1], Qw[:300, i:i+1, :d]), q)
    print("av: ", av.shape)
    print("aq: ", aq.shape)
    return v, q
 
 
def predict_answer(qw, qp, qs, vw, vp, vs):
    Ww = tf.Variable(tf.random_normal([batchsize, 1, 1]))
    hw = tf.matmul(Ww, tf.add(qw, vw))
    hw = tf.nn.tanh(hw)
    print(hw.shape)
 
    Wp = tf.Variable(tf.random_normal([batchsize, 1, 1]))
    hp = tf.matmul(Wp, tf.concat([tf.add(qp, vp), hw], axis=2))
    hp = tf.nn.tanh(hp)
    print(hp.shape)
 
    Ws = tf.Variable(tf.random_normal([batchsize, 1, 1]))
    hs = tf.matmul(Ws, tf.concat([tf.add(qs, vs), hp], axis=2))
    hs = tf.nn.tanh(hs)
    print(hs.shape)
 
    Wh = tf.Variable(tf.random_normal([batchsize, 1, 1]))
    p = tf.matmul(Wh, hs)
    print(p.shape)
 
 
 
word_level, phrase_level, question_level = question_hierarchy(Q)
vw, qw = parallel_co_attention(word_level, V)
vp, qp = parallel_co_attention(phrase_level, V)
vs, qs = parallel_co_attention(question_level, V)
# x = tf.concat([v, q], axis=2)
print("vw: ", vw.shape, "qw: ", qw.shape)
print("vp: ", vp.shape, "qp: ", qp.shape)
print("vs: ", vs.shape, "qs: ", qs.shape)
# print("x: ", x.shape)
predict_answer(qw, qp, qp, vw, vp, vp)