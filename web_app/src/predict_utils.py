import numpy as np
import tensorflow as tf
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
import csv
import numpy as np


def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"'s", "is", text)
    text = re.sub(r"[-()\"#/@;:<>${}+=~'|.?,!&%^*]", "", text)
    return text



def encoding_question(text):
    text = clean_text(text)
    ques_id = np.load('ques_id.npy').item()
    encoded_ques = []

    encoded_ques = [one_hot(text, 1000)]
    encoded_ques = pad_sequences(encoded_ques, maxlen=55, padding='post')

    encoded_ques = np.array(encoded_ques)
    encoded_ques = np.reshape(encoded_ques, [1, 55])

    return encoded_ques


def encoding_answer(question, pred):

    colors = ['red', 'green', 'black', 'blue', 'purple', 'black', 'orange', 'yellow', 'gold', 'white', 'pink']
    numbers = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    yn = ['yes', 'no']
    top_answers = []
    id_answers = np.load('id_answers.npy').item()

    for i in range(0, pred.shape[1]):
        top_answers.append((id_answers[i], pred[0, i]))

    top_answers = sorted(top_answers, key=lambda x: x[1], reverse=True)
    question = question.lower()
    top_color_answers, top_numbers_answers, top_yn_answers, top_other_answers = [], [], [], []
    ans = top_answers[0][0]

    for item in top_answers:
        if 'how' in question and item[0] in numbers:
            top_numbers_answers.append(item)
            if len(top_numbers_answers) == 1:
                ans = top_numbers_answers[0][0]
            if len(top_numbers_answers) == 10:
                print(top_numbers_answers)
                break
        elif 'color' in question and item[0] in colors:
            top_color_answers.append(item)
            if len(top_color_answers) == 1:
                ans = top_color_answers[0][0]
            if len(top_color_answers) == 7:
                print(top_color_answers)
                break
        elif 'how' not in question and 'color' not in question and question[0:2] != 'is' and question[0:3] != 'are':
            top_other_answers.append(item)
            if len(top_other_answers) == 1:
                ans = top_other_answers[0][0]
            if len(top_other_answers) == 10:
                print(top_other_answers)
                break

    print(question)
    print(top_answers)
    print(ans)

    return ans
