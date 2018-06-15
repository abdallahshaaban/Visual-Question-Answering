import numpy as np
import tensorflow as tf
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
import csv
import os
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


# def encoding_question(text):
#     print(text)
#     text = clean_text(text)
#     print(text)
#     ques_id = np.load('ques_id.npy').item()
#     print(ques_id)
#
#     encoded_ques = np.full((1, 55), 0)
#     idx = 0
#     for word in text.split():
#         if (word in ques_id):
#             encoded_ques[0, idx] = ques_id[word]
#         else:
#             encoded_ques[0, idx] = ques_id['UNK']
#         idx = idx + 1
#
#     for idx in range(55):
#         encoded_ques[0, idx] = 0
#     print(encoded_ques)
#     print(encoded_ques.shape)
#     return encoded_ques

# def encoding_question(text):
#     text = clean_text(text)
#
#     ques_id = np.load('ques_id.npy').item()
#     print(ques_id)
#     encoded_ques = []
#
#     for word in text.split():
#         if word in ques_id:
#             encoded_ques.append(ques_id[word])
#         else:
#             encoded_ques.append(ques_id['UNK'])
#
#     Rem = 55 - len(encoded_ques)
#
#     for i in range(0, Rem):
#         encoded_ques.append(0)
#     encoded_ques = np.array(encoded_ques)
#
#     encoded_ques = np.reshape(encoded_ques, [1, 55])
#     print(encoded_ques.shape)
#     print(encoded_ques)
#
#     return encoded_ques

def encoding_question(text):
    text = clean_text(text)
    ques_id = np.load('ques_id.npy').item()
    encoded_ques = []

    encoded_ques = [one_hot(text, 1000)]
    encoded_ques = pad_sequences(encoded_ques, maxlen=55, padding='post')

    # print(encoded_ques)
    # print(encoded_ques.shape)

    encoded_ques = np.array(encoded_ques)
    encoded_ques = np.reshape(encoded_ques, [1, 55])
    # print(encoded_ques)
    # print(encoded_ques.shape)

    return encoded_ques


# def encoding_answer(id):
#     id_answers = np.load('id_answers.npy').item()
#     return id_answers[id[0]]

def encoding_answer(question, pred):

    with open('answers_types/types.txt', 'r') as read:
        types = read.readlines()

    with open('answers_types/answers.txt', 'r') as read:
        answers = read.readlines()

    answer_types = {}
    top_answers = []
    id_answers = np.load('id_answers.npy').item()

    for i in range(0, len(answers)):
        answer_types[answers[i].strip()] = types[i].strip()

    for i in range(0, pred.shape[1]):
        top_answers.append((id_answers[i], pred[0, i]))

    top_answers = sorted(top_answers, key=lambda x: x[1], reverse=True)
    question = question.lower()
    top_objects_answers, top_color_answers, top_numbers_answers, top_locations_answers, top_other_answers = [], [], [], [], []
    ans = top_answers[0][0]

    print(answer_types)
    for item in top_answers:
        if 'what' in question and 'color' not in question and answer_types[item[0]] == '0':
            top_objects_answers.append(item)
            if len(top_objects_answers) == 1:
                ans = top_objects_answers[0][0]
            if len(top_objects_answers) == 10:
                print(top_objects_answers)
                break
        elif 'how' in question and answer_types[item[0]] == '1':
            top_numbers_answers.append(item)
            if len(top_numbers_answers) == 1:
                ans = top_numbers_answers[0][0]
            if len(top_numbers_answers) == 10:
                print(top_numbers_answers)
                break
        elif 'color' in question and answer_types[item[0]] == '2':
            top_color_answers.append(item)
            if len(top_color_answers) == 1:
                ans = top_color_answers[0][0]
            if len(top_color_answers) == 7:
                print(top_color_answers)
                break
        elif 'where' in question and answer_types[item[0]] == '3':
            top_locations_answers.append(item)
            if len(top_locations_answers) == 1:
                ans = top_locations_answers[0][0]
            if len(top_locations_answers) == 10:
                print(top_locations_answers)
                break
        elif 'how' not in question and 'color' not in question and 'where' not in question and 'what' not in question:
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

