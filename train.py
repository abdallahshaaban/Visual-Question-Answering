import numpy as np
import keras
import operator
import random
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM, Dense, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from prepare_data import QuestionsData


class Train(object):
    def __init__(self):
           #Train dataset
           self._questions_reader = QuestionsData("data/vqa/Questions/v2_OpenEnded_mscoco_train2014_questions.json", "data/vqa/Annotations/v2_mscoco_train2014_annotations.json")
	        self._answers = self._questions_reader.get_answers()
	        self._questions_info, self._questions = self._questions_reader.read_questions_data()
	        #Validation dataset
	        self._questions_reader_val = QuestionsData("data/vqa/Questions/v2_OpenEnded_mscoco_val2014_questions.json", "data/vqa/Annotations/v2_mscoco_val2014_annotations.json")
	        self._questions_info_val, self._questions_val = self._questions_reader_val.read_questions_data()
           self._vocab_size, self._max_length = 1000, 15
	        self.targets_size = 0


    def get_top_answers(self):
        answers_freq, answers_prob = {}, {}
        top_answers = []

        # Calculate the frequency of each answer
        for key in self._answers:
            for ans in self._answers[key]:
                if ans not in answers_freq:
                    answers_freq[ans] = 1
                else:
                    answers_freq[ans] = answers_freq[ans] + 1

        # Calculate the probability of each answer
        for ans in answers_freq:
            answers_prob[ans] = answers_freq[ans] / float(len(answers_freq))
        top_answers = sorted(answers_prob.items(), key=operator.itemgetter(1))
        top_answers.reverse()
        dic, cnt = {}, 1
        for item in top_answers[:1000]:
            dic[item[0]] = cnt
            cnt = cnt + 1
        return top_answers[:1000], dic

    def load_data(self):
        # load train data
        x_train = [one_hot(q, self._vocab_size) for q in self._questions]
        x_train = pad_sequences(x_train, maxlen=self._max_length, padding='post')
        x_train = np.array(x_train)
        top_answers, answers_info = self.get_top_answers()
        print(answers_info)
        y_train = []
        for i in range(len(x_train)):
            rand = random.randint(0, 999)
            y_train.append(answers_info[top_answers[rand][0]])

        # Store the unique IDs of answers
        unique_answers = {key: 1 for key in y_train}
        self.targets_size = len(unique_answers)

        # load val dataset
        x_val = [one_hot(q, self._vocab_size) for q in self._questions_val]
        x_val = pad_sequences(x_val, maxlen=self._max_length, padding='post')
        x_val = np.array(x_val)
        y_val = []
        for i in range(len(x_val)):
            rand = random.randint(0, 999)
            y_val.append(answers_info[top_answers[rand][0]])

        # Encoding the output data
        y_train, y_val = np.array(y_train), np.array(y_val)
        encoder = LabelEncoder()
        encoder.fit(y_train)
        encoder.fit(y_val)
        encoded_y_train, encoded_y_val = encoder.transform(y_train), encoder.transform(y_val)
        y_train, y_val = np_utils.to_categorical(encoded_y_train), np_utils.to_categorical(encoded_y_val)
        return x_train, y_train, x_val, y_val




# x_train, y_train, x_val, y_val = Train().load_data()
# # print(y_train)
# # print(y_val)
# # print(p)
# # print(len(y_train), len(y_val), len(p))
# #
# #
# #
# # top_ans, dic = Train().get_top_answers()
# #
# # print(top_ans)
# # print(dic)
# print(y_train )




