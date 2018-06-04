import numpy as np
import tensorflow as tf
import re
from nltk.stem import PorterStemmer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import csv
 
 
Mytypes = ['0','1','2','3']
Questions = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\train\\questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
Answers = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\train\\answers.txt').read().split('\n')
Types = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\train\\types.txt').read().split('\n')
 
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
 
clean_questions = []
 
for i in range(0 , len(Questions)):
    if(Types[i] in Mytypes):
        clean_questions.append(clean_text(Questions[i]))
 
word_freq ={}
 
for question in clean_questions:
   for word in question.split():
      #word = ps.stem(word)
      if word not in word_freq:
        word_freq[word]=1
      else:
        word_freq[word]+=1
 
Topwords = sorted([(count,word) for word,count in word_freq.items()], reverse=True)
 
threshold = 13
 
Vocab = [word for count,word in Topwords]
 
Vocab = Vocab[:1998]
 
ques_id = {Vocab[i]:i+1 for i in range(0,len(Vocab))} 
 
Vocab.append('UNK')
 
ques_id['UNK']= len(Vocab)
 
np.save('ques_id.npy', ques_id) 
 
""" encoding questions """
 
encoding_questions=[]
 
for question in clean_questions:
   encode_question=[]
   for word in question.split():
       #word = ps.stem(word)
       if(word in ques_id):
           encode_question.append(ques_id[word])
       else:
           encode_question.append(ques_id['UNK'])
   encoding_questions.append(encode_question)
 
encoding_questions = pad_sequences(encoding_questions, maxlen=55, padding='post')
 
with open("Q_train" + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(encoding_questions)
 
""" Generating Answers train"""
 
encoding_answers = []
id_answers = {}
answers_id = {}
answer_freq = {}
answer_id = 0
 
for i in range(0,len(Answers)):
     if(Types[i] in Mytypes):
         if(Answers[i] not in answer_freq):
           answer_freq[Answers[i]]=1
         else:
           answer_freq[Answers[i]]+=1
 
 
for word in answer_freq :
    answers_id[word]= answer_id
    id_answers[answer_id]= word
    answer_id+=1 
 
np.save('id_answers.npy',id_answers) 
 
y_true = []
 
for i in range(0,len(Answers)):
     if(Types[i] in Mytypes):
         y_true.append(answers_id[Answers[i]])
 
y_true = np_utils.to_categorical(y_true, answer_id ) #answer_id Number of Classes
 
with open("y_true_train" + ".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(y_true)
 
""" Generating Ques test """
 
q_test = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\test\\questions.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
ans_test = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\test\\answers.txt').read().split('\n')
types_test = open('C:\\Users\\abdullahfcis\\Desktop\\VQA Dataset\\test\\types.txt').read().split('\n')
 
 
clean_questions = []
 
for i in range(0 , len(q_test)):
    if(types_test[i] in Mytypes):
        clean_questions.append(clean_text(q_test[i]))
 
 
encoding_questions=[]
 
for question in clean_questions:
   encode_question=[]
   for word in question.split():
       #word = ps.stem(word)
       if(word in ques_id):
           encode_question.append(ques_id[word])
       else:
           encode_question.append(ques_id['UNK'])
   encoding_questions.append(encode_question)
 
encoding_questions = pad_sequences(encoding_questions, maxlen=55, padding='post')
 
with open("Q_test"+".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(encoding_questions)
 
""" y_true_test """
 
y_true_test = []
 
for i in range(0,len(ans_test)):
     if(types_test[i] in Mytypes):
         y_true_test.append(answers_id[ans_test[i]])
 
y_true_test = np_utils.to_categorical(y_true_test, answer_id )
 
with open("y_true_test"+".csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(y_true_test)