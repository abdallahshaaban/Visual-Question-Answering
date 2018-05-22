# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 21:21:04 2018

@author: Lenovo-PC
"""
import numpy as np
from predict import model
from predict_Utils import encoding_question , encoding_answer
from Extract_Features_of_Images import ExtractFeatures
model.load_weights("C:\\Users\\Lenovo-PC\\Desktop\\Visual-Question-Answering\\data\\model_weights.h5")

def Predict():
    from PIL import ImageTk, Image
    canvas.image = ImageTk.PhotoImage(Image.open(str(ImageName_Entry.get())))
    canvas.create_image(0,0,anchor = 'nw' , image = canvas.image)
    Answer_Entry.delete(0,END)
    pred = model.predict([encoding_question(str(Question_Entry.get())),ExtractFeatures(str(ImageName_Entry.get()))],batch_size=1)
    print(Question_Entry.get())
    ans = encoding_answer(pred.argmax(axis=1))
    print(pred[0,pred.argmax(axis=1)[0]],"\n")
    Answer_Entry.insert(0,ans)


#-----------------------------------------------------------GUI-----------------------------------------------
from tkinter import *
#Creating the main window
root = Tk()
#Controls
ImageName_Label = Label(root , text = "Image Name")
ImageName_Entry = Entry(root,width=100)
Question_Label = Label(root , text = "Question")
Question_Entry = Entry(root,width=100)
Answer_Label = Label(root , text = "Answer")
Answer_Entry = Entry(root,width=100)
Train_Button = Button(root , text = "Predict the Answer" , command = Predict)
canvas = Canvas(root,width=500,height=500,bg = 'gray')
space_Label = Label(root , text = "                 ")
#Controls' positions
ImageName_Label.grid(row=0 , column=0 )
ImageName_Entry.grid(row=0 , column=1 )
Question_Label.grid(row=1 , column=0 )
Question_Entry.grid(row=1 , column=1 )
Answer_Label.grid(row=3 , column=0 )
Answer_Entry.grid(row=3 , column=1 )
Train_Button.grid(row=2,column=1)
canvas.grid(row = 4 , column = 1)
space_Label.grid(row=0,column=2)
#For Making the window still displayed
root.mainloop()

