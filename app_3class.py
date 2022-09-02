from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sklearn
import pickle
from tensorflow.keras.models import load_model
from keras_preprocessing.text import tokenizer_from_json
#import class_def
import json
import io


app = Flask(__name__)

tokenizer = None

def init():
    global model, tokenizer#,graph
    model = keras.models.load_model('./BiLSTM_3class.h5')
    with open('tokenizer_3class.json') as f:
        data = json.load(f)
    tokenizer = tokenizer_from_json(data)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():    
    global tokenizer
    if request.method == 'POST':
        text_masuk = request.form['text']
        data = [text_masuk]
        #tokenizer = Tokenizer(num_words = vocab_size, char_level = False, oov_token = oov_tok)
        seq = tokenizer.texts_to_sequences(data)
        padded = pad_sequences(seq, maxlen = 80, padding = "post", truncating = "post")
        
        pred_masukan_n = (model.predict(padded))
        pred_masukan = np.argmax(pred_masukan_n)
        if pred_masukan==0:
            hasil=0
        elif pred_masukan==1:
            hasil=1
        else:
            hasil=2
    return render_template('result.html',prediction = hasil)

if __name__ == '__main__':
    init()
    app.run(debug=True)