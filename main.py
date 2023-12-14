import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from pydantic import BaseModel, Field
from flask import Flask, render_template, request
import requests
import nltk
#from nltk.tokenize import word_tokenize
#from nltk.corpus import stopwords

#nltk.download('punkt')

app = Flask(__name__)
model = pickle.load(open("GRU_model2.pkl", "rb"))
#tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


class FormQuery(BaseModel):
    Article: str = Field(..., validation_alias="Article")

def token_pad_text(text):

    #text = form_query.Article

    text = text.lower()
    tokens = text.split(' ')
    tokens = [t for t in tokens if t.isalpha()]
    #tokens = [t for t in tokens if t not in stop_words]

    #tokenizer = Tokenizer(num_words=len(list_words_uniq),
    tokenizer = Tokenizer(num_words=100996,
                      char_level = False,
                      oov_token = 'UNKN')

    #tokenizer = Tokenizer()

    tokenizer.fit_on_texts(tokens)
    text_vect = tokenizer.texts_to_sequences(tokens)
    
    text_pad = sequence.pad_sequences(text_vect,
                                  value=0,
                                  padding='post',
                                  truncating='post',
                                  maxlen=400)
    return text_pad

@app.route('/', methods=['GET'])
def fake_news():
        return render_template("index.html")

def local_model_result():
    form_query = FormQuery(**request.form.to_dict(flat=True))

    return form_query.Article

@app.route('/predict/', methods=['POST'])
def result():

    #if request.method == 'POST':

        #Article = request.form['Article']
    Article = local_model_result()       
    text_pad = token_pad_text(Article)

        #reg = model.predict([info_stopwords])
    real_pred = model.predict(text_pad)

    if real_pred >= 0.5:
        real_pred = 1
    else:
        real_pred = 0

    return render_template("prediction.html", isfake=real_pred)

if __name__ == '__main__':
    app.debug = True
    #app.run(
    #    host='0.0.0.0',
    #    port=5000,
    #    debug=True)
