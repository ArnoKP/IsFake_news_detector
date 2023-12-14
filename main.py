import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing import sequence
from pydantic import BaseModel, Field
from flask import Flask, render_template, request
import requests
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Optional
import io
import json

#from nltk.corpus import stopwords

#nltk.download('punkt')

app = Flask(__name__)
model = pickle.load(open("GRU_model2.pkl", "rb"))
tokenizer = pickle.load(open("Tokenizer.pkl", "rb"))

#with open('tokenizer.pkl', 'rb') as handle:
#    tokenizer = pickle.load(handle)

#with open('tokenizer.json') as f:
#    data = json.load(f)
#    tokenizer = tokenizer_from_json(data)

class FormQuery(BaseModel):
    Article: str = Field(..., validation_alias="Article")
    #Article: Optional[list] = Field(..., validation_alias="Article")



def token_pad_text(text):

    #text = form_query.Article

    text = text.lower()
    tokens = word_tokenize(text)
    #tokens = text.split(' ')
    print(tokens)
    print(type(tokens))
    tokens = [t for t in tokens if t.isalpha()]
    #tokens = [t for t in tokens if t not in stop_words]

    #tokenizer = Tokenizer()
    #tokenizer = Tokenizer(num_words=len(list_words_uniq),
    #tokenizer = Tokenizer(num_words=100996,
    #                  char_level = False,
    #                  oov_token = 'UNKN')

    tokenizer.fit_on_texts(tokens)
    text_vect = tokenizer.texts_to_sequences(tokens)
    
    text_pad = sequence.pad_sequences(text_vect,
                                  value=0,
                                  padding='post',
                                  truncating='post',
                                  maxlen=400)
    print(text_pad)
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

    real_pred = model.predict(text_pad, batch_size=1)

    #if real_pred >= 0.5:
    #    real_pred = 1
    #else:
    #    real_pred = 0

    return render_template("prediction.html", isfake=real_pred)

if __name__ == '__main__':
    app.debug = True
    #app.run(
    #    host='0.0.0.0',
    #    port=5000,
    #    debug=True)
