import pickle
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from pydantic import BaseModel, Field
from flask import Flask, render_template, request
import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#NLTK_DATA = '/code/nltk'

#python -m nltk.download -d NLTK_DATA stopwords punkt
#nltk.download('punkt')
#nltk.download('stopwords', download_dir='/code/nltk')
#nltk.download('punkt', download_dir='/code/nltk')

#python -m nltk.downloader -d /code/nltk stopwords punkt

app = Flask(__name__)
model = pickle.load(open("GRU_model.pkl", "rb"))
#model = tf.saved_model.load('saved_GRU_model.h5')

class FormQuery(BaseModel):
    Article: str = Field(..., validation_alias="Article")

def token_pad_text(text):

    #text = form_query.Article

    text = text.lower()
    #tokens = word_tokenize(text)
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha()]
    #tokens = [t for t in tokens if t not in stop_words]

    tokenizer = Tokenizer()
    #tokenizer = Tokenizer(num_words=total_uniq_words)

    tokenizer.fit_on_texts(tokens)
    text_vect = tokenizer.texts_to_sequences(tokens)
    
    text_pad = sequence.pad_sequences(text_vect,
                                  value=0,
                                  padding='post',
                                  truncating='post',
                                  maxlen=128)
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
    real_pred = np.mean(real_pred)
    if real_pred >= 0.5:
        real_pred = 1
    else:
        real_pred = 0

    #pred = [1 if prd > 0.5 else 0 for prd in pred]
    return render_template("prediction.html", isfake=real_pred)

if __name__ == '__main__':
    app.debug = True
    #app.run(
    #    host='0.0.0.0',
    #    port=5000,
    #    debug=True)
