import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from flask import Flask, render_template, request
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#NLTK_DATA = '/code/nltk'

#python -m nltk.download -d NLTK_DATA stopwords punkt
#nltk.download('punkt')
#nltk.download('stopwords', download_dir='/code/nltk')
#nltk.download('punkt', download_dir='/code/nltk')

app = Flask(__name__)
model = pickle.load(open("GRU_model.pkl", "rb"))
#model = tf.saved_model.load('saved_GRU_model.h5')

class FormQuery(BaseModel):
    Article: Text = Field(..., validation_alias="Article")


def token_pad_text(Article):

    text = Article.lower()
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]

    #tokenizer = Tokenizer()
    tokenizer = Tokenizer(num_words=total_uniq_words)

    tokenizer.fit_on_texts(tokens)
    text_vect = tokenizer.texts_to_sequences(tokens)
    
    text_pad = sequence.pad_sequences(test_vect,
                                  value=0,
                                  padding='post',
                                  truncating='post',
                                  maxlen=128)
    return text_pad

@app.route('/', methods=['GET'])

def fake_news():
        return render_template("index.html")

@app.route('/predict/', methods=['POST'])

def result():
        if request.method == 'POST':
            Article = request.form['Article']
            
            text_pad = token_pad_text(Article)

        #reg = model.predict([info_stopwords])
        pred = model.predict(text_pad)
        pred = [1 if prd > 0.5 else 0 for prd in pred]
        return render_template("prediction.html", isfake=pred)

if __name__ == '__main__':
    app.debug = True
    #app.run(
    #    host='0.0.0.0',
    #    port=5000,
    #    debug=True)
