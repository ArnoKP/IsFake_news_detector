import pickle
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from flask import Flask, render_template, request

app = Flask(__name__)
model = pickle.load(open("GRU_model.sav", "rb"))
model = tf.saved_model.load('saved_GRU_model.h5')

tokenizer = Tokenizer()

def token(text):
    tokenizer.fit_on_texts(text)
    tokenizer.texts_to_sequences(text)
    return text_vect
    
def pad(text_vect):
    sequence.pad_sequences(test_vect,
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
                text = request.form['info_stopwords']
        reg = model.predict([info_stopwords])
        return render_template("prediction.html", isfake=reg)

if __name__ == '__main__':
    app.debug = True
    #app.run(
    #    host='0.0.0.0',
    #    port=5000,
    #    debug=True)
