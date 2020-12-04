from flask import Flask, request, render_template
# s# import "model/predict.py"

from keras import models
import numpy as np
import pandas as pd
import pickle

from keras import models
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def create_embedding_index(filename):
    embeddings_index = dict()
    f = open(filename, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def create_embedding_matrix(tokenizer, embeddings_index):
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = zeros((vocab_size, 100))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def preprocess(data, tokenizer_filename, emb_filename, vocab_size=33674):
    """
  data is a pandas series object
  """

    # if (type(data)!='pandas.core.series.Series'):
    #   print("Please enter a series object")

    # loading
    with open(tokenizer_filename, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # fit the tokenizer on the documents
    data = data.apply(str)

    embedding_index = create_embedding_index(emb_filename)
    embedding_matrix = create_embedding_matrix(tokenizer, embedding_index)

    max_len = 50  # change accordingly
    test_sequences = tokenizer.texts_to_sequences(data)
    test_padded = pad_sequences(test_sequences, maxlen=max_len, padding='post')

    return test_padded


model = models.load_model('spam2.h5')


def predict_classes(data):
    preprocessed_data = preprocess(pd.Series(data), "tokenizer.pickle", "embed.txt")

    prediction = model.predict_classes(preprocessed_data).flatten()[0]

    # You may want to further format the prediction to make it more
    # human readable
    return prediction


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # NB_spam_model = open('NB_spam_model.pkl','rb')
    # clf = joblib.load(NB_spam_model)

    # cv_model = open('cv.pkl', 'rb')
    # cv = joblib.load(cv_model)

    # if request.method == 'POST':

    #     message = request.form['message']
    #     data = [message]
    #     vect = cv.transform(data).toarray()
    #     my_prediction = clf.predict(vect)

    data = request.form.get('text-message')
    prediction = ""
    if data == None:
        prediction = 'Got None'
    else:
        # model.predict.predict returns a dictionarys
        prediction = predict_classes(data)
    # return json.dumps(str(prediction))

    return render_template('result.html', prediction=prediction)


# @app.route('/predict',methods=['GET','POST'])
# def predict():
#     data = request.form.get('data')
#     if data == None:
#         return 'Got None'
#     else:
#         # model.predict.predict returns a dictionary
#         prediction = predict(data)
#     return json.dumps(str(prediction))


if __name__ == "__main__":
    app.run(debug=True)
