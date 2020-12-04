from keras import models
import numpy as np
import pandas as pd


def create_embeddings_index(filename):
  embeddings_index = dict()
  f = open( filename )
  for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype= 'float32' )
    embeddings_index[word] = coefs
  f.close()
  return embeddings_index

def create_embedding_matrix(tokenizer, embeddings_index):
  vocab_size=len(tokenizer.word_index)+1
  embedding_matrix = zeros((vocab_size, 100))
  for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
  return embedding_matrix




def preprocess(data, tokenizer_filename, emb_filename, vocab_size= 33674):

  """
  data is a pandas series object
  """
  
  # if (type(data)!='pandas.core.series.Series'):
  #   print("Please enter a series object")

  
  from keras.preprocessing.text import Tokenizer
  from numpy import asarray
  from numpy import zeros
  from keras.preprocessing.text import Tokenizer
  from keras.preprocessing.sequence import pad_sequences

  # loading
  with open(tokenizer_filename, 'rb') as handle:
    tokenizer = pickle.load(handle)

 
  # fit the tokenizer on the documents
  data=data.apply(str)
  
  embedding_index=create_embeddings_index(emb_filename)
  embedding_matrix=create_embedding_matrix(tokenizer, embeddings_index)
  
  max_len=50 #change accordingly
  test_sequences = tokenizer.texts_to_sequences(data)
  test_padded = pad_sequences(test_sequences, maxlen = max_len, padding = 'post' )
  
  return test_padded

model=models.load_model('model/spam2.h5')


def predict(data):

    preprocessed_data=preprocess(pd.Series(data), "tokenizer.pickle", "embeddings.txt")
    
    prediction = model.predict_classes(preprocessed_data).flatten()
    
    # You may want to further format the prediction to make it more
    # human readable
    return prediction