'''
This script is inspired by the following tutorials:
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
https://www.depends-on-the-definition.com/guide-to-word-vectors-with-gensim-and-keras/

This file trains a model to predict the sentiment of text passages.
'''


import pickle
import random
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential

#Load data
print("Loading data")

with open("data/train_pos.pickle", 'rb') as handle:
    train_pos = pickle.load(handle)

with open("data/train_neg.pickle", 'rb') as handle:
    train_neg = pickle.load(handle)

with open("data/test_pos.pickle", 'rb') as handle:
    test_pos = pickle.load(handle)

with open("data/test_neg.pickle", 'rb') as handle:
    test_neg = pickle.load(handle)

data = train_pos + train_neg + test_pos + test_neg

#Embeddings
w2v_path = "models/word2vec.model"
VECTOR_EMBEDDING_DIMENSION = 100

try:
    print("Loading word2vec model")
    model = Word2Vec.load(w2v_path)
    print("Loaded word2vec model")
except:
    print("Loading of word2vec failed")
    print("Generating new word2vec model")
    model = Word2Vec(data, size=VECTOR_EMBEDDING_DIMENSION, window=5, min_count=8, workers=4)
    print("word2vec generated")
    model.save(w2v_path)

vocab = list(model.wv.vocab.keys())
print("Vocabulary size: {}".format(len(vocab)))

counter = Counter()

def countWordAppearances(corpus):
    for sentence in corpus:
        counter.update(sentence)

countWordAppearances(data)

train_label = list(np.concatenate([np.ones(len(train_pos)), np.zeros(len(train_neg))]))
test_label = list(np.concatenate([np.ones(len(test_pos)), np.zeros(len(test_neg))]))

train = train_pos + train_neg
test = test_pos + test_neg

training = list(zip(train, train_label))
testing = list(zip(test, test_label))

random.shuffle(training)
random.shuffle(testing)

train, train_label = zip(*training)
train_label = np.array(train_label)
test, test_label = zip(*testing)
test_label = np.array(test_label)





MAX_SEQUENCE_LENGTH = 500

word_index = {t[0]: i+1 for i,t in enumerate(counter.most_common(len(vocab)))}

getSequences = lambda corpus: [[word_index.get(t, 0) for t in sentence] for sentence in corpus]


train_sequences = getSequences(train)
test_sequences = getSequences(test)

train_padded = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")


wv_matrix = (np.random.rand(len(vocab), VECTOR_EMBEDDING_DIMENSION) - 0.5) / 5.0

for word, i in word_index.items():
    if i >= len(vocab):
        continue
    try:
        embedding_vector = model[word]
        # words not found in embedding index will be all-zeros.
        wv_matrix[i] = embedding_vector
    except:
        pass

embedding_layer = Embedding(len(vocab),
                     VECTOR_EMBEDDING_DIMENSION,
                     mask_zero=False,
                     weights=[wv_matrix],
                     input_length=MAX_SEQUENCE_LENGTH,
                     trainable=False)

batch_size = 32

print('Build model...')
model = Sequential()
model.add(embedding_layer)
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

print('Train...')
model.fit(train_padded, train_label,
          batch_size=batch_size,
          epochs=2)
score, acc = model.evaluate(test_padded, test_label,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save('models/sentiment.h5')
