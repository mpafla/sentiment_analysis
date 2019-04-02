
from keras.models import load_model


import pickle
import numpy as np
from gensim.models import Word2Vec
from collections import Counter
from keras.preprocessing.sequence import pad_sequences
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


from nltk.tokenize import word_tokenize




class sentimentPredictor():
    def __init__(self):

        #Datasets are needed to set up counter for sequencing strings. There might be smarter ways to do this (e.g., Word2Vec provides this functionality too)
        with open("data/train_pos.pickle", 'rb') as handle:
            train_pos = pickle.load(handle)

        with open("data/train_neg.pickle", 'rb') as handle:
            train_neg = pickle.load(handle)

        with open("data/test_pos.pickle", 'rb') as handle:
            test_pos = pickle.load(handle)

        with open("data/test_neg.pickle", 'rb') as handle:
            test_neg = pickle.load(handle)

        data = train_pos + train_neg + test_pos + test_neg

        counter = Counter()

        def countWordAppearances(corpus):
            for sentence in corpus:
                counter.update(sentence)

        countWordAppearances(data)

        w2v_path = "models/word2vec_tuned.model"

        try:
            print("Loading word2vec model")
            self.w2v = Word2Vec.load(w2v_path)
            print("Loaded word2vec model")
        except:
            print("word2vec could not be loaded")

        vocab = list(self.w2v.wv.vocab.keys())

        self.MAX_NB_WORDS = len(vocab)
        self.MAX_SEQUENCE_LENGTH = 500


        word_index = {t[0]: i + 1 for i, t in enumerate(counter.most_common(self.MAX_NB_WORDS))}

        self.getSequences = lambda corpus: [[float(word_index.get(t, 0)) for t in sentence] for sentence in corpus]

        self.model = load_model('models/sentiment.h5')

        self.analyzer = SentimentIntensityAnalyzer()


    def predict(self, text):
        text_tokenized = word_tokenize(text)
        text_sequence = self.getSequences([text_tokenized])
        text_padded = pad_sequences(text_sequence, maxlen=self.MAX_SEQUENCE_LENGTH, padding="pre", truncating="post")
        prediction_value = self.model.predict(text_padded)

        #Get prediction value from vaderSentiment
        vader_value = self.analyzer.polarity_scores(text)['compound']
        vader_value = (vader_value + 1)/2

        #Take mean of the two values and return
        mean_sentiment = (prediction_value + vader_value)/2

        return mean_sentiment


sp = sentimentPredictor()

text = 'i am not happy today'
print(sp.predict(text))
text = 'i am happy today'
print(sp.predict(text))




