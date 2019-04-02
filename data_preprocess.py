
'''
This file preprocesses the data. Download the data from:
#http://ai.stanford.edu/~amaas/data/sentiment/
'''


import os
import pickle
import string
from functools import reduce
from nltk.tokenize import word_tokenize

path_train_pos = 'data/train/pos'
path_train_neg = 'data/train/neg'
path_test_pos = 'data/test/pos'
path_test_neg = 'data/test/neg'

def mergeTextFiles(path):
    data = []

    for filename in os.listdir(path):
        with open(path + '/' + filename, 'r', encoding="utf8") as content_file:
            content = content_file.read()
            data.append(content)
    return data

train_pos = mergeTextFiles(path_train_pos)
train_neg = mergeTextFiles(path_train_neg)
test_pos = mergeTextFiles(path_test_pos)
test_neg = mergeTextFiles(path_test_neg)

#Preprocess
tokenize = lambda corpus: [word_tokenize(review) for review in corpus]
removePunctuation = lambda corpus: [[''.join([c for c in word if not c in string.punctuation]) for word in sentence] for sentence in corpus]
removeEmptyTokens = lambda corpus: [[x for x in sentence if not x == ""] for sentence in corpus]
lowerCase = lambda corpus: [[x.lower() for x in sentence] for sentence in corpus]
removeStopWords = lambda corpus: [[x for x in sentence if not x in "for a of the and to in".split()] for sentence in corpus]

preprocess_functions = [tokenize, removePunctuation, removeEmptyTokens, lowerCase, removeStopWords]

#Apply every function in preprocess_functions to corpus
train_pos = reduce(lambda x, y: y(x), preprocess_functions, train_pos)
train_neg = reduce(lambda x, y: y(x), preprocess_functions, train_neg)
test_pos = reduce(lambda x, y: y(x), preprocess_functions, test_pos)
test_neg = reduce(lambda x, y: y(x), preprocess_functions, test_neg)


with open("data/train_pos.pickle", 'wb') as handle:
    pickle.dump(train_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/train_neg.pickle", 'wb') as handle:
    pickle.dump(train_neg, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/test_pos.pickle", 'wb') as handle:
    pickle.dump(test_pos, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("data/test_neg.pickle", 'wb') as handle:
    pickle.dump(test_neg, handle, protocol=pickle.HIGHEST_PROTOCOL)