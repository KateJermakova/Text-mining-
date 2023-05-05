from pathlib import Path
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize
import numpy
import sklearn_crfsuite
from sklearn_crfsuite import CRF
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

from itertools import chain
import scipy.stats

import gensim

# Importing text file NER-final_test.tsv
cur_dir = Path().resolve()
path_to_file = Path.joinpath(cur_dir, 'NER-final-test.tsv')
text_test = pd.read_csv(path_to_file, sep = '\t')
text_test = text_test.fillna(method="ffill")

text_train = pd.read_csv("/Users/jeroenwalchenbach/Documents/Documents/AI/text mining/ba-text-mining-master/lab_sessions/lab4/kaggle/ner_dataset.csv", encoding="latin1")
text_train = text_train.fillna(method="ffill")

word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format('/Users/jeroenwalchenbach/Documents/Documents/AI/text mining/ba-text-mining-master/lab_sessions/GoogleNews-vectors-negative300.bin', binary=True) 

for index, instance in text_test.iterrows():
    print()
    print(index)
    print(instance) # you can access information by using instance['A COLUMN NAME'] which you can use to convert to a dictionary needed for the feature representation.
    print('NERC label', instance['BIO NER tag'])
    break
    
words = list(set(text_test["token"].values))
tag = list(set(text_test["BIO NER tag"].values))

class SentenceGetterTest(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                     s["BIO NER tag"].values.tolist())]
        token_func = lambda f: [(t) for t in zip(f["token"].values.tolist())]
        self.grouped = self.data.groupby("sentence id").apply(agg_func)
        self.grouped_tokens = self.data.groupby("sentence id").apply(token_func)
        self.sentences = [s for s in self.grouped]
        self.tokens = [f for f in self.grouped_tokens]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

class SentenceGetterTrain(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        
getter_test = SentenceGetterTest(text_test)
getter_train = SentenceGetterTrain(text_train)

sentences_test = getter_test.sentences
tokens_test = getter_test.tokens
sentences_train = getter_train.sentences
sent_train = getter_train.get_next()

# Adding position tags tothe test data.
pos_tags_per_sentence = []
for tokens in tokens_test:
    for token in tokens:
        t = nltk.pos_tag(token)
        pos_tags_per_sentence.append(t)
        
#making the test data compatible with the training data.
n = len(sentences_test)

def tokens_to_sentence(len_sentences, sentences, tokens):
    f = 0
    h = 0
    sentence = []
    for i in range(len_sentences):
        g = len(sentences[i])
        if i == 0:
            sentence.append(tokens[0:g])
            i+=1
        else:
            f = len(sentences_test[i-1])
            h += f
            sentence.append(tokens[h:(h+g)])
            i+=1
    return sentence      
test_data = []
tags_per_sentence = tokens_to_sentence(n, sentences_test, pos_tags_per_sentence)
for tags, label in zip(tags_per_sentence,sentences_test):
    n = len(tags)
    for i in range(n):
        test_data.append([tags[i][0]+(label[i][1],)])


test_data = tokens_to_sentence(len(sentences_test), sentences_test, test_data)
please_work = []
for i in range(len(test_data)):
    for f in range(len(test_data[i])):
        please_work.append(test_data[i][f][0])

test_data = tokens_to_sentence(len(sentences_test), sentences_test, please_work)

print(len(test_data))

def word2vectors(data):
    input_vectors=[]
    labels=[]
    # for token, pos, ne_label in text_train.iob_words():
    for i in range(len(data)):
        for token, pos, ne_label in data[i]:  
            if token!='' and token!='DOCSTART':
                if token in word_embedding_model:
                    vector=word_embedding_model[token]
                else:
                    vector=[0]*300
                input_vectors.append(vector)
                labels.append(ne_label)
        i +=1
    return input_vectors, labels
            
vectors_test, labels_test_svm = word2vectors(test_data)
vectors_train, labels_train_svm = word2vectors(sentences_train)
labels_train_svm =[w.upper() for w in labels_train_svm]

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]
  
X_train_csv = [sent2features(s) for s in sentences_train]
y_train_csv = [sent2labels(s) for s in sentences_train]
for i in range(len(y_train_csv)):
    y_train_csv[i] = [w.upper() for w in y_train_csv[i]]
    i+=1
X_test_csv = [sent2features(s) for s in test_data]
y_test_csv = [sent2labels(s) for s in test_data]

lin_clf = svm.LinearSVC()
lin_clf.fit(vectors_train[:5000], labels_train_svm[:5000])

labels_svm = list(lin_clf.classes_)
labels_svm.remove('O')

y_pred_svm = lin_clf.predict(vectors_test)
metrics.flat_f1_score([labels_test_svm], [y_pred_svm],
                      average='weighted',labels=labels_svm)

sorted_labels_svm = sorted(
    labels_svm, 
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    [labels_test_svm], [y_pred_svm], labels=sorted_labels_svm, digits=3
))
