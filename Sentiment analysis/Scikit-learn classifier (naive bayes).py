import pathlib
import sklearn
import numpy
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load dataset
movie_reviews_folder = "C:\\Users\\pienk\\jupyter projects\\text mining jup\\ba-text-mining-master\\lab_sessions\\lab3\\final project\\movie_reviews"                               

#Model with TF-IDF and min_df=2
# loading all files as training data.
movie_reviews_train = load_files(str(movie_reviews_folder))

mov_vec = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

mov_counts = mov_vec.fit_transform(movie_reviews_train.data)

# Convert raw frequency counts into TF-IDF values
tfidf_transformer = TfidfTransformer()
mov_tfidf = tfidf_transformer.fit_transform(mov_counts)

docs_train, docs_test, y_train, y_test = train_test_split(
    mov_tfidf, # the tf-idf model   
    movie_reviews_train.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

clf = MultinomialNB().fit(docs_train, y_train)

y_pred = clf.predict(docs_test)


print('Classification report of the model with TF-IDF and min_df=2  \n', classification_report(y_true=y_test,
                            y_pred=y_pred,
                            target_names=movie_reviews_train.target_names))


#Model with Bag of words representation ('airline_count') and min_df=2
# loading all files as training data.
movie_reviews_train2 = load_files(str(movie_reviews_folder))

mov_vec2 = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

mov_counts2 = mov_vec2.fit_transform(movie_reviews_train2.data)

docs_train, docs_test, y_train2, y_test2 = train_test_split(
    mov_counts2,  
    movie_reviews_train.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

clf2 = MultinomialNB().fit(docs_train, y_train2)

y_pred2 = clf2.predict(docs_test)


print('Classification report of the model with Bag of words representation and min_df=2  \n',classification_report(y_true=y_test2,
                            y_pred=y_pred2,
                            target_names=movie_reviews_train.target_names))


#Model with TF-IDF and min_df=5
# loading all files as training data.
movie_reviews_train3 = load_files(str(movie_reviews_folder))

mov_vec3 = CountVectorizer(min_df=5, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

mov_counts3 = mov_vec3.fit_transform(movie_reviews_train3.data)

# Convert raw frequency counts into TF-IDF values
tfidf_transformer3 = TfidfTransformer()
mov_tfidf3 = tfidf_transformer3.fit_transform(mov_counts3)

docs_train, docs_test, y_train3, y_test3 = train_test_split(
    mov_tfidf3, # the tf-idf model   
    movie_reviews_train.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

clf3 = MultinomialNB().fit(docs_train, y_train3)

y_pred3 = clf3.predict(docs_test)

#best performance
print('Classification report of the model with TF-IDF and min_df=5  \n', classification_report(y_true=y_test3,
                            y_pred=y_pred3,
                            target_names=movie_reviews_train.target_names))

#Model with TF-IDF and min_df=10
# loading all files as training data.
movie_reviews_train4 = load_files(str(movie_reviews_folder))

mov_vec4 = CountVectorizer(min_df=10, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english')) # stopwords are removed

mov_counts4 = mov_vec4.fit_transform(movie_reviews_train4.data)

# Convert raw frequency counts into TF-IDF values
tfidf_transformer4 = TfidfTransformer()
mov_tfidf4 = tfidf_transformer4.fit_transform(mov_counts4)

docs_train, docs_test, y_train4, y_test4 = train_test_split(
    mov_tfidf4, # the tf-idf model   
    movie_reviews_train.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

clf4 = MultinomialNB().fit(docs_train, y_train4)

y_pred4 = clf4.predict(docs_test)


print('Classification report of the model with TF-IDF and min_df=10  \n', classification_report(y_true=y_test4,
                            y_pred=y_pred4,
                            target_names=movie_reviews_train.target_names))

import pandas as pd
from sklearn.metrics import classification_report

test_location = "C:\\Users\\pienk\\jupyter projects\\text mining jup\\ba-text-mining-master\\lab_sessions\\lab3\\final project\\sentiment-topic-final-test.tsv"              
test_data=pd.read_csv(test_location, sep='\t')
test_data_sent= test_data['text']
test_labels= []
for i in range(0,10):
    if (test_data['sentiment'][i]) == "positive": 
        test_labels.append(1) 
    elif (test_data['sentiment'][i]) == "negative": 
        test_labels.append(0) 
    else: #neutral
        test_labels.append(2) 
    
#print(test_labels)

# transform the test data using the vectorizers used in the models
test_counts = mov_vec.transform(test_data['text'])
test_tfidf = tfidf_transformer.transform(test_counts)
test_counts2 = mov_vec2.transform(test_data['text'])
test_tfidf3 = tfidf_transformer3.transform(mov_vec3.transform(test_data['text']))
test_tfidf4 = tfidf_transformer4.transform(mov_vec4.transform(test_data['text']))

# predict the sentiment of the test data using the trained models
y_pred_tfidf_min_df2 = clf.predict(test_tfidf)
y_pred_counts_min_df2 = clf2.predict(test_counts2)
y_pred_tfidf_min_df5 = clf3.predict(test_tfidf3)
y_pred_tfidf_min_df10 = clf4.predict(test_tfidf4)

# print the classification reports of each model
print('Classification report of the model with TF-IDF and min_df=2\n', 
      classification_report(y_true=test_labels, y_pred=y_pred_tfidf_min_df2,
                            labels=clf.classes_))
print('Classification report of the model with Bag of words representation and min_df=2\n',
      classification_report(y_true=test_labels, y_pred=y_pred_counts_min_df2,
                            labels=clf2.classes_))
print('Classification report of the model with TF-IDF and min_df=5\n',
      classification_report(y_true=test_labels, y_pred=y_pred_tfidf_min_df5,
                            labels=clf3.classes_))
print('Classification report of the model with TF-IDF and min_df=10\n',
      classification_report(y_true=test_labels, y_pred=y_pred_tfidf_min_df10,
                            labels=clf4.classes_))


import string
import sklearn.feature_extraction

movie_reviews_train2 = load_files(str(movie_reviews_folder))

mov_vec2 = CountVectorizer(min_df=2, # If a token appears fewer times than this, across all documents, it will be ignored
                             tokenizer=nltk.word_tokenize, # we use the nltk tokenizer
                             stop_words=stopwords.words('english').append(list(string.punctuation))) # stopwords are removed
                    

mov_counts2 = mov_vec2.fit_transform(movie_reviews_train2.data)

docs_train, docs_test, y_train2, y_test2 = train_test_split(
    mov_counts2,  
    movie_reviews_train2.target, # the category values for each tweet 
    test_size = 0.20 # we use 80% for training and 20% for development
    ) 

clf2 = MultinomialNB().fit(docs_train, y_train2)

y_pred2 = clf2.predict(docs_test)


def important_features_per_class(vectorizer,classifier,n=80): #n is the number of top features
    class_labels = classifier.classes_
    feature_names =vectorizer.get_feature_names()
    topn_class1 = sorted(zip(classifier.feature_count_[0], feature_names),reverse=True)[:n]
    topn_class2 = sorted(zip(classifier.feature_count_[1], feature_names),reverse=True)[:n]
    print("Important words in negative documents")
    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)
    print("-----------------------------------------")
    print("Important words in positive documents")
    for coef, feat in topn_class2:
        print(class_labels[1], coef, feat)  
        
        
        
important_features_per_class(mov_vec2, clf2)

