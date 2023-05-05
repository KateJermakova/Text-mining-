# Import pandas
import pandas as pd
import pathlib
cwd = pathlib.Path.cwd()
test_folder = cwd.joinpath('sentiment-topic-final-test.tsv')
test_data=pd.read_csv(test_folder, sep='\t',header=0)
sentences= test_data['text']
gold= test_data['sentiment']

import nltk 
from nltk.sentiment import vader
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import spacy
nlp = spacy.load("en_core_web_sm")

import sklearn 
from sklearn.metrics import classification_report

vader_model = SentimentIntensityAnalyzer()

# returns the classification (negative/neutral/positive) from vader's scores 
def vader_output_to_label(vader_output):
    """
    map vader output e.g.,
    {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.4215}
    to one of the following values:
    a) positive float -> 'positive'
    b) 0.0 -> 'neutral'
    c) negative float -> 'negative'
    
    :param dict vader_output: output dict from vader
    
    :rtype: str
    :return: 'negative' | 'neutral' | 'positive'
    """
    compound = vader_output['compound']
    
    if compound < -0.3:
        return 'negative'
    elif compound > 0.3:
        return 'positive'
    else:
        return 'neutral'
'''   
assert vader_output_to_label( {'neg': 0.0, 'neu': 1.0, 'pos': 0.0, 'compound': 0.0}) == 'neutral'
assert vader_output_to_label( {'neg': 0.0, 'neu': 0.0, 'pos': 1.0, 'compound': 0.01}) == 'positive'
assert vader_output_to_label( {'neg': 1.0, 'neu': 0.0, 'pos': 0.0, 'compound': -0.01}) == 'negative'
'''

def run_vader(textual_unit, 
              lemmatize=False, 
              parts_of_speech_to_consider=None,
              verbose=0):
    """
    Run VADER on a sentence from spacy
    
    :param str textual unit: a textual unit, e.g., sentence, sentences (one string)
    (by looping over doc.sents)
    :param bool lemmatize: If True, provide lemmas to VADER instead of words
    :param set parts_of_speech_to_consider:
    -None or empty set: all parts of speech are provided
    -non-empty set: only these parts of speech are considered.
    :param int verbose: if set to 1, information is printed
    about input and output
    
    :rtype: dict
    :return: vader output dict
    """
    doc = nlp(textual_unit)
        
    input_to_vader = []

    for sent in doc.sents:
        for token in sent:

            to_add = token.text

            if lemmatize:
                to_add = token.lemma_

                if to_add == '-PRON-': 
                    to_add = token.text

            if parts_of_speech_to_consider:
                if token.pos_ in parts_of_speech_to_consider:
                    input_to_vader.append(to_add) 
            else:
                input_to_vader.append(to_add)

    scores = vader_model.polarity_scores(' '.join(input_to_vader))
    
    if verbose >= 1:
        print()
        print('INPUT SENTENCE', sent)
        print('INPUT TO VADER', input_to_vader)
        print('VADER OUTPUT', scores)

    return scores
  
  
def review_analyzer(test_data, to_lemmatize = False , parts_of_speech_to_consider = None):
    my_reviews = []
    all_vader_output = []
    gold = []
    # reviews with wrong vader output
    mistakes = [] 
    # vader scores
    vader_scores = []
    # counting class instances 
    posi = 0 
    neu = 0 
    neg = 0 

    # settings (to change for different experiments)
    to_lemmatize = True 
    pos = set()

# running restaurant reviews through vader 
    for i in range(0,len(test_data)):
        the_review = test_data['text'][i]
        vader_output = run_vader(the_review, to_lemmatize, verbose=1) # running vader
        vader_label = vader_output_to_label(vader_output) # converting vader output to category
    
        vader_scores.append(vader_output)
        if (test_data['sentiment'][i]) == "positive": 
            posi +=1 
        elif (test_data['sentiment'][i]) == "negative": 
            neg +=1 
        else: 
            neu +=1

    
        my_reviews.append(the_review)
        #print(the_review)
        all_vader_output.append(vader_label)
        score = (test_data['sentiment'][i])
        gold.append(score)
        if vader_label != (score): 
            mistakes.append({"id" : i, "content" : (test_data['text'][i]),"original" : (test_data['sentiment'][i]), "vader" : vader_label})
        #print(score+" vs "+vader_label)

    reviews_and_vader_scores = zip(my_reviews, vader_scores)
 
    # use scikit-learn's classification report
    print("-------- Printing the classification report --------") 
    classification_report = sklearn.metrics.classification_report(gold, all_vader_output)
    print(classification_report)
    print("Total mistakes: ", len(mistakes), "/", len(my_reviews))
    print(f"Class imbalance: negative - {neg}, positive - {posi}, neutral - {neu}.") 

    # Negative misclassifications 
    negative_mistakes = [mistake for mistake in mistakes if mistake["original"] == "negative"]
    print("-------- Printing the negative mistakes --------") 
    print("Total of negative mistakes: ", len(negative_mistakes))
    for i, mistake in enumerate(negative_mistakes): 
        print(mistake)
        
    # Positive misclassifications 
    positive_mistakes = [mistake for mistake in mistakes if mistake["original"] == "positive"]
    print("-------- Printing the positive mistakes --------")
    print("Total of positive mistakes: ", len(positive_mistakes))
    for i, mistake in enumerate(positive_mistakes): 
        print(mistake)
    
    # Positive misclassifications 
    neutral_mistakes = [mistake for mistake in mistakes if mistake["original"] == "neutral"]
    print("-------- Printing the neutral mistakes --------")
    print("Total of neutral mistakes: ", len(neutral_mistakes))
    for i, mistake in enumerate(neutral_mistakes): 
        print(mistake)
        
        
#experiment 
review_analyzer(test_data, to_lemmatize = True , parts_of_speech_to_consider = None)
