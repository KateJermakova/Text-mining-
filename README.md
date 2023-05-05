# Text Mining

The purpose of this project was to do sentimant, topic and NERC analysis on the test text. 

### NERC analysis 

The ML models that were used for NERC task were CRF and SVM, Scikit-learn classifiers. The sequences were represented using one-hot-encoding and word embedding in our to be able to train and test the models. Overall, the models use completely different approaches for analysis and prediction: CRF is a sequential labeling model that models the dependencies between labels in a sequence, whereas SVM is a classification model that learns to classify each token independently. Thus it was interesting for us to see the difference in performance of these 2 models. 
For the training was used a database from Kaggle, named “Annotated Corpus for Named Entity Recognition”.

### Sentiment analysis 

For sentiment analysis were used these two approaches: 

- VADER, the Valence Aware Dictionary and Sentiment Reasoner employs a sentiment lexicon which scores tokens on positive, negative, neutral and compound, as well as uses rules for further analysis. The chosen approach for this semantic analysis is polarity-based, where the sentences loaded from the test set get a positive, negative or neutral label, mapped from the VADER output with the threshold values < -0.3 for negative and >0.3 for positive, and neutral otherwise. The vader function is run on lemmatized and tokenized sentences, considering all parts of speech.

- Scikit-learn Naive Bayes classifier is trained on reviews from the movie_reviews dataset, with a 80% training and 20% test split. The training set is vectorized by two methods: unordered bag-of-words representation and information value measured by TF-IDF. For the latter method, different document frequency thresholds, min_df, are tested with values 2, 5, 10. For each model, the training data is preprocessed by tokenizing the sentence and removing stopwords. 

Sentiment Polarity Dataset (Version 2.0) by Bo Pang and Lillian Lee to use for training the Scikit-learn Naive Bayes sentiment classifier.

### Topic analysis 

For topic analysis were used XLNet and RoBERTa. Both XLNet and RoBERTa trained on “The 20 newsgroups” dataset.


 
