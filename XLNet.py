# Import libraries
! pip install simpletransformers
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import classification_report
from simpletransformers.classification import ClassificationModel, ClassificationArgs
import matplotlib.pyplot as plt 
import seaborn as sn 

from sklearn.datasets import fetch_20newsgroups

# load only a sub-selection of the categories (4 in our case)
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'sci.space'] 

# remove the headers, footers and quotes (to avoid overfitting)
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), categories=categories, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), categories=categories, random_state=42)

sentiment_topic = pd.read_csv("sentiment-topic-final-test.csv", sep = ",")

# Data Exploration part

from collections import Counter
Counter(newsgroups_train.target)
Counter(newsgroups_test.target)

train = pd.DataFrame({'text': newsgroups_train.data, 'labels': newsgroups_train.target})
test = pd.DataFrame({'text': newsgroups_test.data, 'labels': newsgroups_test.target})

from sklearn.model_selection import train_test_split

train, dev = train_test_split(train, test_size=0.1, random_state=0, 
                               stratify=train[['labels']])

#XLNet

# Model configuration # https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model 
model_args = ClassificationArgs()

model_args.overwrite_output_dir=True # overwrite existing saved models in the same directory
model_args.evaluate_during_training=True # to perform evaluation while training the model
# (eval data should be passed to the training method)

model_args.num_train_epochs=10 # number of epochs
model_args.train_batch_size=32 # batch size
model_args.learning_rate=4e-6 # learning rate
model_args.max_seq_length=256 # maximum sequence length
# Note! Increasing max_seq_len may provide better performance, but training time will increase. 
# For educational purposes set max_seq_len set to 256.

# Early stopping to combat overfitting: https://simpletransformers.ai/docs/tips-and-tricks/#using-early-stopping
#model_args.use_early_stopping=True
#model_args.early_stopping_delta=0.01 # "The improvement over best_eval_loss necessary to count as a better checkpoint"
#model_args.early_stopping_metric='eval_loss'
#model_args.early_stopping_metric_minimize=True
#model_args.early_stopping_patience=2
#model_args.evaluate_during_training_steps=32 # how often you want to run validation in terms of training steps (or batches)

# Checking steps per epoch
steps_per_epoch = int(np.ceil(len(train) / float(model_args.train_batch_size)))
print('Each epoch will have {:,} steps.'.format(steps_per_epoch)) # 64 steps = validating 2 times per epoch

model = ClassificationModel('xlnet', 'xlnet-base-cased', num_labels=4, args=model_args, use_cuda=True) # CUDA is enabled

_, history = model.train_model(train, eval_df=dev) 

# Training and evaluation loss
train_loss = history['train_loss']
eval_loss = history['eval_loss']
plt.plot(train_loss, label='Training loss')
plt.plot(eval_loss, label='Evaluation loss')
plt.title('Training and evaluation loss')
plt.legend()

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(dev)
result

predicted, probabilities = model.predict(sentiment_topic.text.to_list())
sentiment_topic['predicted'] = predicted

# Result (note: your result can be different due to randomness in operations)
# XLNET 
print(classification_report(sentiment_topic['labels'], sentiment_topic['predicted']))
