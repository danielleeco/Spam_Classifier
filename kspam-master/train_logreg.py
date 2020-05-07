from smart_open import smart_open
import pandas as pd
import numpy as np
from numpy import random
import gensim
import nltk
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


df = pd.read_csv('spam.csv')[['label', 'sms']]
df

my_tags = df.label.unique()
train_data, test_data = train_test_split(df, test_size=0.33, random_state=42)


def plot_confusion_matrix(cm, title='Predicted', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title = title
    plt.colorbar()
    tick_marks = np.arange(len(my_tags))
    target_names = my_tags
    plt.xticks(tick_marks, target_names, rotation=0)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def LogisticRegressionClassifier(train_data):    
    tfidf = TfidfVectorizer(min_df=2, 
                        tokenizer=nltk.word_tokenize,
                        preprocessor=None, 
                        stop_words='english')
    
    train_data_features = tfidf.fit_transform(train_data['sms'])
    logreg = linear_model.LogisticRegression(n_jobs=1, C=1e5)
    logreg = logreg.fit(train_data_features, train_data['label'])

    return logreg, tfidf


def get_accuracy(model, tfidf, test_data, tags):
    target = test_data['label']
    data_features = tfidf.transform(test_data['sms'])
    predictions = model.predict(data_features)

    cm = confusion_matrix(target, predictions, labels=my_tags)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plot_confusion_matrix(cm_normalized)

    accuracy = accuracy_score(target, predictions)
    return accuracy


def logreg_predict(model, tfidf, emails):
    data = pd.Series(emails)
    data_features = tfidf.transform(data)
    predictions = model.predict(data_features)
    return predictions[0]


def predict(emails):
    return logreg_predict(model, tfidf, emails)



model, tfidf = LogisticRegressionClassifier(train_data)
print('accuracy:', get_accuracy(model, tfidf, test_data, my_tags))
print(predict('jack what u think about it'), 'jack what u think about it')
