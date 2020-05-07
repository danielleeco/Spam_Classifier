import nltk
from nltk.corpus import stopwords
import string
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


df = pd.read_csv('spam.csv')[['label', 'sms']]

df.groupby('label').describe()

df['length'] = df['sms'].apply(len)


def process_text(texts):
    '''
    What will be covered:
    1. Remove punctuation
    2. Remove stopwords
    3. Return list of clean text words
    '''
    
    #1
    nopunc = [char for char in texts if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    
    #3
    return clean_words


df['sms'].apply(process_text).head()

X, y = df[['sms']], df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

pipeline = Pipeline([
    ('bow',CountVectorizer(analyzer=process_text)), # converts strings to integer counts
    ('tfidf',TfidfTransformer()), # converts integer counts to weighted TF-IDF scores
    ('classifier',MultinomialNB()) # train on TF-IDF vectors with Naive Bayes classifier
])

# fit the model
pipeline.fit(X_train.squeeze(), y_train.squeeze())

# predict model by testing data
predictions = pipeline.predict(X_test.squeeze())

# classification metrics
print(classification_report(y_test.squeeze(), predictions))

# accuracy
print(accuracy_score(y_test.squeeze(), predictions))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test.squeeze(), predictions),annot=True)
