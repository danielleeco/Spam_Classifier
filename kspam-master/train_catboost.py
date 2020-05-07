import pandas as pd
import numpy as np
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('spam.csv')[['label', 'sms']]
X, y = df[['sms']], df.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cat_features = []
text_features = ['sms']

def catboost_model(X_train, X_test, y_train, y_test, catboost_params={}, verbose=100, plot=False):
    learn_pool = Pool(    
        X_train, 
        y_train, 
        cat_features=cat_features,
        text_features=text_features,
        feature_names=list(X_train)
    )
    test_pool = Pool(
        X_test, 
        y_test, 
        cat_features=cat_features,
        text_features=text_features,
        feature_names=list(X_train)
    )
    
    catboost_default_params = {
        'iterations': 1000,
#         'learning_rate': 0.1,
        'eval_metric': 'Accuracy',
        'task_type': 'GPU'
    }
    
    catboost_default_params.update(catboost_params)
    
    model = CatBoostClassifier(**catboost_default_params)
    # обучение модели
    model.fit(learn_pool, eval_set=test_pool, verbose=verbose, plot=plot)
    prediction = model.predict(X_test)

    return model

model = catboost_model(X_train, X_test, y_train, y_test, plot=True)

model.predict([['ag'], ['call number 666']])

model.save_model('cb.model.bin')
