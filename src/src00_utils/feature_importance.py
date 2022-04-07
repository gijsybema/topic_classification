# Import packages
import numpy as np
import pandas as pd

# Define functions
def feature_importance_linear(pipe):
    model = pipe.fit(X_train, y_train)
    feature_names = model['vec'].get_feature_names()
    coefs = model['clf'].coef_[0]
    return pd.DataFrame(zip(feature_names, coefs), columns =['feature', 'coef']).sort_values(by='coef', ascending=False).head(10)

def feature_importance_tree(pipe):
    model = pipe.fit(X_train, y_train)
    feature_names = model['vec'].get_feature_names()
    coefs = model['clf'].feature_importances_
    return pd.DataFrame(zip(feature_names, coefs), columns =['feature', 'coef']).sort_values(by='coef', ascending=False).head(10)

def feature_importance_svm(pipe):
    model = pipe.fit(X_train, y_train)
    feature_names = model['vec'].get_feature_names()
    coefs = model['clf'].coef_.T.toarray()[:,0]
    return pd.DataFrame(zip(feature_names, coefs), columns =['feature', 'coef']).sort_values(by='coef', ascending=False).head(10)

def feature_importance(pipe, name):
    if name in ['Naive Bayes CountVec', 'Naive Bayes Tfidf', 'LogisticRegression CountVec', 'LogisticRegression Tfidf']:
        output = feature_importance_linear(pipe)
    elif name in ['SVM CountVec', 'SVM Tfidf']:
        output = feature_importance_svm(pipe)
    elif name in ['Random Forest CountVec', 'Random Forest Tfidf', 'Adaboost Countvec', 'Adaboost Tfidf', 'GradientBoosting Countvec', 'GradientBoosting Tfidf']:
        output = feature_importance_tree(pipe)
    else:
        output = 'Name not found'
    return output