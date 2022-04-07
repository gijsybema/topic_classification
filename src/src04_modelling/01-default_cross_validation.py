import os 
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Machine Learning preparation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Machine Learning Cross-validation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Machine Learning Algorithms
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Machine Learning metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Saving Machine Learning model
import joblib

# topic (sys argument index 1)
topic = sys.argv[1]
#topic = 'telefoon_contact'

# dictionary for input and respective topic
topics_dict = {
'afhandeling_proces' :'Afhandeling & Processen'
,'algemeen' :'Algemene ervaring'
,'telefoon_contact' :'Contact leggen met medewerker'
,'digitaal' :'Digitale mogelijkheden'
,'dig_account' :'Account & login web'
,'dig_contact_web' :'Contact via web'
,'dig_func_web' :'Functionaliteit web'
,'dig_gebruik_web' :'Gebruiksgemak web'
,'houding_gedrag' :'Houding & Gedrag medewerker'
,'info' :'Informatievoorziening'
,'kennis_vaardigheden' :'Kennis & Vaardigheden medewerker'
,'kv_deskundig' :'Deskundigheid medewerker'
,'kv_advies' :'Kwaliteit van advies medewerker'
,'prijs_kwaliteit' :'Prijs & Kwaliteit producten'
}

topic_name = topics_dict.get(topic)

col_names = ['NPS_feedback_prep', topic_name]

# change working directory to point to directory with cleaned data
path_in = os.path.join('/home/cdsw', 'data', '03-cleaned')
os.chdir(path_in)

# load training file
df = pd.read_csv('train_clean.csv', usecols = col_names)

print(f'Number of rows: {df.shape}')

# Build model

print(f'Topic to train model on: {topic_name}')

# Define X and y
X = df['NPS_feedback_prep']
y = df[topic_name]

# change datatype of y to int
y = y.astype('int')

print(f'Number of rows in full dataset: {X.shape[0]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f'Number of rows in train dataset: {X_train.shape[0]}')
print(f'Number of rows in test dataset: {X_test.shape[0]}')

print('Cross-validation is performed on data in train dataset')

# Algorithm and vectorizer selection of promising models

# Initializing Vectorizers
vec_count = CountVectorizer(strip_accents='unicode')
vec_tfidf = TfidfVectorizer(strip_accents='unicode')

# Initializing Classifiers
clf_nb  = MultinomialNB()
clf_lr  = LogisticRegression(solver='lbfgs', random_state=42, max_iter=1000)
clf_rf  = RandomForestClassifier(random_state=42)
clf_svm = SVC(kernel = 'linear', random_state=42)
clf_ada = AdaBoostClassifier(random_state = 42)
clf_gb = GradientBoostingClassifier(random_state = 42)

# Building the model pipelines incl. preprocessing where needed 
pipe_nb_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_nb)])

pipe_nb_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_nb)])

pipe_lr_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_lr)])

pipe_lr_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_lr)])

pipe_rf_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_rf)])

pipe_rf_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_rf)])

pipe_svm_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_svm)])

pipe_svm_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_svm)])

pipe_ada_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_ada)])

pipe_ada_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_ada)])

pipe_gb_count  = Pipeline([('vec', vec_count),
                           ('clf', clf_gb)])

pipe_gb_tfidf  = Pipeline([('vec', vec_tfidf),
                           ('clf', clf_gb)])

ests = (pipe_nb_count, pipe_nb_tfidf
        , pipe_lr_count, pipe_lr_tfidf
        , pipe_rf_count, pipe_rf_tfidf
        , pipe_svm_count, pipe_svm_tfidf
        , pipe_ada_count, pipe_ada_tfidf
        , pipe_gb_count, pipe_gb_tfidf
       )

names_ests = ('Naive Bayes CountVec', 'Naive Bayes Tfidf'
              , 'LogisticRegression CountVec', 'LogisticRegression Tfidf'
              , 'Random Forest CountVec', 'Random Forest Tfidf'
              , 'SVM CountVec', 'SVM Tfidf'
              , 'Adaboost Countvec', 'Adaboost Tfidf'
              , 'GradientBoosting Countvec', 'GradientBoosting Tfidf'
             )

print('Initialized models')

# create a new path to store results
PATH_TO_RESULTS_MODEL_SELECTION = os.path.join('/home/cdsw', 'results', 'model_selection')
destination_folder = os.path.join(PATH_TO_RESULTS_MODEL_SELECTION, topic)
Path(destination_folder).mkdir(parents=True, exist_ok=True)

print(f'Created folder to store cross-validation results - {topic}')

# specify filename
file_name = f'01_default_cv_results_{topic}.txt'

destination_and_file = os.path.join(destination_folder, file_name)

# first clear file, if exists
open(destination_and_file, "w").close()

print('Writing cross-validation results to file')

# Save cross-validation results to a file
for est, name in zip(ests, names_ests):
    y_train_pred = cross_val_predict(est, X_train, y_train, cv=5)
    accuracy = accuracy_score(y_train, y_train_pred)
    precision = precision_score(y_train, y_train_pred)
    recall = recall_score(y_train, y_train_pred)
    f1 = f1_score(y_train ,y_train_pred)
    confusion_matrix = pd.crosstab(y_train, y_train_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    with open(destination_and_file, 'a') as f:
        f.write(f'{name} - Cross-validation results\n')
        f.write(f'CV Accuracy Score : {accuracy}\n')
        f.write(f'CV Precision Score : {precision}\n')
        f.write(f'CV Recall Score : {recall}\n')
        f.write(f'CV F1 Score : {f1}\n')
        f.write(f'CV Confusion Matrix\n')
        f.write(f'{confusion_matrix}\n')
        f.write(f'\n')

print(f'Default cross-validation results for topic {topic_name} are stored under {destination_and_file}')     

# read file and display results in Shell
with open(destination_and_file, 'r') as f:
    print(f.read())
    
print('END OF SCRIPT')