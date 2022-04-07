import os 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import time

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

# quick and dirty solution, rewrite with argparse when having time
topic = sys.argv[1]
models_list = []
if len(sys.argv) <3:
    raise ValueError('No models provided as input')
elif len(sys.argv) == 3:
    models_list.append(sys.argv[2])
elif len(sys.argv) == 4:
    models_list.append(sys.argv[2])
    models_list.append(sys.argv[3])
elif len(sys.argv) == 5:
    models_list.append(sys.argv[2])
    models_list.append(sys.argv[3])
    models_list.append(sys.argv[4])
elif len(sys.argv) == 6:
    models_list.append(sys.argv[2])
    models_list.append(sys.argv[3])
    models_list.append(sys.argv[4])
    models_list.append(sys.argv[5])
elif len(sys.argv) == 7:
    models_list.append(sys.argv[2])
    models_list.append(sys.argv[3])
    models_list.append(sys.argv[4])
    models_list.append(sys.argv[5])
    models_list.append(sys.argv[6])

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

# get topic name
topic_name = topics_dict.get(topic)

print(f'Topic to train model on: {topic_name}')

# dictionary of possible models
ests_names_dict = {
  'nb_count' : 'Naive Bayes CountVec'
, 'nb_tfidf' : 'Naive Bayes Tfidf'
, 'lr_count' : 'LogisticRegression CountVec'
, 'lr_tfidf' : 'LogisticRegression Tfidf'
, 'rf_count' : 'Random Forest CountVec'
, 'rf_tfidf' : 'Random Forest Tfidf'
, 'svm_count' : 'SVM CountVec'
, 'svm_tfidf' : 'SVM Tfidf'
, 'ada_count' : 'Adaboost Countvec'
, 'ada_tfidf' : 'Adaboost Tfidf'
, 'gb_count' : 'GradientBoosting Countvec'
, 'gb_tfidf' : 'GradientBoosting Tfidf' 
}

# store chosen models in a dictionary
chosen_ests_dict = {}
for model in models_list:
    est_name = ests_names_dict.get(model)
    chosen_ests_dict[model] = est_name

print('Chosen models:')
for key, value in chosen_ests_dict.items():
    print(value)

# specify columns to load data
col_names = ['NPS_feedback_prep', topic_name]

# change working directory
path_in = os.path.join('/home/cdsw', 'data', '03-cleaned')
os.chdir(path_in)

# load training file
df = pd.read_csv('train_clean.csv', usecols = col_names)

print(f'Number of rows: {df.shape}')

# Build model

# Define X and y
X = df['NPS_feedback_prep']
y = df[topic_name]

# change datatype of y to int
y = y.astype('int')

print(f'Number of rows in full dataset: {X.shape[0]}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

print(f'Number of rows in train dataset: {X_train.shape[0]}')
print(f'Number of rows in test dataset: {X_test.shape[0]}')

# Nested cross-validation and vectorizer selection of promising models

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

# Parameter grids
# Naive Bayes
param_grid_nb = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__alpha': [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]}]

# Logistic Regression
param_grid_lr = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__penalty': ['l2'] 
  ,'clf__C': np.logspace(-4, 4, 9)}]

# SVM
param_grid_svm = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__kernel': ['rbf'] 
  ,'clf__C': np.logspace(-4, 4, 8) 
  ,'clf__gamma': np.logspace(-4, 0, 4)}
 , {'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
    ,'vec__min_df': [1,2,3,5] 
    ,'clf__kernel': ['linear'] 
    ,'clf__C': np.logspace(-4, 4, 8)}]

# Random Forest
param_grid_rf = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
  ,'clf__max_features': ['sqrt'] 
  ,'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] 
  ,'clf__min_samples_split': [2, 5, 10] 
  ,'clf__min_samples_leaf': [1, 2, 4] }]

# AdaBoost 
param_grid_ada = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
  ,'clf__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] }]

# Gradient Boosting
param_grid_gb = [{'vec__ngram_range': ((1, 1), (1, 2), (1,3), (1,4)) 
  ,'vec__min_df': [1,2,3,4,5] 
  ,'clf__learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1] 
  ,'clf__n_estimators': [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] 
  ,'clf__max_features': ['sqrt'] 
  ,'clf__max_depth': [int(x) for x in np.linspace(10, 110, num = 11)] 
  ,'clf__min_samples_split': [2, 5, 10] 
  ,'clf__min_samples_leaf': [1, 2, 4] }]

# Need three tuples for nested cross-validation
# 1. names_promising: strings with names of promising models
# 2. ests_promising: initialized pipelines of promising models
# 3. pgrids_promising: parameter grids of promising models

# create tuple of names of promising models
ests_names_list = []
for model in models_list:
    ests_name = ests_names_dict.get(model)
    ests_names_list.append(ests_name)
names_promising = tuple(ests_names_list)

# dictionary of pipelines
ests_pipes_dict = {
  'nb_count' : pipe_nb_count
, 'nb_tfidf' : pipe_nb_tfidf
, 'lr_count' : pipe_lr_count
, 'lr_tfidf' : pipe_lr_tfidf
, 'rf_count' : pipe_rf_count
, 'rf_tfidf' : pipe_rf_tfidf
, 'svm_count' : pipe_svm_count
, 'svm_tfidf' : pipe_svm_tfidf
, 'ada_count' : pipe_ada_count
, 'ada_tfidf' : pipe_ada_tfidf
, 'gb_count' : pipe_gb_count
, 'gb_tfidf' : pipe_gb_tfidf
}

# create tuple of pipelines of promising models
ests_promising_list = []
for model in models_list:
    est_pipe = ests_pipes_dict.get(model)
    ests_promising_list.append(est_pipe)
ests_promising = tuple(ests_promising_list)

# dictionary of parameter grids
ests_pgrids_dict = {
  'nb_count' : param_grid_nb
, 'nb_tfidf' : param_grid_nb
, 'lr_count' : param_grid_lr
, 'lr_tfidf' : param_grid_lr
, 'rf_count' : param_grid_rf
, 'rf_tfidf' : param_grid_rf
, 'svm_count' : param_grid_svm
, 'svm_tfidf' : param_grid_svm
, 'ada_count' : param_grid_ada
, 'ada_tfidf' : param_grid_ada
, 'gb_count' : param_grid_gb
, 'gb_tfidf' : param_grid_gb
}

# create tuple of parameter grids of promising models
pgrids_promising_list = []
for model in models_list:
    pgrid = ests_pgrids_dict.get(model)
    pgrids_promising_list.append(pgrid)
pgrids_promising = tuple(pgrids_promising_list)

print(f'Create folder to store nested cross-validation results - {topic}')

# create a new path to store results
PATH_TO_RESULTS_MODEL_SELECTION = os.path.join('/home/cdsw', 'results', 'model_selection')
destination_folder = os.path.join(PATH_TO_RESULTS_MODEL_SELECTION, topic)
Path(destination_folder).mkdir(parents=True, exist_ok=True)

# specify filename
file_name = f'02_nested_cv_results_{topic}.txt'

destination_and_file = os.path.join(destination_folder, file_name)

# first clear file, if exists
open(destination_and_file, "w").close()

# Nested cross-validation

print('Writing cross-validation results to file')

# specify the number of iterations you would like to try of the random parameters in RandomizedSearch
n_iterations = 50

# Setting up multiple RandomizedCV objects, one for each algorithm
gridcvs = {}
inner_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for pgrid, est, name in zip(pgrids_promising, ests_promising, names_promising):
    gcv = RandomizedSearchCV(estimator=est
                             ,param_distributions=pgrid
                             ,n_iter=n_iterations
                             ,cv=inner_cv
                             ,scoring='f1'
                             ,n_jobs=-1
                             ,verbose=0
                             ,refit=True)
    gridcvs[name] = gcv

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=5)
outer_scores = {}

for name, gs_est in sorted(gridcvs.items()):
    start_time = time.time()
    print(f'Running CV {name}...')
    nested_score = cross_val_score(gs_est, 
                                   X=X_train, 
                                   y=y_train, 
                                   cv=outer_cv,
                                   n_jobs=-1)
    outer_scores[name] = nested_score
    end_time = time.time()
    elapsed_time = round((end_time - start_time), 1)
    mean_nested_score = round((100*nested_score.mean()), 3)
    std_nested_score = round((100*nested_score.std()), 3)
    with open(destination_and_file, 'a') as f:
        f.write(f'{name} - Nested cross-validation results\n')
        f.write(f'Outer F1 Score : {mean_nested_score} +/- {std_nested_score}\n')
        f.write(f'Time elapsed : {elapsed_time} seconds\n')
        f.write(f'\n')
print('Nested Cross-validation completed')

print(f'Default cross-validation results for topic {topic_name} are stored under {destination_and_file}')     

# read file and display results in Shell
with open(destination_and_file, 'r') as f:
    print(f.read())

print('END OF SCRIPT')