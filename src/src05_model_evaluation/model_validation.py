import os 
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd

# Importing Machine Learning model
import joblib

#import evaluation functions from utils functions
src_dir = os.path.join('/home/cdsw', 'src')
sys.path.append(src_dir)
from src00_utils.evaluation_functions import *

# Load cleaned validation data

# change working directory
path_in = os.path.join('/home/cdsw', 'data', '03-cleaned')
os.chdir(path_in)

df = pd.read_csv('validation_clean.csv')

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

topics = [value for key,value in topics_dict.items()]

# Load final models into dictionary
path_in = os.path.join('/home/cdsw', 'models', 'final_models')
os.chdir(path_in)

models = {}
for key, value in topics_dict.items():
    models[value] = joblib.load(f'final_model_{key}.pkl')

# Make predictions

# store X_values in X_val
X_val = df['NPS_feedback_prep'].copy()

# save true values in dataframe
df_true = df[topics].reset_index(drop=True)

# make predictions for each topic
pred = {}
for key, value in models.items():
    print(f'Predicted for topic: {key}')
    pred[key] = value.predict(X_val)

df_pred = pd.DataFrame(pred)

# Performance metrics

print(f'Create folder to store model evaluation results')

# create a new path to store results
destination_folder = os.path.join('/home/cdsw', 'results', 'model_evaluation')
Path(destination_folder).mkdir(parents=True, exist_ok=True)

# specify filename
file_name = f'model_evaluation_results_validation_set.csv'

destination_and_file = os.path.join(destination_folder, file_name)

# metrics for each topic
df_results = label_metrics(df_true, df_pred, labels=topics)

df_results.index.name = 'Performance_metric'

print(f'Save results to file: {file_name}')

df_results.to_csv(destination_and_file, index=True)

print(f'Save confusion matrices to file')

# specify filename
file_name = f'Evaluation_confusion_matrices.txt'

destination_and_file = os.path.join(destination_folder, file_name)

# first clear file, if exists
open(destination_and_file, "w").close()

dicts_confusion_matrix = confusion_matrix_label(df_true, df_pred, labels=topics)

for label, matrix in dicts_confusion_matrix.items():
    with open(destination_and_file, 'a') as f:
        f.write(f'Confusion matrix for topic {label}:\n')
        f.write(f'{matrix}\n')
        f.write(f'\n')

print(f'Confusion matrices for each topic:')

# read file and display results in Shell
with open(destination_and_file, 'r') as f:
    print(f.read())