import sys
import os
import argparse
from datetime import datetime
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from stop_words import get_stop_words
import spacy

# import data_manipulation_tools.py
src_dir = os.path.join('/home/cdsw', 'src')
sys.path.append(src_dir)
from src00_utils.data_manipulation import *
from src00_utils.data_cleaning import *

#define list of topics to choose from when parsing arguments
list_topics = ['afhandeling_proces'
               , 'algemeen'
               , 'telefoon_contact'
               , 'digitaal'
               , 'dig_account'
               , 'dig_contact_web'
               , 'dig_func_web'
               , 'dig_gebruik_web'
               , 'houding_gedrag'
               , 'info'
               , 'kennis_vaardigheden'
               , 'kv_deskundig'
               , 'kv_advies'
               , 'prijs_kwaliteit']

class ExtendAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        items = getattr(namespace, self.dest) or []
        items.extend(values)
        setattr(namespace, self.dest, items)

parser = argparse.ArgumentParser()

parser.register('action', 'extend', ExtendAction)

parser.add_argument('topics'
                    , action='extend'
                    , type=str                    
                    , nargs='+'
                    , choices=list_topics
                    )

args = parser.parse_args()

topics = args.topics
print(topics)

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

# get topic names
topic_names = []
for topic_abbrev in topics:
    topic_names.append(topics_dict.get(topic_abbrev))

print(f'Topics to predict {topic_names}')

print('Load expoints datadump')
# Load raw datadump from Expoints

# change working directory
path_in = os.path.join('/home/cdsw', 'data', '01-raw', '01-datadumps')
os.chdir(path_in)

df = pd.read_excel('Aon datadump totaal 1-2-2021 042027.xlsx', header = 0, dtype = object)

print(f'Shape of datadump {df.shape}')

#### Create new columns
print('Create necessary columns')

# Create new column YearMonth with readable format
df['YearMonth'] = df['Datum'].apply(yearmonth)

df['NPS_group'] = NPS_group(df)

# Combine the three NPS open feedback questions to one column
df['NPS_feedback'] = combine_feedback_columns_NPS(df)

#### Clean texts
print('Clean texts')

# Clean texts
stopwords = set(get_stop_words('dutch'))
df['NPS_feedback_prep'] = df['NPS_feedback'].apply(text_cleaning, stopwords = stopwords)

# apply lemmatization with spacy model
nlp = spacy.load('nl_core_news_sm')
df['NPS_feedback_prep'] = df['NPS_feedback_prep'].apply(lemmatize_text, nlp = nlp)

#### Load final models into dictionary
print('Load final models for specified topics')

# specify location of final models
path_in = os.path.join('/home/cdsw', 'models', 'final_models')
os.chdir(path_in)

# load models into dictionary
models = {}
for topic in topics:
    models[topic] = joblib.load(f'final_model_{topic}.pkl')

#### Predictions
print('Make predictions')

# Specify X_val
X_val = df['NPS_feedback_prep'].copy()

feedback_index = X_val.index

# make predictions for each topic, store in dataframe
pred = {}
for key, value in models.items():
    name = topics_dict.get(key)
    print(f'Predicted for topic: {name}')    
    pred[name] = value.predict(X_val)
df_pred = pd.DataFrame(pred, index=feedback_index)

# merge with original data
df_pred_topics = df.merge(df_pred, how='left', left_index=True, right_index=True)

print(f'Shape of final dataframe: {df_pred_topics.shape}')

#### Store data with added predictions in new .csv file with current date
      
# create a new path to store results
destination_folder = os.path.join('/home/cdsw', 'data', '04-predicted')
Path(destination_folder).mkdir(parents=True, exist_ok=True)

# specify filename
d1 = date.today().strftime("%Y%m%d")
file_name = f'datadump_predicted_{d1}.csv'
destination_and_file = os.path.join(destination_folder, file_name)

# to csv
print(f'Store final dataframe with predictions in: {destination_and_file}')      
      
df_pred_topics.to_csv(destination_and_file, index=False)