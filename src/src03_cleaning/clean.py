import os 
import sys
import numpy as np
import pandas as pd
from stop_words import get_stop_words
import spacy

#import utils functions
src_dir = os.path.join('/home/cdsw', 'src')
sys.path.append(src_dir)
from src00_utils.data_cleaning import *

# input file (sys argument index 1)
input = sys.argv[1]
#input = 'train'

if input == 'train':
    input_csv_file = 'training_data.csv'
elif input == 'validation':
    input_csv_file = 'validation_data.csv'
else: 
    print('Input is incorrect. Provide either train or validation')

print(f'Cleaning {input_csv_file}')   
    
# import data
path_in = os.path.join('/home/cdsw', 'data', '02-intermediate')
os.chdir(path_in)
df = pd.read_csv(input_csv_file, header = 0)

# initialize stopwords
stopwords = set(get_stop_words('dutch'))

# clean text
df['NPS_feedback_prep'] = df['NPS_feedback'].apply(text_cleaning, stopwords = stopwords)

# Remove that do not contain feedback
df = df[~df['NPS_feedback_prep'].apply(is_not_feedback)]

# load spacy model
nlp = spacy.load('nl_core_news_sm')

# apply lemmatization
df['NPS_feedback_prep'] = df['NPS_feedback_prep'].apply(lemmatize_text, nlp = nlp)

path_out = os.path.join('/home/cdsw', 'data', '03-cleaned')
os.chdir(path_out)

df.to_csv(f'{input}_clean.csv', sep=',', index=False)

print(f'Done, cleaned dataset stored in {input}_clean.csv in folder {path_out}')