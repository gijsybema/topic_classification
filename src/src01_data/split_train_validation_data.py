import os 
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import utils functions
#src_dir = os.path.join('/home/cdsw', 'src')
#sys.path.append(src_dir)
from src.src00_utils.data_cleaning import *

# input file (sys argument index 1)
input_csv_file = sys.argv[1]
#input_csv_file = '20210111_Labels_NPS_feedback_subtopics_1.csv'

path = os.path.join('/home/cdsw', 'data', '01-raw', '03-labeled_data')
os.chdir(path)

# read csv
df = pd.read_csv(input_csv_file, sep = ';', header = 0, dtype = object)
print(df.shape)

# remove rows where topic is 'Leeg'
df = df[df['Topic_1'] != 'Leeg']

# create list of columns that contain different topics
columns_to_keep = [*df.columns[:3]]
df_wide = restructure_data(df, columns = columns_to_keep)

# remove rows that do not contain feedback
df_wide = df_wide[~df_wide['NPS_feedback'].apply(is_not_feedback)]

# create train and validation set
# validation set should be n = 500 
train, val = train_test_split(df_wide, test_size=500, random_state=42)

# Save train and validation data in separate csv files

path = os.path.join('/home/cdsw', 'data', '02-intermediate')
os.chdir(path)

print(f'Write files to training_data.csv and validation_data.csv in directory {path}')
train.to_csv('training_data.csv', index=False)
val.to_csv('validation_data.csv', index=False)
print('Success')