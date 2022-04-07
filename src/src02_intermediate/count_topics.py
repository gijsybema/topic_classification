import os 
import sys
import numpy as np
import pandas as pd

# input file (sys argument index 1)
input = sys.argv[1]
#input = 'train'

if input == 'train':
    input_csv_file = 'training_data.csv'
elif input == 'validation':
    input_csv_file = 'validation_data.csv'
else: 
    print('Input is incorrect. Provide either train or validation')

# import data
path = os.path.join('/home/cdsw', 'data', '02-intermediate')
os.chdir(path)
df = pd.read_csv(input_csv_file, header = 0)

# how often does each topic occur
columns_to_keep = [*df.columns[:3]]
topics = [element for element in [*df.columns] if element not in columns_to_keep]

# save results
a = df[topics].sum(axis=0) 
a.name = 'frequency'
b = df[topics].sum(axis=0) / len(df)
b.name = 'proportion'
results = pd.merge(a, b, right_index = True, left_index = True).reset_index().rename(columns={'index':'topic'}) 

path = os.path.join('/home/cdsw', 'results', 'reports')
os.chdir(path)
results.to_csv(f'{input}_topic_counts.csv', sep='\t', index=False)

print('Success')