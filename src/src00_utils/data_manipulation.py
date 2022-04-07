# Import packages
import numpy as np
import pandas as pd
from datetime import datetime 

# Define functions

def combine_feedback_columns_NPS(df, NPS_column = '-1_NPS'):
    """
    Returns one array of NPS feedback instead of 3 separate ones
    Based on following conditions:
    If NPS score is between 0 and 6, take the feedback in column ['1_Reden lager dan 7']
    If NPS score is between 7 and 8, take the feedback in column ['2_Wat doen voor een 9 of 10']
    If NPS score is between 9 and 10, take the feedback in column ['3_Reden 9 of 10']
    Else, assign value np.nan
    
    Args:
        df (dataframe): dataframe to be processed
        NPS_column: name of column in dataframe with NPS score
        
    Returns:
        output (array): Use this output to assign it to new column in dataframe
    """
    # Define conditions
    conditions = [
        ((df[NPS_column] >= 0) & (df[NPS_column] <= 6)),
        ((df[NPS_column] >= 7) & (df[NPS_column] <= 8)),
        ((df[NPS_column] >= 9) & (df[NPS_column] <= 10))
    ]

    # Create a list of the values we want to assign for each condition
    values = [
        df['1_Reden lager dan 7']
        , df['2_Wat doen voor een 9 of 10']
        , df['3_Reden 9 of 10']
    ]

    # create a new column and use np.select to assign values to it using our lists as arguments
    output = np.select(conditions, values, default=np.nan)
    
    # Return output
    return output

def yearmonth(value):
    """ Returns the input converted to a string in format YYYY-mm
    
    Args:
        value (datetime): date to convert
    
    Returns:
        output (str): formatted in YYYY-mm
    """
    output = datetime.strftime(value, '%Y-%m')
    return output

def NPS_group(df, NPS_column = '-1_NPS'):
    """
    Returns one array of NPS groups, belonging to the feedback
    Based on following conditions:
    If NPS score is between 0 and 6, Detractor
    If NPS score is between 7 and 8, Passive
    If NPS score is between 9 and 10, Promoter
    Else, assign value np.nan
    
    Args:
        df (dataframe): dataframe to be processed
        NPS_column: name of column in dataframe with NPS score
        
    Returns:
        output (array): Use this output to assign it to new column in dataframe
    """
    # Define conditions
    conditions = [
        ((df[NPS_column] >= 0) & (df[NPS_column] <= 6)),
        ((df[NPS_column] >= 7) & (df[NPS_column] <= 8)),
        ((df[NPS_column] >= 9) & (df[NPS_column] <= 10))
    ]

    # Create a list of the values we want to assign for each condition
    values = ['Detractor', 'Passive', 'Promoter']

    # create a new column and use np.select to assign values to it using our lists as arguments
    output = np.select(conditions, values, default=np.nan)
    
    # Return output
    return output