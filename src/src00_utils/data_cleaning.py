# Import packages
import numpy as np
import pandas as pd
import string
from stop_words import get_stop_words
import spacy

# Define functions

def encode_units(x):
    """ Returns 1 if x is higher than 1, returns 0 if x is lower than 0
    
    Args:
    x (int): integer to be examined
    
    Returns:
    integer 0 or 1
    """
    if x <= 0:
        return 0
    if x >= 1:
        return 1

def restructure_data(df, columns):
    """Restructure a dataframe to the right structure for topic classification
  
    Args: 
    df (DataFrame): Dataframe to be restructured
    columns (list): a list of columns that need to be kept how they are
    
    Returns:
    df_wide (DataFrame): Dataframe in right structure
    """
    # Transform data to long format
    df_long = pd.wide_to_long(df, stubnames = 'Topic', i = columns, j = 'Onderwerp', sep = '_').reset_index()
    # Transform data to wide format
    df_wide = pd.pivot_table(df_long, index = columns, columns = 'Topic', aggfunc = len, fill_value = 0)
    # make adjustments to indices and reset index
    df_wide.columns = df_wide.columns.droplevel(0)
    # make sure all topic columns are 0-1 encoded
    df_wide = df_wide.applymap(encode_units)
    df_wide = df_wide.reset_index().rename_axis(None, axis=1)
    return df_wide
    
def str_lower_strip(value):
    """Transforms input to str datatype, changes to lowercase and deletes trailing and leading whitespace
    
    Args: 
        value (any type): value to be preprocessed
    
    Returns:
        str_out (str): string after preprocessing
    """
    if value is None:
        str_out = ''
    elif value is np.nan:
        str_out = ''
    else:
        str_out = str(value).lower().strip()
    return str_out
  
def remove_digits(value):
    """Removes digits from string
    
    Args: 
        value (str): string to be preprocessed
    
    Returns:
        str_out (str): string after preprocessing
    """
    str_out = ''.join([i for i in value if not i.isnumeric()])
    return str_out
  
def remove_punctuation(value):
    """Removes punctuation from string
    
    Args: 
        value (str): string to be preprocessed
    
    Returns:
        str_out (str): string after preprocessing
    
    Notes:
        Function uses string package to retrieve all different punctuation marks   
    """
    puncs = set(string.punctuation)
    str_out = ''.join([i for i in value if not i in puncs])
    return str_out
  
def remove_stopwords(value, stopwords=None):
    """Removes stopwords from string
    
    Args: 
        value (str): string to be preprocessed
    
    Returns:
        str_out (str): string after preprocessing
    
    Notes: 
        The function uses the function str_lower_strip() because it only works if all words in the strings are lowercase
    """
    value = str_lower_strip(value)
    if stopwords is None:
        str_out = value
    elif type(stopwords) not in [str, list, set]:
        str_out = value
    else:    
        str_out = ' '.join([i for i in value.split() if i not in stopwords])
    return str_out

def text_cleaning(value, stopwords=None):
    """Applies the four cleaning funtions to a value. 
    Turns value into string, makes lowercase, strips trailing and leading spaces, and removes digits, punctuation, and stopwords
    
    Args: 
        value (str): string to be cleaned
    
    Returns:
        str_out (str): string after cleaning
    """
    value = str_lower_strip(value)
    value = remove_digits(value) 
    value = remove_punctuation(value)
    value = remove_stopwords(value, stopwords)
    str_out = value
    return str_out

def empty_string(value):
    """Indicate if the value is an empty string or not
    
    Args: 
        value (str): string to be checked
    
    Returns:
        out (bool): Boolean that indicates True or False
    """
    value = str_lower_strip(value)
    out = value == ''
    return out
  
def only_digit_punct(value):
    """Indicate if the  value contains only digits or punctuation. 
    True if it only contains digits or punctuation, False if it does not only contain digits or puncutation
        
    Args: 
        value (): value to be checked
    
    Returns:
        out (bool): Boolean that indicates True or False
    
    Notes:
        Function uses string package to retrieve all different punctuation marks   
    """
    value = str_lower_strip(value)
    puncs = set(string.punctuation)
    out = all(j.isnumeric() or j in puncs or j == ' ' for j in value)
    return out
  
def is_word_in_list(value):
    """Checks if value is equal to one of the words/word combinations from a list specified in the function
    True if it is, False if it is not
        
    Args: 
        value (): value to be checked
    
    Returns:
        out (bool): Boolean that indicates True or False
    
    Notes:
        The list of words/word combinations is specified within the function  
    """
    value = str_lower_strip(value)
    value = remove_digits(value) 
    value = remove_punctuation(value)
    value = value.strip()
    list_words = ['ja', 'nee', 'geen mening', 'geen', 'geen opmerkingen', 'niets' 'weet ik niet', 'weet niet', 'geen aanvullingen', 'idem', 'zie vorige', 'n.v.t.', 'nvt', 'niet van toepassing', 'geen idee']
    out = value in list_words
    return out

def is_not_feedback(value):
    """Check if value does not contain feedback. This returns True if one of the three conditions is True:
    1. only empty string
    2. only digits or punctuation
    3. is word in list
    If the function returns True, then this row needs the label 'Is not feedback' and needs to be excluded from the classification model 
    
    Args: 
        value (): value to be checked
    
    Returns:
        out (bool): Boolean that indicates True or False
    
    Notes:
        The list of words/word combinations is specified within the function  
    
    """
    if empty_string(value) or only_digit_punct(value) or is_word_in_list(value):
        out = True
    else:
        out = False
    return out

def lemmatize_text(text, nlp):
    """Lemmatize text 
    
    Args: 
        text (str): string to be lemmatized
        nlp (spacy.lang.nl.Dutch): Dutch language model from Spacy
    
    Returns:
        str_out (str): string after preprocessing
    
    Notes: 
        The function uses the function str_lower_strip() because it only works if all words in the strings are lowercase
    """
    text = str_lower_strip(text)
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    str_out = ' '.join(lemmas)
    return str_out
  