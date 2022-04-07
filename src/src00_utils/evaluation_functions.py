# Import packages
import os 
import sys
import time
import numpy as np
import pandas as pd

# Define functions

def divide_zero(value_a, value_b):
    """ 
    Fuction to resolve divide by zero error
    Returns 0 if divided by 0.
    
    Args: 
        value_a (int, float):
        value_b (int, float):
    
    Returns:
        result (int, float): 
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.true_divide( value_a, value_b )
        result[ ~ np.isfinite( result )] = 0 # accept the value 0 when divided by 0
    return result

def calc_PN_matrix(y_test, y_pred):
    """ 
    Creates a matrix with indications for True and False Postives and Negatives
    Useful for multi-label classification.
    Purpose is to compare true labels with predicted label for each observation and each label.
    In the matrix, each row is an observation, and each column is a label
    
    Args:
        y_test (dataframe or array): contains true labels
        y_pred (dataframe or array): contains predicted labels
    
    Returns:
        PN_matrix (numpy array): has same shape as y_test and y_pred
    """
    y_test_array = y_test.to_numpy()
    y_pred_array = y_pred.to_numpy()
    conditions = [
        (y_test_array == 0) & (y_pred_array == 0),    
        (y_test_array == 0) & (y_pred_array == 1),
        (y_test_array == 1) & (y_pred_array == 0),
        (y_test_array == 1) & (y_pred_array == 1)]
    choices = ['TN', 'FP', 'FN', 'TP']
    PN_matrix = np.select(conditions, choices)
    return PN_matrix

def label_based_PN_labels(y_test, y_pred):
    """ 
    Returns arrays with counts of each of the possible classification indications. (TN: True Negative, TP: True Positive, FP: False Positive, FN: False Negative, PP: Predicted Positive, PN: Predicted Negative, RP: Real Positive, RN: Real Negative)
    Useful for multi-label classification.
    Purpose is to make calculations in other functions easier
    
    Args:
        y_test (dataframe or array): contains true labels
        y_pred (dataframe or array): contains predicted labels
    
    Returns: 
        8 lists, one for each indication. Each list contains the counts for the respective indication. Each element in the list represents the count for a label
    
    Notes: 
        Uses the function calc_PN_matrix()
    """
    y_test_array = y_test.to_numpy()
    y_pred_array = y_pred.to_numpy()
    PN_matrix = calc_PN_matrix(y_test, y_pred)
    TN_rows = np.count_nonzero(PN_matrix == 'TN', axis=0)
    TP_rows = np.count_nonzero(PN_matrix == 'TP', axis=0)
    FP_rows = np.count_nonzero(PN_matrix == 'FP', axis=0)
    FN_rows = np.count_nonzero(PN_matrix == 'FN', axis=0)
    PP_rows = np.sum(y_pred_array == 1, axis=0)
    PN_rows = np.sum(y_pred_array == 0, axis=0)
    RP_rows = np.sum(y_test_array == 1, axis=0)
    RN_rows = np.sum(y_test_array == 0, axis=0)
    return TN_rows, TP_rows, FP_rows, FN_rows, PP_rows, PN_rows, RP_rows, RN_rows


def label_metrics(y_test, y_pred, labels):
    """
    Returns a dataframe that returns evaluation metrics (accuracy, precision, recall, F1-score) for all labels. 
    Useful for multi-label classification.
    
    Args:
        y_test (dataframe or array): contains true labels
        y_pred (dataframe or array): contains predicted labels
        labels (list): list of labels
    
    Returns: 
        Dataframe: The rows are the performance metrics, the columns are the labels
    
    Notes: 
        Uses the functions label_based_PN_labels() and divide_zero()
    """
    TN_rows, TP_rows, FP_rows, FN_rows, PP_rows, PN_rows, RP_rows, RN_rows = label_based_PN_labels(y_test, y_pred)
    accuracy = (TP_rows + TN_rows) / y_pred.shape[0]
    precision = divide_zero(TP_rows, PP_rows) 
    recall = divide_zero(TP_rows, RP_rows) 
    F1_score = 2*TP_rows / (PP_rows + RP_rows)
    data_dict = {'Accuracy': accuracy, 'Precision': precision, 'Recall':recall, 'F1_Score':F1_score}
    return pd.DataFrame(data=data_dict.values(), index=data_dict.keys(), columns=labels)

def confusion_matrix_label(y_test, y_pred, labels):
    """
    Returns dictionary of confusion matrices for each label, provided in labels
    
    Args:
        y_test (dataframe or array): contains true labels
        y_pred (dataframe or array): contains predicted labels
        labels (list): list of labels
        
    Returns:
        conf_mat_dict (dict): a matrix containing pairs of label:confusion matrix
    """
    conf_mat_dict = {}
    for label_col in labels:
        conf_mat_dict[label_col] = pd.crosstab(y_test[label_col], y_pred[label_col], rownames=['Actual'], colnames=['Predicted'], margins=True)
    return conf_mat_dict
