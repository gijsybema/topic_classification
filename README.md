# Topic Classification Project

Goal: identify which topics are mentioned in texts

## Directory structure

```
├── README.md          <- The top-level README for developers using this project.
│
├── data
│   ├── 01_raw         <- Imutable input data
│   │   ├── 01-datadumps
│   │   ├── 02-unlabeled_data
│   │   └── 03-labeled_data
│   │
│   ├── 02_intermediate<- Restructured version of raw: Stores training & validation data
│   ├── 03_processed   <- The data used for modelling, e.g. cleaned training & validation data
│   └── 04_external    <- external data
│
├── notebooks          <- Jupyter notebooks
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── results
│   ├── figures        <- Figures
│   └── reports        <- Reports
│      
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── .gitignore         <- Avoids uploading data, credentials, outputs, system files etc
│
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── src00_utils      <- Functions used across the project
    │   └── data_cleaning.py
    │
    ├── src01_data       <- Scripts to reading and writing data etc.
    │    ├── 01-create_unlabeled_data       <- scripts to create unlabeled dataset
    │    └── split_train_validation_data.py <- script to split raw labeled data into train & validation sets
    │
    │
    ├── src02_intermediate <- Intermediate analyses
    │   └── 
    │
    ├── src03_cleaning <- Scripts to turn intermediate data into modelling input
    │   └── 
    │
    ├── src04_modelling  <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   └── 
    │
    ├── src05_model_evaluation<- Scripts that analyse model performance and model selection
    │   └── 
    │    
    ├── src06_reporting  <- Scripts to produce reporting tables
    │   └── 
    │
    └── src07_visualization<- Scripts to create frequently used plots
        └── 
```

## Usage

For each topic, a separate classification model should be trained.
In the end, models with a satisfactory performance can be put into production.

### 1 - Data. Creating training and validation datasets

#### Step 1a. Store labeled data under `data/01-raw/03-labeled_data`

#### Step 1b. Split data in training and validation data.  
In shell: run following command with 'labeled_data_source' replaced with .csv file with labeled data.   
Note: `split_train_validation.py` only takes semicolon-separated (;) files
```
$ python3 src/src01_data/split_train_validation_data.py labeled_data_source
```
This results in two output files, that will be stored under `data/02-intermediate` as `training_data.csv` and `validation_data.csv`

`training_data.csv` will be used to perform cross-validation and model training. 
`validation_data.csv` will be used as a holdout set, to evaluate the final performance of the models

### 2 - Intermediate (optional). Save the counts of each topic in the training and validation data.  
In shell: run following commands:  
```
$ python3 src/src02_intermediate/count_topics.py train
```
and
```
$ python3 src/src02_intermediate/count_topics.py validation
```
Note: count_topics.py only takes arguments 'train' and 'validation'. For other purposes, rewrite the code.  
   
The results will be stored under `results/reports` as `train_topic_counts.csv` and `validation_topic_counts.csv`

### 3 - Cleaning: Clean data
Before we can train and test models, we need to clean the training & validation sets.  
Note: We clean after splitting into train & validation sets, to ensure both datasets stay independent of each other. 

#### Step 3a. Clean train and validation data
Several cleaning steps will be performed:
- change to lowercase
- remove digits and punctuation
- remove stopwords

Then, rows that do not contain feedback will be deleted:
- feedback text that is only an empty string
- feedback text that contains only digits and/or punctuations
- feedback text that occurs in pre-specified list of words/word combinations that do not contain feedback

In shell: run following commands:  
```
$ python3 src/src03_cleaning/clean.py train
```
and
```
$ python3 src/src03_cleaning/clean.py validation 
```
Note: count_topics.py only takes arguments 'train' and 'validation'. For other purposes, rewrite the code.  

Results will be stored under `data/03-cleaned` as `train_clean.csv` and `validation_clean.csv`

### 4 - Modelling. Training the models for each topic
Once data is cleaned, we can train the models.
For each topic, we are training a separate model. 
The model predicts whether the particular topic occurs in a text or not.

Results will be saved in a subfolder for each topic under `/home/cdsw/model_selection/{topic}`.  
Model training is separated in several different .py files
The user is expected to provide input for each stage

To make input and output of the scripts more user-friendly, topics have been assigned a shorter name. Subtopics have an abbreviation of the main topic as prefix.

|Topic short name | Topic |
|---|---|
|afhandeling_proces |Afhandeling & Processen |
|algemeen |Algemene ervaring|
|dig_account | Account & login web|
|dig_contact_web | Contact via web|
|dig_func_web | Functionaliteit web|
|dig_gebruik_web | Gebruiksgemak web|
|digitaal |Digitale mogelijkheden|
|houding_gedrag | Houding & Gedrag medewerker|
|info | Informatievoorziening|
|kennis_vaardigheden | Kennis & Vaardigheden medewerker|
|kv_advies | Kwaliteit van advies medewerker|
|kv_deskundig | Deskundigheid medewerker|
|prijs_kwaliteit | Prijs & Kwaliteit producten|
|telefoon_contact |Contact leggen met medewerker|  

Note: when topic tree is revisioned, the dictionaries in the .py scripts might also need to be revisioned. 

#### Model evaluation strategy
In step 1 we have already splitted the data into a train and validation set. This validation set acts as a final holdout set for the model evaluation.   
To do the model selection, we also need to split the train data created in step 1 into a train and test set. Note that the second train data is a subselection of the original train data.  
In this way, we can estimate the expected performance on new data based on the performance on the test set.  
The validation set acts as a final dataset to validate the expected performance.

#### Step 4a. Cross-validation on default models to determine promising models
In this step, we do cross-validation on all specified algorithm-vectorizer combinations to find the most promising models.

This consist of a couple of steps:
1. Split the data into train and test set.  
    It is necessary to do this before any model selection. We want the test data to remain unseen before model evaluation
2. Run all possible algorithm-vectorizer combinations (with default hyperparameters). 
3. Save cross-validation results to file `01_default_cv_results_{topic}.txt`

For this step, use the script `01-default_cross_validation.py`.  
User should provide script with shortcut name of topic.  
Run in shell example:
```
$ python3 src/src04_modelling/01-default_cross_validation.py telefoon_contact
```

Results will be stored under `results/model_selection/{topic}` as `01_default_cv_results_{topic}.txt`

Based on results, user should choose a maximum of 5 most promising models.
A promising model has a high mean F1 Score and a low standard deviation of F1 Scores

#### Step 4b. Nested cross-validation on most promising models to determine best performing model
In the previous step we have determined the most promising models.
In this step, we do nested cross-validation on the most promising models, to determine the best performing model

This consists of a couple of steps:
1. Run nested cross-validation over most promising models
2. Save nested cross-validation results to file `02_nested_cv_results_{topic}.txt`

In the table below, we have specified the 
|Model short name | Model |
|---|---|
| nb_count | Naive Bayes CountVec|
| nb_tfidf | Naive Bayes Tfidf|
| lr_count | LogisticRegression CountVec|
| lr_tfidf | LogisticRegression Tfidf|
| rf_count | Random Forest CountVec|
| rf_tfidf | Random Forest Tfidf|
| svm_count | SVM CountVec|
| svm_tfidf | SVM Tfidf|
| ada_count | Adaboost Countvec|
| ada_tfidf | Adaboost Tfidf|
| gb_count | GradientBoosting Countvec|
| gb_tfidf | GradientBoosting Tfidf|

For this script we use the script `02-nested_cross_validation.py`
User should provide script with shortcut name of topic & maximum of 5 shorcut names of most promising models:
Run in shell example:
```
$ python3 src/src04_modelling/02-nested_cross_validation.py telefoon_contact nb_count nb_tfidf lr_count
```

Results will be stored under `results/model_selection/{topic}` as `02_nested_cv_results_{topic}.txt`

Based on results, user should choose best and most stable performing model

#### Step 4c. Hyperparamter tuning and training final model
    - Hyperparameter tuning
    - Save hyperparameter tuning to file
    - Print most important variables for best model
    - Train final model on whole training data

User should provide script with shortcut name of topic & name of best performing model:

Run in shell example:
```
$ python3 src/src04_modelling/03-hyperparameter_tuning.py telefoon_contact nb_count
```

Results will be stored under `results/model_selection/{topic}` as `03_hyperparameter_tuning_results_{topic}.txt`
The final model will be stored under `models/final_models` as `final_model_{topic}.pkl` 

### Repeat step 4 for all topics and subtopics
For each topic we create a separate model.   

Can use Bash scripting to run each step for all topics simulateously. In this way, manual shell commands are reduced to a minimum.

#### 4a. Cross-validation on default models for all topics simultaneously:

Run bash script in shell:
``` 
bash src/src04_modelling/01_bash_default_cross_validation.sh
```

After this step: 
- Review the results to find most promising models
- Modify the file `02_bash_nested_cross_validation.sh` to include the most promising models for each topic.

#### 4b. Nested Cross-validation on promising models for all topics simultaneously:
Make sure the file `02_bash_nested_cross_validation.sh` is modified so it includes the right arguments (most promising models) for each topic

Run bash script in shell:
``` 
bash src/src04_modelling/02_bash_nested_cross_validation.sh
```

After running the bash script:
- Review the result to find best performing model
- Modify the file `src/src04_modelling/03_bash_hyperparameter_tuning.sh` to include the best performing model for each topic

#### 4c. Nested Cross-validation on promising models for all topics simultaneously:
Make sure the file `03_bash_nested_cross_validation.sh` is modified so it includes the right argument (best performing model) for each topic

Run bash script in shell:
``` 
bash src/src04_modelling/03_bash_hyperparameter_tuning.sh
```

### 5 - Model Evaluation. Evaluation of a model on validation set
Once we are satisfied with the performance of the model on the test set, we want to evaluate the model on new unseen data.   
This unseen labeled data is stored in the validation set (created in step 1b)

#### Evaluate models separately
NOT YET DONE

#### Evaluate all models at once

Use `model_validation.py` in src/src05_model_evaluation/

Run .py script in shell:
```
$ python3 src/src05_model_evaluation/model_validation.py
```

The results will be stored in two separate files under `results/model_evaluation`
1. The first file contains the most important performance metrics for each topic in a table `model_evaluation_results_validation_set.csv`
2. The second file contains the confusion matrix for each topic in a text file `Evaluation_confusion_matrices.txt`

### 6. Prediction on new data. Put together all satisfactory models
After evaluation, we can decide which topics have a model with satisfactory performance.  
For those topics we want to predict on new texts

We put together all satisfactory models, using an approach similar to the Binary Relevance approach.  
In this way, we can use all  models at once to predict on new data.  

The final script will add columns for each topic that has a satisfactory model whether it occurs in the text.

Use `predict.py` in src/src06_prediction/
The script requires positional arguments for which topics to predict. It only accepts the specified abbrevations of the topics.

Run in shell example:
```
$ python3 src/src05_model_evaluation/model_validation.py afhandeling_proces prijs_kwaliteit
```

NOTE: The datafile and column on which to predict is specified in the script. To predict on other data or columns, the script would have to be adjusted.

## Jobs
TO DO
