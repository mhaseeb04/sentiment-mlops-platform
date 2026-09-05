import os
import re
import sys
import string
import pandas as pd
import numpy as np
from src.exception import CustomerException
from sklearn.metrics import accuracy_score
import nltk
for pkg in ['punkt', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer =  WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

punctuations = string.punctuation
import dill 

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomerException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test,models):
    try:
        report={}
        
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            
            model.fit(X_train, y_train)
            
            y_train_pred= model.predict(X_train)
            
            y_test_pred= model.predict(X_test)
            
            train_model_score = accuracy_score(y_train, y_train_pred)
            
            test_model_score = accuracy_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            print("Done", i)
        return report
    except Exception as e:
        raise CustomerException(e,sys)

def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)       
    
    except Exception as e:
        raise CustomerException(e, sys)


def preprocess_text(text):
        if isinstance(text, pd.Series):
            return text.apply(preprocess_text)
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Removing the punctuation
        text = text.translate(str.maketrans('', '', punctuations))
        # Tokenization
        tokens = word_tokenize(text)
        # lemmatizing
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in punctuations and word.isalpha()]
        return ' '.join(tokens)
