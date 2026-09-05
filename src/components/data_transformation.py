from src.logger import logging
from src.exception import CustomerException
from src.utils import save_obj
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.model_selection import train_test_split
import re
import os
import sys
import string
import nltk
for pkg in ['punkt', 'stopwords', 'wordnet']:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
lemmatizer =  WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

punctuations = string.punctuation
class DataTransformation:
    def __init__(self, raw_file_path="Data/raw/raw_data.csv", output_file_path="Data/processed"):
        self.raw_file_path = raw_file_path
        self.output_file_path = output_file_path
    
    def preprocess_text(self, text):
        if isinstance(text, pd.Series):
            return text.apply(self.preprocess_text)
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
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def preprocess_dataset(self):
        try:
            print("Inside preprocess method")
            df = pd.read_csv(self.raw_file_path)
            
            # Dropping the null values
            df.dropna(inplace=True)
            # Dropping the duplicate values
            df.drop_duplicates(inplace=True)  
            
            df=df[['text', 'sentiment']]        
            
            df['text'] = df['text'].apply(self.preprocess_text)
            
            df = df[df['text'].str.strip().astype(bool)]
            
            train_data, test_data=train_test_split(df, test_size=0.2, random_state=42) 
            
            train_data['sentiment'] = label_encoder.fit_transform(train_data['sentiment'])
            test_data['sentiment'] = label_encoder.transform(test_data['sentiment'])
                        
            # df['sentiment'] = label_encoder.transform(df['sentiment'])
            # df['text'] = df['text'].apply(self.preprocess_text)
            
            os.makedirs("src/model", exist_ok=True)
            save_obj(file_path="src/model/label_encoder.pkl",obj=label_encoder)

            logging.info("The dataset is cleaned now.")
            
            return train_data, test_data
        except Exception as e:
            raise CustomerException(e, sys)

    def save_preprocessd_datasets(self, preprocessed_train_df, preprocessed_test_df):
        try:
            print("Inside the saving preprocess dataset files function")
            preprocessed_train_df.dropna(subset=['text', 'sentiment'], inplace=True)
            preprocessed_test_df.dropna(subset=['text', 'sentiment'], inplace=True)
            
            os.makedirs(self.output_file_path, exist_ok=True)

            preprocessed_train_dataset_file_path=os.path.join(self.output_file_path, "preprocess_train_data.csv")
            preprocessed_train_df.to_csv(preprocessed_train_dataset_file_path, index=False)
            
            preprocessed_test_dataset_file_path=os.path.join(self.output_file_path, "preprocess_test_data.csv")
            preprocessed_test_df.to_csv(preprocessed_test_dataset_file_path, index=False)          
            
            print("Done saving the preprocess dataset files")
            logging.info("Preprocessed dataset files are saved successfully.")
            
        except Exception as e:
            print("Error in saving the preprocess data")
            raise CustomerException(e, sys)
   
        
if __name__=="__main__":
    dt = DataTransformation(raw_file_path="Data/raw/raw_data.csv", output_file_path="Data/processed")
    preprocessed_train_df, preprocessed_test_df=dt.preprocess_dataset()
    dt.save_preprocessd_datasets(preprocessed_train_df, preprocessed_test_df)
