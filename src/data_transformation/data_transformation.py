from src.logger import logging
from src.exception import CustomerException
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
from sklearn.model_selection import train_test_split
import re
import os
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer =  WordNetLemmatizer()
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

punctautions = string.punctuation
class DataTransformation:
    def __init__(self, raw_file_path="Data/raw/raw_data.csv",
                 raw_train_file_path="Data/raw/raw_train.csv",
                 raw_test_file_path="Data/raw/raw_test.csv",
                 output_file="Data/processed"):
        self.raw_file_path = raw_file_path
        self.raw_train_file_path = raw_train_file_path
        self.raw_test_file_path = raw_test_file_path
        self.output_file = output_file
    
    

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
        text = text.translate(str.maketrans('', '', punctautions))

        # Tokenization
        tokens = word_tokenize(text)

        # stemming
        stemmer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return ' '.join(tokens)

    def preprocess_dataset(self, file_path):
        print("Inside preprocess method")
        df = pd.read_csv(file_path)
        
        # Dropping the null values
        df.dropna(inplace=True)
        # Dropping the duplicate values
        df.drop_duplicates(inplace=True)
        # Label encoding the sentiment column
        df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
        # Cleaning the text
        df['text'] = df['text'].apply(self.preprocess_text)
        
        return df

    def save_preprocess_dataset(self, preprocessed_df):
        print("Inside the saving preprocess dataset function")
        os.makedirs(self.output_file, exist_ok=True)
        preprocessed_file_path=os.path.join(self.output_file, "preprocess_data.csv")
        preprocessed_df.to_csv(preprocessed_file_path)
        print("Done saving the preprocess data")
    
    def save_train_test_preprocess_dataset(self,raw_train_file_path,raw_test_file_path  ):
        pass
        
        
# if __name__=="__main__":
#     dt = DataTransformation(raw_file_path="Data/raw/raw_data.csv",
#                  raw_train_file_path="Data/raw/raw_train.csv",
#                  raw_test_file_path="Data/raw/raw_test.csv")
#     preprocess_data=dt.preprocess_dataset("Data/raw/raw_data.csv")
#     dt.save_preprocess_dataset(preprocess_data)
    
#     train_preprocess_data = dt.preprocess_dataset("Data/raw/raw_train.csv")
#     dt.train_preprocess_data(preprocess_data)
    

