import sys
import os
import pandas as pd
from src.exception import CustomerException
from src.utils import preprocess_text
from src.utils import load_obj

class PredictPipeline:
    def __init__(self, label_encoder_path="src/model/label_encoder.pkl", vectorizer_path="src/model/vectorizer.pkl", model_path="src/model/model.pkl" ):
        self.label_encoder_path=label_encoder_path
        self.vectorizer_path=vectorizer_path
        self.model_path=model_path
    
    def predict(self, text):
        
        model = load_obj(file_path=self.model_path)
        vectorizer = load_obj(file_path=self.vectorizer_path)
        label_encoder = load_obj(file_path= self.label_encoder_path)
        
        data_processed = preprocess_text(text)
        data = vectorizer.transform([data_processed])
        
        y_pred = model.predict(data)
        
        decoded_label = label_encoder.inverse_transform(y_pred)
        
        print("Prediction: ", decoded_label[0])
    
if __name__=="__main__":
    sentences = [
    "I love this new feature! It’s really helpful and easy to use.",
    "This update ruined everything. Totally useless.",
    "The event starts at 5 PM according to the schedule.",
    "I don’t understand what this tweet is talking about.",
    "Not bad, but it could be better next time."
    ]
    pp=PredictPipeline()
    
    for text in sentences:
        print(f"Text: {text}")
        pp.predict(text)
        print()
    