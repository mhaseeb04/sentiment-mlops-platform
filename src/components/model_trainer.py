import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score
from src.exception import CustomerException
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from src.logger import logging
from src.utils import save_obj, evaluate_models
from sklearn.feature_extraction.text import TfidfVectorizer

class ModelTrainer:
    def __init__(self, model_file_path,train_data, test_data):
        self.model_file_path=model_file_path
        self.train_data=train_data
        self.test_data=test_data
    
    def initiate_model_training(self):
        try:
            print("Inside the model training function...")
            
            vectorizer=TfidfVectorizer(stop_words='english', ngram_range=(1,2),min_df=2, max_df=0.9195617995240445)
            train_df=pd.read_csv(self.train_data)
            test_df=pd.read_csv(self.test_data)
            
            X_train_text=train_df['text']
            X_test_text=test_df['text']

            y_train=train_df['sentiment']
            y_test=test_df['sentiment']
            
            print("Vectorization start ...")
            X_train=vectorizer.fit_transform(X_train_text)
            X_test=vectorizer.transform(X_test_text)
            print("Done vectorization")
            
            os.makedirs("src/model", exist_ok=True)
            save_obj(file_path="src/model/vectorizer.pkl", obj=vectorizer)

            models = {
                    "Logistic Regression": LogisticRegression(max_iter=2000),
                    "Support Vector Machine": LinearSVC(),
                    "Naive Bayes": MultinomialNB(),
                    # "Random Forest": RandomForestClassifier(n_estimators=200, random_state=2)
            }
            
            print("Start applying multiple models")
            model_report=evaluate_models(X_train, y_train, X_test, y_test, models)
           
            best_model_score=max(sorted(model_report.values()))
            
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomerException("No best model found.")
            
            save_obj(
                file_path="src/model/model.pkl", obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            
            acc = accuracy_score(y_test, predicted)
            
            return acc


            # model = LinearSVC(C=1.9693974310581017, random_state=42)
            # print("Fitting the model on training data...")
            # model.fit(X_train, y_train)
            # y_pred = model.predict(X_test)
            # acc = accuracy_score(y_test, y_pred)
            # print("Accuracy:", acc)
        
        except Exception as e:
            raise CustomerException(e,sys)

if __name__ =="__main__":
    mt = ModelTrainer(model_file_path="src/model",train_data="Data/processed/preprocess_train_data.csv", test_data="Data/processed/preprocess_test_data.csv")
    acc=mt.initiate_model_training()
    print("Last",acc)
