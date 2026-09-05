from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
# from src.data_transformation import DataTransformation

if __name__=="__main__":
    di = DataIngestion(input_path='test_data/twitter_dataset.csv',output_path='Data/raw')
    data=di.read_data()
    di.save_raw_data(data)
    di.save_train_test_data(data)
    
    # dt = DataTransformation(raw_file_path="Data/raw/raw_data.csv", output_file_path="Data/processed")
    # preprocessed_train_df, preprocessed_test_df=dt.preprocess_dataset()
    # dt.save_preprocessd_datasets(preprocessed_train_df, preprocessed_test_df)

    # mt = ModelTrainer(model_file_path="src/model",train_data="Data/processed/preprocess_train_data.csv", test_data="Data/processed/preprocess_test_data.csv")
    # acc=mt.initiate_model_training()
    # print("Last",acc)

