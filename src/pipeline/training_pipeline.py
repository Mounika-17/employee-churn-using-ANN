from src.logger import logger
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

import warnings
warnings.filterwarnings("ignore")
from src.config import continuous_numerical_columns, categorical_encode, target_column

if __name__ == "__main__":
    logger.info("Starting the training pipeline...")
    logger.info("Step 1: Data Ingestion")

    # Step 1: Data Ingestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
    logger.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}")
    logger.info("Step 2: Data Transformation")

    # Step 2: Data Transformation (create preprocessor pipeline)
    data_transformation = DataTransformation() 
    train_array, test_array, _= data_transformation.initiate_data_transformation(train_data_path, test_data_path, continuous_numerical_columns, categorical_encode, target_column)
    logger.info("Data transformation pipeline created.")
    logger.info("Step 3: Model Training")

   # Verify if the data that is being passed to the model trainer is correct
    logger.info(f"Train data path: {train_data_path}")
    logger.info(f"Test data path: {test_data_path}")
  
   # Step 3: Model Training
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_array, test_array)

