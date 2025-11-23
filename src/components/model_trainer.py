import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import datetime
import os
from src.logger import logger
from src.logger import get_logger
from dataclasses import dataclass
from src.components.model import build_ann_model
import warnings
warnings.filterwarnings("ignore")
from src.exception import CustomException
from src.utils import save_object

# 1. Preprocessing (Outliers + Encoding + Scaling)

# Keras model (Sequential) only accepts pure numeric NumPy arrays or tensors.You first preprocess manually (fit_transform / transform) and then train using the processed data.
# model.fit --> it does not perform any preprocessing like build_pipeline() or scaling/encoding.
# From the data_transformation file, we get the processed data of train and test data. 

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.h5")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

# 2. Build and Train ANN Model

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logger.info("Splitting train/test arrays")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Get unified run-specific logger
            log_subdir_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_final"
            run_logger, log_dir = get_logger(os.path.join("fit", log_subdir_name))

            run_logger.info("Starting ANN model training...")

            model = build_ann_model(X_train.shape[1])

            tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
            # patience=10 means that the model will stop training if it doesn't improve for 10 epochs
            # monitor='val_loss' means that the model will stop training if the validation loss doesn't improve
            # restore_best_weights=True means that the model will restore the weights of the best epoch
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train the model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=100,
                batch_size=32,
                callbacks=[tensorboard_callback, early_stopping],
                verbose=1
            )

            run_logger.info("Training completed successfully.")
            run_logger.info(f"TensorBoard logs and model logs saved under: {log_dir}")

            y_pred_prob = model.predict(X_test).flatten()
            y_pred = (y_pred_prob > 0.5).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            run_logger.info(f"Model Accuracy: {accuracy:.4f}")

            run_logger.info("Saving trained model...")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )

            run_logger.info("Model saved successfully.")

        except Exception as e:
            raise CustomException(e, sys)
