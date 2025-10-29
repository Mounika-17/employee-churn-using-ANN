# 🧠 ANN Employee Churn Classification Project

A deep learning project using Artificial Neural Networks (ANN) to predict customer churn based on demographic and financial data.

## 📂 Project Structure

ANN-CLASSIFICATION/
│
├── artifacts/
│   ├── data.csv
│   ├── train.csv
│   ├── test.csv
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── logs/
│   └── (log files generated during data processing, training & prediction)
│
├── notebooks/
│   └── Employee_Churn_Exploration.ipynb
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── predict_pipeline.py
│   │
│   ├── __init__.py
│   ├── config.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
│
├── .gitignore
├── app.py
├── README.md
├── requirements.txt
└── setup.py

🧩 Explanation of Key Folders & Files

| Path               | Description                                                                      |
| ------------------ | -------------------------------------------------------------------------------- |
| `artifacts/`       | Stores saved model (`model.h5`) and preprocessing pipeline (`preprocessor.pkl`). |
| `logs/`            | Contains generated log files for debugging and monitoring.                       |
| `notebooks/`       | Used for Jupyter notebooks — exploratory data analysis, experiments, etc.        |
| `src/components/`  | Contains core modules for data ingestion, transformation, and model training.    |
| `src/pipeline/`    | Includes training and prediction pipeline scripts.                               |
| `src/utils.py`     | Helper functions for loading/saving models, pickles, etc.                        |
| `src/exception.py` | Custom exception class for unified error handling.                               |
| `src/logger.py`    | Logging configuration for the project.                                           |
| `app.py`           | Streamlit app file for user interaction and model prediction.                    |
| `requirements.txt` | Python package dependencies.                                                     |
| `setup.py`         | Makes the project installable as a Python package.                               |



## 🚀 How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 🧩 Tech Stack
- Python, TensorFlow, scikit-learn
- Streamlit for UI
- Modular structure (src/, models/, notebooks/)


