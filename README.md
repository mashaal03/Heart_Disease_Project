# Heart Disease Prediction Project

A comprehensive machine learning pipeline to predict heart disease using the UCI Heart Disease dataset. This project includes data preprocessing, feature selection, model training, hyperparameter tuning, and deployment as a Streamlit web application.

## Features
- **Data Cleaning**: Handles missing values and prepares data.
- **Feature Engineering**: Uses One-Hot Encoding for categorical variables.
- **Modeling**: Implements Logistic Regression, Random Forest, and SVM.
- **Tuning**: Optimizes the best model using GridSearchCV.
- **Deployment**: A live web UI built with Streamlit and deployed via Ngrok.

## How to Run

1.  **Clone the repository:**
    `git clone https://github.com/mashaal03/Heart_Disease_Project.git`
2.  **Create and activate the conda environment:**
    `conda create --name heart_disease_env python=3.9`
    `conda activate heart_disease_env`
3.  **Install dependencies:**
    `pip install -r requirements.txt`
4.  **Run the Streamlit app:**
    `streamlit run ui/app.py`
