# Customer Churn Prediction with Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.11.7-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-latest-red.svg)](https://streamlit.io/)

## Overview

This project implements machine learning solutions using Artificial Neural Networks (ANN) with two primary objectives:
1. **Churn Prediction (Classification)**: Predict which customers are likely to leave a bank
2. **Salary Estimation (Regression)**: Predict a customer's estimated salary based on their banking profile

Both models help businesses understand their customers better, enabling targeted retention strategies and personalized service offerings.

## Features

- **Data Preprocessing**: Handles categorical variables, scaling, and feature engineering
- **Neural Network Models**: 
  - Classification model for churn prediction
  - Regression model for salary estimation
- **Interactive Web App**: User-friendly Streamlit interface for making predictions
- **Performance Monitoring**: Uses TensorBoard for visualization of training metrics

## Dataset

The model is trained on the "Churn_Modelling.csv" dataset containing the following customer information:
- Customer ID and demographic data
- Banking relationship details (tenure, balance, products)
- Credit score and estimated salary
- Target variable: "Exited" (1 if customer left the bank, 0 if stayed)

## Model Architectures

### Classification Model (Churn Prediction)
The neural network consists of:
- Input layer: Matches the feature dimensions
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron with Sigmoid activation for binary classification

### Regression Model (Salary Estimation)
The neural network consists of:
- Input layer: Matches the feature dimensions
- Hidden layer 1: 64 neurons with ReLU activation
- Hidden layer 2: 32 neurons with ReLU activation
- Output layer: 1 neuron (linear activation) for salary prediction

## Getting Started

### Prerequisites

- Python 3.11.7
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ANN-Classification-Churn.git
   cd ANN-Classification-Churn
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit web application:
```bash
streamlit run app.py
```

## Project Structure

- `app.py` - Streamlit web application
- `experiments.ipynb` - Jupyter notebook with classification model development
- `churnRegression.ipynb` - Jupyter notebook with regression model development
- `prediction.ipynb` - Notebook demonstrating model inference
- `model.h5` - Saved trained classification model
- `regression_model.h5` - Saved trained regression model
- `*.pkl` - Saved preprocessors (encoders and scaler)
- `Churn_Modelling.csv` - Dataset
- `requirements.txt` - Project dependencies
- `logs/` - TensorBoard logs for classification model
- `regressionlogs/` - TensorBoard logs for regression model

## Usage

### Churn Prediction (Classification)
1. Access the web application
2. Enter customer information (demographics, banking details, etc.)
3. Get prediction on whether the customer is likely to churn

### Salary Estimation (Regression)
The regression model for salary estimation is implemented in `churnRegression.ipynb`. To use:
1. Open the notebook in Jupyter or VS Code
2. Execute the cells to train the model or load the pre-trained model
3. Input customer data to get salary predictions

### Model Performance
- **Classification Model**: Evaluated using accuracy and binary cross-entropy
- **Regression Model**: Evaluated using Mean Absolute Error (MAE)

Both models include TensorBoard integration for detailed performance monitoring and visualization.

## License

This project is licensed under the terms of the LICENSE file included in the repository.

## Acknowledgments

- The dataset used is a modified version of a publicly available banking dataset
- Built with TensorFlow, scikit-learn, pandas, and Streamlit