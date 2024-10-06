# forecasting-electricity-prices


## Project Overview
This project aims to forecast electricity prices by analyzing historical data on energy generation, consumption, and weather conditions. The key approach involves preprocessing the data, performing dimensionality reduction, and leveraging advanced deep learning models like LSTM (Long Short-Term Memory) and Stacked LSTM to predict future electricity prices.

## Key Objectives:
Preprocess energy and weather data by handling categorical features and scaling numerical features.

Reduce dimensionality using Principal Component Analysis (PCA).

Develop predictive models using LSTM and Stacked LSTM neural networks to forecast electricity prices.

## Data Sources
1. Energy Data (energy_dataset.csv):

2. Weather Data (weather_features.csv):

1. Handling Categorical Data:
   
3. Handling outliers

The weather_main column was encoded using One-Hot Encoding to convert the categorical weather conditions into a numerical format that can be fed into the model.

## 5. Scaling:
Numerical features were normalized using Min-Max Scaling to bring all values into a similar range, typically between 0 and 1. This step is crucial for improving model performance, especially in neural networks.
## 6. Dimensionality Reduction:
To reduce the complexity of the dataset and avoid overfitting, Principal Component Analysis (PCA) was applied. The number of components was selected to retain 75% of the variance in the data. This technique helped in reducing the dimensionality of the feature space while maintaining the key information.

## Modeling Approach
## 1. LSTM (Long Short-Term Memory) Model:
LSTM is a type of recurrent neural network (RNN) designed to capture temporal dependencies in time series data.

The model was trained to forecast electricity prices using the energy generation data, electricity demand, and the preprocessed weather features.


LSTM effectively captures patterns over time by remembering long-term dependencies in the data.

## 2. Stacked LSTM:
A Stacked LSTM model is a more advanced variant, where multiple LSTM layers are stacked together. This architecture allows the model to capture even deeper patterns in the data by passing the outputs of one LSTM layer as inputs to the next.
The Stacked LSTM model was trained using the same preprocessed dataset and compared with the regular LSTM model to assess improvements in prediction accuracy.
Model Training and Evaluation


Comparison: The LSTM and Stacked LSTM models were compared to determine which architecture provided better forecasting accuracy.

## Installation and Usage
## Requirements:
Python 3.x

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

tensorflow or keras (for LSTM models)



Results
The LSTM and Stacked LSTM models successfully learned patterns in the data and provided reasonable forecasts of electricity prices.

The dimensionality reduction using PCA allowed the model to process data more efficiently without sacrificing too much predictive power.

## Future Work
Model Enhancement: Further fine-tuning of the model hyperparameters, such as learning rate, number of layers, and sequence length, could yield even better results.


