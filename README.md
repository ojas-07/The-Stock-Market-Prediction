

---

# Stock Market Prediction with Random Forest Regressor

## Overview

This repository contains a simple Python script for predicting stock prices using a Random Forest Regressor. The code uses the pandas, scikit-learn, and matplotlib libraries for data manipulation, machine learning, and visualization.

## Prerequisites

Make sure you have the following dependencies installed:

- pandas
- scikit-learn
- matplotlib

You can install them using:

pip install pandas scikit-learn matplotlib


## Usage

1. Clone the repository:


git clone https://github.com/your-username/stock-market-prediction.git
cd stock-market-prediction


2. Download the stock data file (tesla-stock-price.csv) and place it in the project directory.(Recommended to use the hello-world.csv file during first execution to train the model)

3. Run the Python script:

bash
python stock_prediction.py


The script loads the stock data, splits it into training and test sets, trains a Random Forest Regressor, and makes a prediction for the next day's stock price. Finally, it generates a line plot comparing the predicted and actual stock prices.

## Files

- stock_prediction.py: Python script for stock market prediction.
- tesla-stock-price.csv: Sample stock price data for Tesla.

## Results

The script generates a plot showing the predicted and actual stock prices, providing insights into the model's performance.

//



---


