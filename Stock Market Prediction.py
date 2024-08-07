import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the stock data
df = pd.read_csv('tesla-stock-price.csv', index_col='date', parse_dates=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df[['close']], df['high'], test_size=0.25, random_state=42)

# Create the machine learning model
model = RandomForestRegressor()

# Train the model
model.fit(X_train, y_train)

# Make a prediction for the next day's stock price 

next_day_prediction = model.predict(X_test)

# Print the prediction
formatted_number = f'{next_day_prediction[0]:.2f}'
print('Predicted stock price for next day:', formatted_number)

# Load the stock data
df = pd.read_csv('tesla-stock-price.csv', index_col='date', parse_dates=True)

# Create a line plot of the predicted and actual stock prices
plt.plot(df['close'], label='Predicted Close')
plt.plot(df['high'], label='Actual Close')

# Set the labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Stock Market Prediction')

# Show the plot
plt.legend()
plt.show()