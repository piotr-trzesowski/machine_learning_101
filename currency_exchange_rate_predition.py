import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import yfinance as yf
import matplotlib.pyplot as plt

# Fetch EUR/USD exchange rate data
data = yf.download("EURUSD=X", start="2020-01-01", end="2023-12-31")
data = data[['Close']].rename(columns={'Close': 'EURUSD'})

# Create features (lagged values)
for lag in [1, 2, 3, 5, 10]:
    data[f'lag_{lag}'] = data['EURUSD'].shift(lag)

# Add moving averages
data['ma_7'] = data['EURUSD'].rolling(7).mean()
data['ma_21'] = data['EURUSD'].rolling(21).mean()

# Drop NA values and prepare X, y
data = data.dropna()
X = data.drop('EURUSD', axis=1)
y = data['EURUSD']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"RMSE: {np.sqrt(mse):.5f}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, predictions, label='Predicted', alpha=0.7)
plt.title("EUR/USD Exchange Rate Prediction")
plt.xlabel("Date")
plt.ylabel("Exchange Rate")
plt.legend()
plt.show()