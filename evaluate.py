from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd
from train import predictions, y_test

# Load the trained model
# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test, predictions)
print("R-squared (R2):", r2)
