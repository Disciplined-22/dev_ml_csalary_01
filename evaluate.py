from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

# Load the test data
# You may need to adjust this part to load your test data
# For example, if your test data is stored in a CSV file, you can load it using pandas
# test_data = pd.read_csv('test_data.csv')

# Assuming you have test data stored in variables X_test and y_test
# Replace X_test and y_test with your actual test data
# Predict using the trained model
predictions = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error (MSE):", mse)

# Calculate R-squared (coefficient of determination)
r2 = r2_score(y_test, predictions)
print("R-squared (R2):", r2)
