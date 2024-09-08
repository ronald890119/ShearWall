import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# load dataset
file_path = '/Users/ziminli/Desktop/USYD/COMP5703 Capstone/Dataset/ConcreteStrengthData (CC0- Public Domain).csv'
data = pd.read_csv(file_path)

# Split feature (X) and target variable (y)
X = data.drop(columns=['Strength'])
y = data['Strength']

# Divide the data set into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train gradient lift regression models
gbr = GradientBoostingRegressor(learning_rate = 0.05, n_estimators = 500, max_depth = 5, random_state = 42)
gbr.fit(X_train, y_train)

# Predictive training set
y_train_pred = gbr.predict(X_train)

# Predictive test set
y_test_pred = gbr.predict(X_test)

# Calculate the mean square error of the model (Training set)
train_mse = mean_squared_error(y_train, y_train_pred)
print(f"Training Mean square error (MSE): {train_mse}")

# Calculate R-squared (R^2) (Training set)
train_r2 = r2_score(y_train, y_train_pred)
print(f"Training R-squared (R^2): {train_r2}")

# Calculate the mean square error of the model (Test set)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"Test Mean square error (MSE): {test_mse}")

# Calculate R-squared (R^2) (Test set)
test_r2 = r2_score(y_test, y_test_pred)
print(f"Test R-squared (R^2): {test_r2}")
