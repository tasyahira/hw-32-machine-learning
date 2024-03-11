from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# List of n_estimators to try
n_estimators_list = [50, 100, 200]

# Variables to store best model and metrics
best_model = None
best_rmse = float('inf')
best_mae = float('inf')

# Loop through different n_estimators
for n_estimators in n_estimators_list:
    # Create Random Forest model
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

    # Train the model
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Calculate RMSE and MAE
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)

    # Check if current model has lower error
    if rmse < best_rmse and mae < best_mae:
        best_rmse = rmse
        best_mae = mae
        best_model = rf_model

    # Print results for current n_estimators
    print(f"Results for n_estimators={n_estimators}:")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print("-" * 30)

# Print results for the best model
print("Best Model:")
print(f"RMSE: {best_rmse}")
print(f"MAE: {best_mae}")
