import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

# Generate a synthetic sales dataset
np.random.seed(0)
n_samples = 1000
X = np.random.rand(n_samples, 2) * 10  # Two features
y = 3 * X[:, 0] + 2 * X[:, 1] + 1 + np.random.randn(n_samples)

# Create a DataFrame for easier data manipulation
data = pd.DataFrame(data={'Feature1': X[:, 0], 'Feature2': X[:, 1], 'Sales': y})

# Data preprocessing
X = data[['Feature1', 'Feature2']]
y = data['Sales']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model selection and hyperparameter tuning using pipelines
models = {
    'Lasso Regression': Lasso(),
    'Ridge Regression': Ridge(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor()
}

best_model = None
best_score = -float('inf')

for model_name, model in models.items():
    if model_name in ['Lasso Regression', 'Ridge Regression']:
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
        regressor = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        regressor = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)
    elif model_name == 'Gradient Boosting':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        regressor = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5)

    # Feature selection using SelectKBest with f_regression
    feature_selector = SelectKBest(score_func=f_regression, k=2)
    pipeline = Pipeline([('feature_selector', feature_selector), ('regressor', regressor)])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    if score > best_score:
        best_score = score
        best_model = pipeline

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
explained_variance = explained_variance_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Best Model:", best_model.named_steps['regressor'].best_estimator_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)
print("Explained Variance Score:", explained_variance)
print("Mean Absolute Error:", mae)

# Cross-validation for the best model
cv_scores = cross_val_score(best_model, X_scaled, y, scoring='neg_mean_squared_error', cv=5)
cv_rmse = np.sqrt(-cv_scores)

print("Cross-Validation RMSE:", cv_rmse)

# Predict sales for new data
new_data = np.array([[5.0, 3.0]])
new_data_scaled = scaler.transform(new_data)
predicted_sales = best_model.predict(new_data_scaled)
print("Predicted sales for new data:", predicted_sales[0])

