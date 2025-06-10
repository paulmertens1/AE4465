import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

from import_data import df_train, df_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

################

# Data voorbereiding

################


df_train['max_cycle'] = df_train.groupby('engine')['cycle'].transform('max')
df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
df_train['RUL'] = df_train['RUL'].clip(upper=125)
df_train.drop(columns='max_cycle', inplace=True)

feature_cols = [
    'T24', 'T30', 'T50',
    'P30',
    'Nf', 'Nc', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR',
    'htBleed',
    'W31', 'W32'
]

feature_cols_test = ['Nc', 'NRc', 'Ps30', 'T24','T30', 'T50','P30']

df_test_last = df_test.groupby('engine').tail(1).copy()
X_train = df_train[feature_cols]
y_train = df_train['RUL']

true_rul = pd.read_csv(
    'CMAPSSData/RUL_FD001.txt',
    header=None
)
df_test_last['RUL'] = true_rul.values.flatten()

X_test = df_test_last[feature_cols]
y_test = df_test_last['RUL']

############

# Models

############

# Single Decision Tree Regressor
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)


# Radnom Forest Regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, min_samples_leaf=5, n_jobs = -1)
rf.fit(X_train, y_train)

# XGBoost Regressor
xg = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=6,random_state=42)
xg_fit = xg.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
rf_pred = rf.predict(X_test)
xg_pred = xg_fit.predict(X_test)

###############

# Evaluatie modellen

###############
rmse_dt = np.sqrt(mean_squared_error(y_test, dt_pred))
mae_dt = mean_absolute_error(y_test, dt_pred)
r2_dt = r2_score(y_test, dt_pred)
print(f'Decision Tree - RMSE: {rmse_dt}, MAE: {mae_dt}, R2: {r2_dt}')
rmse_rf = np.sqrt(mean_squared_error(y_test, rf_pred))
mae_rf = mean_absolute_error(y_test, rf_pred)
r2_rf = r2_score(y_test, rf_pred)
print(f'Random Forest - RMSE: {rmse_rf}, MAE: {mae_rf}, R2: {r2_rf}')
rmse_xg = np.sqrt(mean_squared_error(y_test, xg_pred))
mae_xg = mean_absolute_error(y_test, xg_pred)
r2_xg = r2_score(y_test, xg_pred)
print(f'XGBoost - RMSE: {rmse_xg}, MAE: {mae_xg}, R2: {r2_xg}')

results = pd.DataFrame({
    'True RUL': y_test,
    'Predicted RUL DT': dt_pred,
    'Predicted RUL RF': rf_pred,
    'Predicted RUL XG': xg_pred
})

results_sort = results.sort_values(by='True RUL').reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(results_sort.index, results_sort['True RUL'], label='True RUL', marker='o', linestyle='-', color='orange')
plt.plot(results_sort.index, results_sort['Predicted RUL DT'], label='Predicted RUL DT', marker='x', linestyle='--', color='black')
plt.plot(results_sort.index, results_sort['Predicted RUL RF'], label='Predicted RUL RF', marker='^', linestyle='--', color='green')
plt.plot(results_sort.index, results_sort['Predicted RUL XG'], label='Predicted RUL XG', marker='s', linestyle='--', color='blue')
plt.title('True vs Predicted RUL')
plt.xlabel('Engines (sorted by True RUL)')
plt.ylabel('Remaining Useful Life (RUL)')
plt.legend()
plt.grid()
plt.savefig('comb_rul_prediction_plot.png')
plt.show()

param_grid = {
    'max_depth': [3, 6, 10],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters found: ", grid_search.best_params_)

# Print the best score (lowest RMSE or highest R² depending on the scoring)
print("Best score (neg_mean_squared_error): ", grid_search.best_score_)