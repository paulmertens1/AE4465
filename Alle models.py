# Gemaakt door Paul Mertens (5261406) en Mats Thijssen (4954114)

import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, train_test_split
from import_data import df_train, df_test
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

####################
# Data voorbereiding
####################

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


###############
# Model tunning
###############

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
tuning = False 
if tuning == True:
# Decision Tree
    dt_params = {'max_depth': [3, 5, 10], 'min_samples_leaf': [1, 5, 10], 'max_leaf_nodes': [10, 20]}
    dt_grid = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=3, scoring='neg_mean_squared_error')
    dt_grid.fit(X_train, y_train)
    best_dt = dt_grid.best_estimator_

# Random Forest
    rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_leaf': [1, 5, 10]}
    rf_grid = GridSearchCV(RandomForestRegressor(), rf_params, cv=3, scoring='neg_mean_squared_error')
    rf_grid.fit(X_train, y_train)
    best_rf = rf_grid.best_estimator_

# XGBoost
    xgb_params = {'n_estimators': [50, 100], 'max_depth': [3, 6, 10], 'learning_rate': [0.01, 0.1, 0.2]}
    xgb_grid = GridSearchCV(XGBRegressor(), xgb_params, cv=3, scoring='neg_mean_squared_error')
    xgb_grid.fit(X_train, y_train)
    best_xgb = xgb_grid.best_estimator_

# Evaluation with validation data 
    tuned_params = ['max_depth', 'n_estimators', 'max_leaf_nodes', 'min_samples_leaf', 'learning_rate']
    #read the best hyperparameters from the grid search
    for model, name in zip([best_dt, best_rf, best_xgb], ['DT', 'RF', 'XG']):
        y_pred = model.predict(X_val)
        params = model.get_params()
        filtered_params = {k: v for k, v in params.items() if k in tuned_params}
        print(f"{name} - Parameters: {filtered_params}")
        print(f"{name} - RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.2f}, "
            f"MAE: {mean_absolute_error(y_val, y_pred):.2f}, "
            f"R²: {r2_score(y_val, y_pred):.2f}")

#####################
# Running the models
#####################

true_rul = pd.read_csv(
    'CMAPSSData/RUL_FD001.txt',
    header=None
)
df_test_last['RUL'] = true_rul.values.flatten()

X_test = df_test_last[feature_cols]
y_test = df_test_last['RUL']

# Single Decision Tree Regressor, Random Forest Regressor, and XGBoost Regressor
# take the best hyperparameters from the tuning step
dt = DecisionTreeRegressor(max_depth=10, max_leaf_nodes=20, min_samples_leaf=1, random_state=1)
dt.fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_leaf=5, n_jobs = -1,random_state=1)
rf.fit(X_train, y_train)
xg = XGBRegressor(objective='reg:squarederror', n_estimators=50, learning_rate=0.1, max_depth=6,random_state=1)
xg_fit = xg.fit(X_train, y_train)

#add a bias if you want to ensure conservative predictions
bias = 0.0
dt_pred = dt.predict(X_test)+ bias
rf_pred = rf.predict(X_test) +bias 
xg_pred = xg_fit.predict(X_test) +bias

#####################
# Evaluatie modellen
#####################
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


####################
# Data visualization
####################

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


# Checking which sensor is the most important
models = {
    'Decision Tree': dt,
    'Random Forest': rf,
    'XGBoost': xg_fit
}
for name in models:
    importances = models[name].feature_importances_
    indices = np.argsort(importances)[::-1]
    print(f"\n{name} Feature Importances greater than 5%:")
    for i in indices:
        if importances[i] > 0.05:
            print(f"{feature_cols[i]}: {importances[i]:.4f}")
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_cols[i] for i in indices], rotation=45)
    plt.title(f'{name} Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig(f'{name}_feature_importances.png')
    plt.show()
