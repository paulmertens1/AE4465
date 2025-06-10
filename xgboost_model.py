import xgboost as xgb
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from import_data import df_train, df_test
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

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

X_train = df_train[feature_cols]
y_train = df_train['RUL']

df_test_last = df_test.groupby('engine').tail(1).copy()
true_rul = pd.read_csv(
    'CMAPSSData/RUL_FD001.txt',
    header=None
)

df_test_last['RUL'] = true_rul.values.flatten()

X_test = df_test_last[feature_cols]
y_test = df_test_last['RUL']

xg = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=6)
xg_fit = xg.fit(X_train, y_train)

y_pred = xg_fit.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')


results = pd.DataFrame({
    'True RUL': y_test,
    'Predicted RUL': y_pred
})

results_sort = results.sort_values(by='True RUL').reset_index(drop=True)


plt.figure(figsize=(10, 6))
plt.plot(results_sort.index, results_sort['True RUL'], label='True RUL', marker='o', linestyle='-', color='green')
plt.plot(results_sort.index, results_sort['Predicted RUL'], label='Predicted RUL', marker='x', linestyle='--', color='purple')
plt.title('True vs Predicted RUL xg Boost' )
plt.xlabel('Engines (sorted by True RUL)')
plt.ylabel('Remaining Useful Life (RUL)')
plt.legend()
plt.grid()
plt.savefig('xg_rul_prediction_plot.png')
plt.show()


z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([0, max(y_test.max(), y_pred.max())], [0, max(y_test.max(), y_pred.max())], 'r--')
plt.plot(y_test, p(y_test), 'b-', label='Fit Line')
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("XG Boost: Predicted vs True RUL")
plt.grid(True)
plt.tight_layout()
plt.savefig('xg_predicted_vs_true_rul.png')
plt.show()