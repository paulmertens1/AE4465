
import pandas as pd
import numpy as np
from import_data import df_train, df_test
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Compute RUL for training data
# -------------------------------

df_train['max_cycle'] = df_train.groupby('engine')['cycle'].transform('max')
df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
df_train['RUL'] = df_train['RUL'].clip(upper=125)
df_train.drop(columns='max_cycle', inplace=True)

# -------------------------------
# Step 2: Feature selection
# -------------------------------

feature_cols = [
    'T24', 'T30', 'T50',
    'P30',
    'Nf', 'Nc', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR',
    'htBleed',
    'W31', 'W32'
]


X_train = df_train[feature_cols]
y_train = df_train['RUL']

# -------------------------------
# Step 3: Prepare test data
# -------------------------------
# Predict RUL only for the last cycle of each engine in the test set
df_test_last = df_test.groupby('engine').tail(1).copy()

# You will need to load true RULs from a separate file like RUL_FD001.txt
# Example (adjust the path as needed):
true_rul = pd.read_csv(
    'CMAPSSData/RUL_FD001.txt',
    header=None
)



df_test_last['RUL'] = true_rul.values.flatten()

X_test = df_test_last[feature_cols]
y_test = df_test_last['RUL']

# -------------------------------
# Step 4: Train single decision tree
# -------------------------------
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

# -------------------------------
# Step 5: Evaluate model
# -------------------------------
y_pred = dt.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

results = pd.DataFrame({
    'True RUL': y_test,
    'Predicted RUL': y_pred
})

results_sort = results.sort_values(by='True RUL').reset_index(drop=True)


plt.figure(figsize=(10, 6))
plt.plot(results_sort.index, results_sort['True RUL'], label='True RUL', marker='o', linestyle='-', color='red')
plt.plot(results_sort.index, results_sort['Predicted RUL'], label='Predicted RUL', marker='x', linestyle='--', color='black')
plt.title('True vs Predicted RUL single Decision Tree')
plt.xlabel('Engines (sorted by True RUL)')
plt.ylabel('Remaining Useful Life (RUL)')
plt.legend()
plt.grid()
plt.savefig('dt_rul_prediction_plot.png')
plt.show()




print(f"Decision Tree Regressor Results:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")

# -------------------------------
# Step 6: Visualize predictions
# -------------------------------

z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([0, max(y_test.max(), y_pred.max())], [0, max(y_test.max(), y_pred.max())], 'r--')
plt.plot(y_test, p(y_test), 'b-', label='Fit Line')
plt.xlabel("True RUL")
plt.ylabel("Predicted RUL")
plt.title("Decision Tree: Predicted vs True RUL")
plt.savefig('dt_predicted_vs_true_rul.png')
plt.grid(True)
plt.tight_layout()
plt.show()
