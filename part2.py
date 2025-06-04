import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_test, df_train, operational_condition_names, sensor_names
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error



lifetimes = df_train.groupby("engine")["cycle"].max().values
# print(f"lifetimes in train set: {lifetimes}")




engine_1_df = df_train[df_train['engine'] == 1]

# Drop index columns to get only sensor and operational data
sensor_data = engine_1_df.drop(columns=['engine', 'cycle'])
sensor_data.index = engine_1_df['cycle']
sensor_data.plot(subplots=True, figsize=(15, 25), layout=(6, 4), title='Sensor and operational data for Engine 1')
plt.tight_layout()
# plt.show()

# def plot_sensor_opacity(df, sensors, cycle_col='cycle', engine_col='engine', alpha=0.1):
#     """
#     Overlays each engine's sensor data as a transparent line for all sensors.
#     """
#     plt.figure(figsize=(18, 25))
    
#     for i, sensor in enumerate(sensors):
#         plt.subplot(6, 4, i + 1)
        
#         for engine_id in df[engine_col].unique():
#             eng_data = df[df[engine_col] == engine_id]
#             plt.plot(eng_data[cycle_col], eng_data[sensor], alpha=alpha)
        
#         plt.title(sensor)
#         plt.xlabel('Cycle')
#         plt.ylabel('Sensor reading')
#         plt.grid(True)
    
#     plt.tight_layout()
#     plt.suptitle("Opacity Curves for All Engines (each line = 1 engine)", fontsize=18, y=1.02)
#     plt.show()

sensor_cols = [col for col in df_train.columns if col not in ['engine', 'cycle']]  # drop non-sensor columns if needed
# plot_sensor_opacity(df_train, sensors=sensor_cols, alpha=0.1)

df_train = df_train.drop(columns=['altitude','TRA','mach_nr', 'T2','P2','P15','epr','farB','Nf_dmd','PCNfR_dmd'])  # drop non-sensor columns if needed
print(df_train.columns)

df_train['max_cycle'] = df_train.groupby('engine')['cycle'].transform('max')
df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
df_train['RUL'] = df_train['RUL'].clip(upper=125)
df_train.drop(columns='max_cycle', inplace=True)

X = df_train
Y = df_train['RUL']

feature_cols = [
    'T24', 'T30', 'T50',
    'P30',
    'Nf', 'Nc', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR',
    'htBleed',
    'W31', 'W32'
]

X_train = df_train[df_train['engine'][feature_cols]]
y_train = df_train[df_train['engine']['RUL']]

X_val = df_test[df_test['engine'][feature_cols]]
y_val = pd.read_csv('CMAPSSData/RUL_FD001.txt')

from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(10, 6))
plt.scatter(y_val, dt.predict(X_val), label='Decision Tree', alpha=0.5)
# plt.scatter(y_val, rf.predict(X_val), label='Random Forest', alpha=0.5)
# plt.scatter(y_val, xgb.predict(X_val), label='XGBoost', alpha=0.5)
plt.plot([0, 150], [0, 150], 'k--')
plt.xlabel("True RUL")
# plt.ylabel("Predicted RUL")
# plt.title("Predicted vs True RUL")
plt.legend()
plt.grid(True)
plt.show()
