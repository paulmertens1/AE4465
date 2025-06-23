import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_test, df_train, operational_condition_names, sensor_names
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# In the end this file only does the data preprocessing and visualization of how the sensors behave

lifetimes = df_train.groupby("engine")["cycle"].max().values
print(f"lifetimes in train set: {lifetimes}")

engine_1_df = df_train[df_train['engine'] == 1]

sensor_data = engine_1_df.drop(columns=['engine', 'cycle'])
sensor_data.index = engine_1_df['cycle']
sensor_data.plot(subplots=True, figsize=(15, 25), layout=(6, 4), title='Sensor and operational data for Engine 1')
plt.tight_layout()
plt.show()

def plot_sensor_opacity(df, sensors, cycle_col='cycle', engine_col='engine', alpha=0.1):

    plt.figure(figsize=(18, 25))
    
    for i, sensor in enumerate(sensors):
        plt.subplot(6, 4, i + 1)
        
        for engine_id in df[engine_col].unique():
            eng_data = df[df[engine_col] == engine_id]
            plt.plot(eng_data[cycle_col], eng_data[sensor], alpha=alpha)
        
        plt.title(sensor)
        plt.xlabel('Cycle')
        plt.ylabel('Sensor reading')
        plt.grid(True)
    
    plt.tight_layout()
    plt.suptitle("Opacity Curves for All Engines (each line = 1 engine)", fontsize=18, y=1.02)
    plt.show()

sensor_cols = [col for col in df_train.columns if col not in ['engine', 'cycle']] 
plot_sensor_opacity(df_train, sensors=sensor_cols, alpha=0.1)

df_train = df_train.drop(columns=['altitude','TRA','mach_nr', 'T2','P2','P15','epr','farB','Nf_dmd','PCNfR_dmd'])  
print(df_train.columns)

df_train['max_cycle'] = df_train.groupby('engine')['cycle'].transform('max')
df_train['RUL'] = df_train['max_cycle'] - df_train['cycle']
df_train['RUL'] = df_train['RUL'].clip(upper=125)
df_train.drop(columns='max_cycle', inplace=True)

# the featured columns used for training the model
feature_cols = [
    'T24', 'T30', 'T50',
    'P30',
    'Nf', 'Nc', 'Ps30', 'phi',
    'NRf', 'NRc', 'BPR',
    'htBleed',
    'W31', 'W32']