import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_test, df_train, operational_condition_names, sensor_names

print(df_test)