import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_test, df_train, operational_condition_names, sensor_names



#######################

# Part 1: Preventive Maintenance

#######################



lifetimes = df_train.groupby("engine")["cycle"].max().values
plt.hist(lifetimes, bins=20, density=True, alpha=0.5, edgecolor="black", label="Lifetime data")

fitters = {
    "Weibull_2P": Fit_Weibull_2P(failures=lifetimes, show_probability_plot=False),
    "Exponential_1P": Fit_Exponential_1P(failures=lifetimes, show_probability_plot=False),
    "Lognormal_2P": Fit_Lognormal_2P(failures=lifetimes, show_probability_plot=False),
    "Normal_2P": Fit_Normal_2P(failures=lifetimes, show_probability_plot=False),
}
# Determine the best distribution based on AIC
best_fit_name = None
best_fit_aic = float("inf")
best_fit = None

for name, fitter in fitters.items():
    aic_value = getattr(fitter, "AICc", getattr(fitter, "AIC", np.inf))

    print(f"{name} AIC: {aic_value}")
    if aic_value < best_fit_aic:
        best_fit_aic = aic_value
        best_fit_name = name
        best_fit = fitter

print("\nBest distribution based on AIC:")
print(best_fit_name)
for attr in dir(best_fit):
    if not attr.startswith("_") and isinstance(getattr(best_fit, attr), (int, float)):
        print(f"  {attr} = {getattr(best_fit, attr)}")


x = np.linspace(min(lifetimes), max(lifetimes), 1000)

for name, fit in fitters.items():
    if name == "Weibull_2P":
        dist = Weibull_Distribution(alpha=fit.alpha, beta=fit.beta)
    elif name == "Lognormal_2P":
        dist = Lognormal_Distribution(mu=fit.mu, sigma=fit.sigma)
    elif name == "Exponential_1P":
        dist = Exponential_Distribution(Lambda=fit.Lambda)
    elif name == "Normal_2P":
        dist = Normal_Distribution(mu=fit.mu, sigma=fit.sigma)
    else:
        continue  # skip unknown distributions

    y = dist.PDF(x)
    plt.plot(x, y, label=f"{name}")

plt.title("Histogram with PDFs of Fitted Distributions")
plt.xlabel("Flight Cycles Until Failure")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


best_dist = Lognormal_Distribution(mu=best_fit.mu, sigma=best_fit.sigma)
pdf_vals = best_dist.PDF(x)
sf_vals = best_dist.SF(x)
hazard_vals = np.where(sf_vals > 0, pdf_vals / sf_vals, np.nan) 
plt.figure(figsize=(10, 6))
plt.plot(x, hazard_vals, label=f"{best_fit_name} Hazard")
plt.title(f"Hazard Function of {best_fit_name}")
plt.xlabel("Flight Cycles")
plt.ylabel("Hazard Rate h(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


Cp = 10000 # cost of preventive maintenance
Cf = 100000 # cost of failure
print(min(lifetimes))
t_range =  np.arange(min(lifetimes), max(lifetimes), 1)
S_t = best_dist.SF(t_range)
g_t = (Cp * S_t + Cf * (1 - S_t)) / t_range

min_index = np.argmin(g_t)
t_star = t_range[min_index]
g_star = g_t[min_index]
