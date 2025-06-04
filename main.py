import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_test, df_train, operational_condition_names, sensor_names



################################

# Part 1: Preventive Maintenance

################################

#Extract lifetimes and get fitters

lifetimes = df_train.groupby("engine")["cycle"].max().values

fitters = {
    "Weibull_2P": Fit_Weibull_2P(failures=lifetimes, show_probability_plot=False),
    "Exponential_1P": Fit_Exponential_1P(failures=lifetimes, show_probability_plot=False),
    "Lognormal_2P": Fit_Lognormal_2P(failures=lifetimes, show_probability_plot=False),
    "Normal_2P": Fit_Normal_2P(failures=lifetimes, show_probability_plot=False),
}

# Plotting the fitted distributions
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
        continue  

    y = dist.PDF(x)
    plt.plot(x, y, label=f"{name}")

plt.hist(lifetimes, bins=20, density=True, alpha=0.5, edgecolor="black", label="Lifetime data")
plt.title("Histogram with PDFs of Fitted Distributions")
plt.xlabel("Flight Cycles Until Failure")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("pdf_fit_comparison.png")
plt.close()


# Determine the best distribution
# check both AIC and BIC values
best_fit_name = None
best_fit_aic = float("inf")
best_fit = None

print("\n Comparison of AIC values")
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


print("\n Comparison based on BIC values:")
best_fit_bic_name = None
best_fit_bic_value = float("inf")
for name, fitter in fitters.items():
    bic_value = getattr(fitter, "BIC", np.inf)
    print(f"{name} BIC: {bic_value}")
    if bic_value < best_fit_bic_value:
        best_fit_bic_value = bic_value
        best_fit_bic_name = name

print(f"\nBest distribution based on BIC: {best_fit_bic_name}")

# Hazard Function Calculation
if best_fit_name == "Weibull_2P":
    best_dist = Weibull_Distribution(alpha=best_fit.alpha, beta=best_fit.beta)
elif best_fit_name == "Exponential_1P":
    best_dist = Exponential_Distribution(Lambda=best_fit.Lambda)
elif best_fit_name == "Lognormal_2P":
    best_dist = Lognormal_Distribution(mu=best_fit.mu, sigma=best_fit.sigma)
elif best_fit_name == "Normal_2P":
    best_dist = Normal_Distribution(mu=best_fit.mu, sigma=best_fit.sigma)


pdf_vals = best_dist.PDF(x)
sf_vals = best_dist.SF(x)
hazard_vals =  pdf_vals / sf_vals

plt.figure(figsize=(10, 6))
plt.plot(x, hazard_vals, label=f"{best_fit_name} Hazard")
plt.title(f"Hazard Function of {best_fit_name}")
plt.xlabel("Flight Cycles")
plt.ylabel("Hazard Rate h(t)")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.show()
plt.savefig("hazard_function.png")
plt.close()


# Cost Analysis for Preventive Maintenance
Cp = 10000 # cost prevent
Cf = 100000 # cost fail
t_range =  np.arange(100, max(lifetimes), 1)
S_t = best_dist.SF(t_range)
print(f"Survival function S(t) at t = {t_range[23]}: {S_t[23]}")
g_t = (Cp * S_t + Cf * (1 - S_t)) / t_range

min_index = np.argmin(g_t)
t_star = t_range[min_index]
g_star = g_t[min_index]
print(f"Optimal replacement time t* = {t_star} with minimum cost g(t*) = {g_star}")

plt.figure(figsize=(10, 6))
plt.plot(t_range, g_t, label='g(t): Avg. cost per cycle')
plt.axvline(t_star, color='r', linestyle='--', label=f'Optimal t* = {t_star}')
plt.xlabel('Replacement time t (flight cycles)')
plt.ylabel('g(t): Expected cost per cycle')
plt.title('Optimal Preventive Replacement Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("g_of_t_cost_curve.png")
plt.close()



# Test the optimal replacement time on test data

test_lifetimes = df_test.groupby("engine")["cycle"].max().values


Cp = 10000
Cf = 100000

costs = []

for T_i in test_lifetimes:
    if T_i >= t_star:
        # Preventive replacement at t*, engine survived
        cost = Cp
    else:
        # Engine failed before t*, pay failure cost
        cost = Cf
    costs.append(cost)

average_cost = np.mean(costs)
print(f"Average cost per engine in test set (using t* = {t_star}): {average_cost:.2f}")
cost_per_cycle = average_cost / t_star
print(f"Average cost per cycle: {cost_per_cycle:.2f}")


plt.hist(test_lifetimes, bins=20, alpha=0.6, label="Test lifetimes")
plt.axvline(t_star, color='red', linestyle='--', label=f"t* = {t_star}")
plt.title("Test Set Engine Lifetimes with Preventive Replacement Cutoff")
plt.xlabel("Cycles")
plt.ylabel("Number of Engines")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("test_lifetimes_with_t_star.png")
plt.close()

#print(f"lifetimes in train set: {lifetimes}")
#print(f"lifetimes in test set: {test_lifetimes}")
