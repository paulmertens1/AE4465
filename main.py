import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.Fitters import Fit_Weibull_2P, Fit_Exponential_1P, Fit_Lognormal_2P, Fit_Normal_2P
from reliability.Distributions import Weibull_Distribution, Exponential_Distribution, Lognormal_Distribution, Normal_Distribution
from import_data import df_train


#lifetimes uit de training data halen
# distributions fitten op de lifetimes

lifetimes = df_train.groupby("engine")["cycle"].max().values

fitters = {
    "Weibull_2P": Fit_Weibull_2P(failures=lifetimes, show_probability_plot=False),
    "Exponential_1P": Fit_Exponential_1P(failures=lifetimes, show_probability_plot=False),
    "Lognormal_2P": Fit_Lognormal_2P(failures=lifetimes, show_probability_plot=False),
    "Normal_2P": Fit_Normal_2P(failures=lifetimes, show_probability_plot=False),
}

# histogram plotten met de fitted distributions
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


# AIC en BIC vergelijken 
# dit was eigenlijk al gedaan in de fitters, daar hebben we ook de log-likelidhood en AD tests
#maarja nu krijgen we ook de bic en aic allemaal in een keer
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

print(f"\nBest distribution based on AIC:{ best_fit_name}")

print("\n Comparison of BIC values:")
best_fit_bic_name = None
best_fit_bic_value = float("inf")
for name, fitter in fitters.items():
    bic_value = getattr(fitter, "BIC", np.inf)
    print(f"{name} BIC: {bic_value}")
    if bic_value < best_fit_bic_value:
        best_fit_bic_value = bic_value
        best_fit_bic_name = name

print(f"\nBest distribution based on BIC: {best_fit_bic_name}")

# Hazard Function berekenen voor de beste fit
if best_fit_name == "Weibull_2P":
    best_dist = Weibull_Distribution(alpha=best_fit.alpha, beta=best_fit.beta)
elif best_fit_name == "Exponential_1P":
    best_dist = Exponential_Distribution(Lambda=best_fit.Lambda)
elif best_fit_name == "Lognormal_2P":
    best_dist = Lognormal_Distribution(mu=best_fit.mu, sigma=best_fit.sigma)
elif best_fit_name == "Normal_2P":
    best_dist = Normal_Distribution(mu=best_fit.mu, sigma=best_fit.sigma)

# hazard functie iets eerder berekenen dan bij de histogram
x = np.linspace(100, max(lifetimes), 1000)
pdf_vals = best_dist.PDF(x)
sf_vals = best_dist.SF(x) # SF = 1-F(x)
hazard_vals =  pdf_vals / sf_vals
# en plotten hazard functie
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


# Average Cost per Cycle berekenen
Cp = 10000 # cost prevent
Cf = 100000 # cost fail
t_range =  np.arange(100, max(lifetimes), 1)
S_t = best_dist.SF(t_range)
g_t_list = []

#gt berekenen voor elke t
for i, t in enumerate(t_range):
    u_vals = np.linspace(0, t, 500)
    s_vals = best_dist.SF(u_vals, show_plot=False)
    e_time = np.trapezoid(s_vals, u_vals)

    SF_t = S_t[i]
    g = (Cp * SF_t + Cf * (1 - SF_t)) / e_time
    g_t_list.append(g)

# minimum g(t) vinden
min_index = np.argmin(g_t_list)
t_star = t_range[min_index]
g_star = g_t_list[min_index]
print(f"Optimal replacement time t*= {t_star} with minimum cost g(t*) = {g_star}")

#plotten van g(t) en t*
plt.figure(figsize=(10, 6))
plt.plot(t_range, g_t_list, label='g(t): Avg. cost per cycle')
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


print("Nog wat dubbel checks van dingen")
print(np.mean(lifetimes))
print(np.exp(best_fit.mu)) #mean van distribution in de buurt van lifetimes mean
print(f"cost average between 0 to t* ={g_star*t_star}")
print(S_t[t_star-100])
print(S_t[t_star-100]*10000 + (1 - S_t[t_star-100]) * 100000)  
# kosten average lijken te kloppen met de g(t) kosten