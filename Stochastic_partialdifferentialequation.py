import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from main import task_1
import os



#OPEN ended Analysis Using Stochastic Partial Differnetial Equations to fit price data, predict the future distribution

ercot_df=task_1("ERCOT")

settlement_df=ercot_df[ercot_df["SettlementPoint"]=="LZ_WEST"]
settlement_df = settlement_df[settlement_df["Date"] < pd.to_datetime("2017-04-01")]
#I only consider the first couple of months up to 2017-04 because beyond that time are jumps that need a more advanced model.

# -----------------------------------------------------------
# 1) Fit OU process to price series (It assumes normality but this is just an exercise, there are more advanced models)
# -----------------------------------------------------------
def fit_ou_mle(df, price_col="Price", time_col="Date"):
    """
    MLE for OU process dP = kappa*(theta-P)dt + sigma dW.
    Handles irregular dt using exact transition density.
    """
    x = df[[time_col, price_col]].dropna().copy()
    x[time_col] = pd.to_datetime(x[time_col])
    x = x.sort_values(time_col)

    P = x[price_col].to_numpy()
    t = x[time_col].astype("int64").to_numpy() / 1e9  # seconds
    dt = np.diff(t) / 3600.0                          # hours
    Pt, Ptm1 = P[1:], P[:-1]

    def nll(params):
        kappa, theta, sigma = params
        if kappa <= 1e-9 or sigma <= 1e-12:
            return np.inf
        e = np.exp(-kappa * dt)
        mean = theta + (Ptm1 - theta) * e
        var = (sigma**2) * (1 - e**2) / (2 * kappa)
        if np.any(var <= 0):
            return np.inf
        resid = Pt - mean
        return 0.5 * np.sum(np.log(2*np.pi*var) + resid**2 / var)

    # Initial guesses
    dt_med = np.median(dt) if len(dt) else 1.0
    b_init = 0.99
    kappa0 = -np.log(max(b_init, 1e-4)) / max(dt_med, 1e-6)
    theta0 = np.mean(P)
    sigma0 = np.std(np.diff(P)) / np.sqrt(max(dt_med, 1e-6))

    res = minimize(
        nll, 
        x0=np.array([kappa0, theta0, sigma0]),
        bounds=[(1e-6, None), (None, None), (1e-9, None)],
        method="L-BFGS-B"
    )
    kappa, theta, sigma = res.x
    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "dt_median_hours": float(dt_med),
        "ll": -float(res.fun)
    }

# -----------------------------------------------------------
# 2) Simulate OU from start + extrapolate into future (Monte carlo simulation)
# -----------------------------------------------------------
def ou_simulate_full_and_future(df, params, until, dt_hours=None, random_state=None):
    """
    Simulate OU process from the very first observation through the end of history
    and then extrapolate to 'until'.
    """
    x = df[['Date', 'Price']].dropna().copy()
    x['Date'] = pd.to_datetime(x['Date'])
    x = x.sort_values('Date')

    kappa, theta, sigma = params['kappa'], params['theta'], params['sigma']
    dt = params.get('dt_median_hours', 1.0) if dt_hours is None else float(dt_hours)

    until = pd.to_datetime(until)
    total_hours = (until - x['Date'].iloc[0]).total_seconds() / 3600.0
    n_steps = int(np.ceil(total_hours / dt))

    e = np.exp(-kappa * dt)
    q = (sigma**2) * (1 - e**2) / (2 * kappa)

    rng = np.random.default_rng(random_state)
    sim_prices = np.empty(n_steps + 1)
    sim_prices[0] = x['Price'].iloc[0]
    for t in range(1, n_steps + 1):
        mean = theta + (sim_prices[t-1] - theta) * e
        sim_prices[t] = mean + np.sqrt(q) * rng.standard_normal()

    all_dates = pd.date_range(
        start=x['Date'].iloc[0], periods=n_steps + 1, freq=pd.to_timedelta(dt, unit="h")
    )
    
    
    
    
    
    return all_dates, sim_prices

# -----------------------------------------------------------
# 3) Fit, simulate, and plot
# -----------------------------------------------------------
def ou_terminal_distribution_kde(df, params, horizon_days=90, dt_hours=None,
                                  n_paths=10000, random_state=42, bandwidth=None):
    """
    Simulate many OU paths from the last observed price out to 'horizon_days'
    and plot histogram + kernel density estimate. 
    """
    x = df[['Date','Price']].dropna().copy().sort_values('Date')
    P0 = float(x['Price'].iloc[-1])

    # time step
    dt = params.get('dt_median_hours', 1.0) if dt_hours is None else float(dt_hours)
    steps = int(np.ceil(horizon_days*24.0 / dt))

    # OU parameters
    kappa, theta, sigma = params['kappa'], params['theta'], params['sigma']
    e = np.exp(-kappa * dt)
    q = (sigma**2) * (1 - e**2) / (2 * kappa)

    # simulate
    rng = np.random.default_rng(random_state)
    P = np.empty((steps + 1, n_paths))
    P[0, :] = P0
    for t in range(1, steps + 1):
        mean = theta + (P[t-1, :] - theta) * e
        shocks = rng.standard_normal(n_paths) * np.sqrt(q)
        P[t, :] = mean + shocks

    terminal_prices = P[-1, :]

    # KDE fit
    kde = gaussian_kde(terminal_prices, bw_method=bandwidth)

    # grid for KDE curve
    price_grid = np.linspace(terminal_prices.min(), terminal_prices.max(), 500)
    kde_vals = kde(price_grid)

    # plot histogram + KDE
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, "density_OU_LZ_West_2017-06-30.png")
    
    plt.figure(figsize=(10, 5))
    plt.hist(terminal_prices, bins=60, alpha=0.5, density=True, label="Histogram")
    plt.plot(price_grid, kde_vals, color="red", linewidth=2, label="KDE")
    plt.title("OU Theoretical distribution of the Price for the LZ-West on 2017-06-30")
    plt.xlabel("Price")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save before showing
    plt.savefig(save_path, dpi=300)
    plt.show()
    print(f"saved in {save_path}")
    return terminal_prices, (price_grid, kde_vals)


#################################################################Program Starts here

#1: Fit OU to price data
params = fit_ou_mle(settlement_df, price_col="Price", time_col="Date")
print("Fitted parameters:", params)

#2: Simulate from first date through +90 days after last observed date
future_end = settlement_df["Date"].iloc[-1] + pd.Timedelta(days=90)
all_dates, sim_prices = ou_simulate_full_and_future(
    settlement_df, params, until=future_end, random_state=42
)

current_dir = os.getcwd()
save_path = os.path.join(current_dir, "OU_Simulation_From_Start.png")

plt.figure(figsize=(12, 5))
plt.plot(settlement_df["Date"], settlement_df["Price"], label="Observed", linewidth=1.5)
plt.plot(all_dates, sim_prices, label="Simulated (OU)", alpha=0.9, color="orange")
plt.axvline(settlement_df["Date"].iloc[-1], color="red", linestyle="--", alpha=0.6, label="Extrapolation Start")
plt.title("OU Simulation from Start + Future Extrapolation")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
plt.savefig(save_path, dpi=300)
plt.show()

print(f"Saved in {save_path}")
#3: Forecast the return distribution 90 days ahead using parameters calibrated from #1

# Simulate & plot with KDE
terminal_prices, kde_data = ou_terminal_distribution_kde(
    settlement_df, params, horizon_days=90, n_paths=20000, random_state=123

)
