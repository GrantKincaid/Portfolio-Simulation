import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import glob
import os
from numba import njit
from scipy.stats import norm
import matplotlib.colors as mcolors
import pandas as pd

def read_pickle(path):
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

def load_pickles(folder):
    files = glob.glob(os.path.join(folder, 'Simulations', '*.pkl'))
    with mp.Pool() as pool:
        results = pool.map(read_pickle, files)

    joined_object = []
    for result in results:
        for symbols, performance in result:
            data = [symbols, performance]
            joined_object.append(data)

    return joined_object

@njit
def pearson_correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    xm = x - x_mean
    ym = y - y_mean
    numerator = np.dot(xm, ym)
    denominator = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
    return numerator / denominator

def plot_simulations(data):
    i = 0
    for symbol, arr in data:
        # Graph every 100th simulation
        if not 0 == i % 100:
            i += 1
            continue
        plt.plot(arr)
        i += 1
    plt.show()

def sharp_ratio(data):
    # Initialize lists to store computed portfolio statistics
    annual_returns = []
    annual_vols = []
    sharpe_ratios = []
    symbols = []

    annualization_factor = np.sqrt(252)  # For volatility annualization
    risk_free_rate = 0.043 # Adjust if you have a risk-free rate

    # Loop through each portfolio
    for symbol, arr in data:
        # Store symbol for labeling later
        symbols.append(symbol)
        
        # Convert cumulative returns to daily relative returns:
        # daily_return = (current_day / previous_day) - 1
        daily_returns = np.diff(arr) / arr[:-1]
        daily_returns = daily_returns[1:-1]
        
        # Compute mean and standard deviation of daily returns
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)
        
        # Annualize the return and volatility
        ann_return = mean_daily * 252  # Rough approximation
        ann_vol = std_daily * annualization_factor
        
        # Compute Sharpe ratio (assume risk-free rate = 0 or adjust accordingly)
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan
        
        # Append the results
        annual_returns.append(ann_return)
        annual_vols.append(ann_vol)
        sharpe_ratios.append(sharpe)

    # Convert lists to numpy arrays for ease of manipulation
    annual_returns = np.array(annual_returns)
    annual_vols = np.array(annual_vols)
    sharpe_ratios = np.array(sharpe_ratios)

    # Find the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.nanargmax(sharpe_ratios)
    print(f'max sharp ratio {sharpe_ratios[max_sharpe_idx]}')

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.xlim(-10, 100)  # Limits annual volatility from 0 to 100
    plt.ylim(-75, 75)  # Limits annual return from 0 to 100
    sc = plt.scatter(annual_vols, annual_returns, c=sharpe_ratios, cmap='viridis', s=100, edgecolors='k')
    plt.xlabel('Annualized Volatility')
    plt.ylabel('Annualized Return')
    plt.title('Efficiency Frontier')
    plt.colorbar(sc, label='Sharpe Ratio')

    # Highlight the portfolio with the maximum Sharpe ratio
    plt.scatter(annual_vols[max_sharpe_idx], annual_returns[max_sharpe_idx],
                color='red', marker='*', s=300, label='Max Sharpe Ratio')
    plt.legend()

    plt.show()

def VaR_CVaR(data):
    # Assume 'portfolio_returns' is your list of numpy arrays (each simulation over time)
    # Extract the final return from each simulation
    final_returns = np.array([arr[-1] for symbol, arr in data])

    # Plot a histogram of the final returns
    plt.hist(final_returns, bins=30, density=True, alpha=0.6, color='g')

    # Fit a normal distribution to the data
    mu, sigma = norm.fit(final_returns)

    # Create a range of values for plotting the fitted distribution
    xmin, xmax = final_returns.min(), final_returns.max()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, sigma)
    plt.plot(x, p, 'r', linewidth=2)

    # Set the confidence level for VaR and CVaR calculations
    alpha = 0.05

    # Calculate VaR directly from the simulation data (5th percentile)
    VaR_percentile = np.percentile(final_returns, 100 * alpha)

    # Calculate CVaR (data-based): average of returns worse than VaR
    CVaR_percentile = final_returns[final_returns <= VaR_percentile].mean()

    # Alternatively, for the normal fit, calculate VaR and CVaR:
    z = norm.ppf(alpha)
    VaR_normal = mu + z * sigma
    # CVaR from a normal distribution can be computed as:
    CVaR_normal = mu - sigma * (norm.pdf(z) / alpha)

    print("VaR (5th percentile from data):", VaR_percentile)
    print("CVaR (from data):", CVaR_percentile)
    print("VaR (from normal fit):", VaR_normal)
    print("CVaR (from normal fit):", CVaR_normal)

    # Plot vertical lines for VaR and CVaR from the simulation data
    plt.axvline(VaR_percentile, color='b', linestyle='dashed', linewidth=2, 
                label=f'VaR (data): {VaR_percentile:.2%}')
    plt.axvline(CVaR_percentile, color='m', linestyle='dashdot', linewidth=2, 
                label=f'CVaR (data): {CVaR_percentile:.2%}')

    # Optionally, plot the normal fit VaR and CVaR as well
    plt.axvline(VaR_normal, color='k', linestyle='dotted', linewidth=2, 
                label=f'VaR (normal): {VaR_normal:.2%}')
    plt.axvline(CVaR_normal, color='orange', linestyle=':', linewidth=2, 
                label=f'CVaR (normal): {CVaR_normal:.2%}')

    # Enhance the plot with title, labels, and legend
    plt.title('Histogram of Final Portfolio Returns with VaR and CVaR')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

def cumalative_3D_distribution(data):
    # Assume 'portfolio_returns' is a list of numpy arrays 
    # where each array is a simulation over T time steps.
    # First, convert the list into a 2D numpy array with shape (n_simulations, T)
    sim_data = np.zeros((len(data), data[0][1].shape[0]-28), dtype=np.float32)
    i = 0
    for symbol, arr in data:
        sim_data[i] = arr[28:]
        i += 1

    # Transpose to get an array of shape (T, n_sim)
    # Each row corresponds to a time step and contains the portfolio values from all simulations.
    data = sim_data.T  # shape: (T, n_sim)
    T, n_sim = data.shape
    time = np.arange(T)

    min_return = data.min()
    max_return = data.max()
    x = np.linspace(min_return, max_return, 100)

    density_matrix = np.zeros((T, len(x)))
    epsilon = 1e-6  # small value to avoid zero sigma

    for t in range(T):
        mu_t, sigma_t = norm.fit(data[t])
        # Check for zero variance (as happens at t=0) and adjust if needed
        if sigma_t == 0:
            sigma_t = epsilon
        density_matrix[t, :] = norm.pdf(x, mu_t, sigma_t)

    X_mesh, Y_mesh = np.meshgrid(x, time)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    norm_power = mcolors.PowerNorm(gamma=0.25)
    surf = ax.plot_surface(X_mesh, Y_mesh, density_matrix, cmap='viridis', edgecolor='none', alpha=0.9, norm=norm_power)

    # Rotate the plot by setting the elevation and azimuth angles
    ax.view_init(elev=30, azim=90)

    ax.set_xlabel('Return')
    ax.set_ylabel('Time Step')
    ax.set_zlabel('Density')
    plt.title('3D Cumulative Distribution Over Time (Adjusted t=0)')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def portfolio_selection(data):
    risk_free_rate= 0.043  # risk-free rate
    annualization_factor = np.sqrt(252)  # For volatility annualization

    N = 5  # Number of top Sharpe ratios to select

    # Preallocate arrays for keeping results; indices remain constant.
    num_portfolios = len(data)
    sharpe_ratios = np.empty(num_portfolios)
    symbols = [None] * num_portfolios

    # Loop through each portfolio, computing daily returns and then the annualized Sharpe ratio.
    for i, (symbol, arr) in enumerate(data):
        symbols[i] = symbol
        # Compute daily returns from cumulative returns:
        # r_t = (arr[t] / arr[t-1]) - 1
        daily_returns = np.diff(arr) / arr[:-1]
        daily_returns = daily_returns[1:-1]
        
        mean_daily = np.mean(daily_returns)
        std_daily = np.std(daily_returns)

        ann_return = mean_daily * 252  # Rough approximation
        ann_vol = std_daily * annualization_factor
        
        # Compute Sharpe ratio, adjusting for the annualized risk-free rate.
        sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else np.nan
        sharpe_ratios[i] = sharpe

    # Use np.argsort to get the indices of the N best Sharpe ratios, preserving original indexing.
    best_indices = np.argsort(sharpe_ratios)[-N:][::-1]

    print("Indices of the best Sharpe ratios:", best_indices)
    print("Sharpe Ratios:", sharpe_ratios[best_indices])

    li_li_ratios = [symbols[i] for i in best_indices]
    flattened = [d for sublist in li_li_ratios for d in sublist]
    df = pd.DataFrame(flattened)
    # Group by 'symbol' and calculate the average of 'ratio'
    result_df = df.groupby('symbol', as_index=False)['ratio'].mean()
    result_df.to_csv('portfolio_ratios.csv')

if __name__ == '__main__':
    app_dir = os.path.dirname(os.path.abspath(__file__))

    print('loading data')
    # data = [dict[i]{symbol, ratio}, numpy_arr(performance)]
    data = load_pickles(app_dir)

    print('graphing simulation')
    plot_simulations(data)

    print('calculating sharp ratios')
    sharp_ratio(data)

    print('calculating VaR & CVaR')
    VaR_CVaR(data)

    print('building 3D distribution')
    cumalative_3D_distribution(data)

    print('find best portfolios')
    portfolio_selection(data)