import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
import glob
import os
from numba import njit

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
    risk_free_rate_daily = np.sqrt(0.043) / annualization_factor  # Adjust if you have a risk-free rate

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
        sharpe = (ann_return - risk_free_rate_daily * 252) / ann_vol if ann_vol != 0 else np.nan
        
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

def return_distribution(data):
    pass

def CVar(data):
    pass

def Var(data):
    pass

def cumalative_3D_distribution(data):
    pass

if __name__ == '__main__':
    app_dir = os.path.dirname(os.path.abspath(__file__))

    print('loading data')
    # data = [dict[i]{symbol, ratio}, numpy_arr(performance)]
    data = load_pickles(app_dir)

    print('graphing simulation')
    plot_simulations(data)

    print('calculating sharp ratios')
    sharp_ratio(data)
