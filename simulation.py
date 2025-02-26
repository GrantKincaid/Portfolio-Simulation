import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
from numba import njit
from numba.typed import List
import itertools
import matplotlib.pyplot as plt

class TransitionMatrixces():
    def __init__(self, num_buckets):
        self.num_buckets = num_buckets

    def calculate_OC_ratios(self, data):
        """
        Creates the slope for each candle open to close
        Args:
            array (numpy.ndarray): array of candle open and close values [open, close]
        Return:
            array (numpy.ndarray): slopes of candles
        """
        eps = 1e-8  # Small value to replace zero
        safe_open_prices = np.where(data[:, 0] == 0, eps, data[:, 0])  # Replace zeros in open prices
        candle_slopes = (data[:, 1] / safe_open_prices) - 1
        return candle_slopes
    
    def bucket_data_quantile(self, array):
        """
        Buckets the data into a specified number of bins using quantiles.
        Args:
            array (numpy.ndarray): The input array of "candel ratio" values.
            num_buckets (int): The number of quantile-based buckets.
        Returns:
            numpy.ndarray: Array of bucketed values (discrete states).
        """
        quantiles = np.quantile(array, np.linspace(0, 1, self.num_buckets + 1))  # Compute bin edges
        bucketed_values = np.digitize(array, quantiles, right=True) - 1  # Assign each value to a quantile bucket
        return bucketed_values
    
    def average_slope_values_of_buckets(self, ratio_data, buckets):
        # Compute average slope for each bucket (state)
        slope_values = np.zeros(self.num_buckets)
        for state in range(self.num_buckets):
            # Get indices where the bucket matches the current state
            state_slopes = ratio_data[buckets == state]
            if state_slopes.size > 0:
                slope_values[state] = np.mean(state_slopes)
            else:
                slope_values[state] = np.nan  # or 0, depending on your needs

        return slope_values

    def build_transition_matrix(self, bucketed_array):
        """
        Constructs a transition matrix from the bucketed state sequence.
        
        Args:
            bucketed_array (numpy.ndarray): Sequence of bucketed values (discrete states).
            num_buckets (int): The number of unique states (buckets).
            
        Returns:
            numpy.ndarray: Transition probability matrix of shape (num_buckets, num_buckets).
        """
        # Initialize a transition matrix
        transition_counts = np.zeros((self.num_buckets, self.num_buckets))

        # Count transitions
        for i in range(len(bucketed_array) - 1):
            current_state = bucketed_array[i] - 1  # Adjust for zero-indexing
            next_state = bucketed_array[i + 1] - 1
            transition_counts[current_state, next_state] += 1

        # Normalize counts to get probabilities
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_counts, row_sums, where=row_sums > 0)  # Avoid division by zero

        return transition_matrix
    
    def generate_transition_matrix(self, candle_data):
        """
        All in one method to create transition matrix
        Args:
            array (numpy.ndarray): candle data in format [open, close]

        Returns:
            numpy.ndarray: Transition probability matrix of shape (num_buckets, num_buckets).
        """
        N = candle_data[0][-1][1]
        path = candle_data[1].split('\\')
        path = path[-1].split('.')
        symbol = path[0]

        ratio_data = self.calculate_OC_ratios(candle_data[0])
        buckets = self.bucket_data_quantile(ratio_data)
        transition_matrix = self.build_transition_matrix(buckets)
        average_slopes = self.average_slope_values_of_buckets(ratio_data, buckets)    

        return transition_matrix, N, symbol, average_slopes


    @staticmethod
    def deterministic_next_state(current_state, transition_matrix):
        """
        Predicts the next state based on the Markov transition matrix.
        
        Args:
            current_state (int): The current state (bucket index).
            transition_matrix (numpy.ndarray): The transition probability matrix.
            
        Returns:
            int: The predicted next state.
        """
        probabilities = transition_matrix[current_state - 1]  # Adjust for zero-indexing
        return np.argmax(probabilities) + 1  # Convert back to one-based index
    
    def probabalistic_next_state(self, current_state, transition_matrix):
        # Get the row corresponding to the current state (adjust for zero-indexing)
        probabilities = transition_matrix[current_state - 1].copy()

        # Remove any negative values (set them to zero)
        probabilities = np.clip(probabilities, 0, None)
        
        # Normalize probabilities so they sum to 1
        total = probabilities.sum()
        if total == 0:
            # If the row sums to zero (no transitions), assign a uniform distribution
            probabilities = np.full_like(probabilities, 1.0 / len(probabilities))
        else:
            probabilities /= total

        states = np.arange(1, self.num_buckets + 1)  # 1-indexed states
        next_state = np.random.choice(states, p=probabilities)
        return next_state

        
    def plot_value_distribution(self, array):
        """
        Groups similar values and plots their distribution as a bar chart.

        Args:
            array (numpy.ndarray): Input numerical data.

        Returns:
            None (Displays the histogram).
        """
        # Quantile-based binning (each bin has approximately the same number of points)
        bins = np.quantile(array, np.linspace(0, 1, self.num_buckets + 1))

        # Assign values to bins
        bucketed_values = np.digitize(array, bins, right=True)
        
        # Count occurrences per bucket
        unique_bins, counts = np.unique(bucketed_values, return_counts=True)
        
        # Plot bar chart
        plt.figure(figsize=(8, 5))
        plt.bar(unique_bins, counts, width=0.7, edgecolor="black")
        plt.xlabel("Bin Index")
        plt.ylabel("Frequency")
        plt.title(f"Value Distribution (quantile)")
        plt.xticks(unique_bins)  # Set x-ticks to bin indices
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

    @staticmethod
    def plot_transition_matrix_heatmap(transition_matrix):
        """
        Plots a heatmap of the Markov transition matrix.
        
        Args:
            transition_matrix (numpy.ndarray): The transition probability matrix.
            
        Returns:
            None (Displays the heatmap).
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=range(1, transition_matrix.shape[1] + 1),
                    yticklabels=range(1, transition_matrix.shape[0] + 1),
                    linewidths=0.5, linecolor='black', cbar=True)
        
        plt.xlabel("Next State")
        plt.ylabel("Current State")
        plt.title("Markov Transition Matrix Heatmap")
        plt.show()



def Load_data(path):
    df = pd.read_csv(path, usecols=['Open', 'Close'])
    return df.values, path

def filter_data(array):
    if array.shape[0] > 1000:
        return True
    else:
        return False
    

def simulation(iteration, transition_matrix):
    start_cash = 10_000_000
    simulation_steps = 1_000
    num_buckets=20
    tm = TransitionMatrixces(num_buckets)

    idx_symbol = {}

    i = 0
    for matrix, N, symbol, avg_slopes in transition_matrix:
        arr_values = np.zeros((simulation_steps), dtype=np.float16)

        last_state = None
        for j in range(simulation_steps):
            if not last_state:
                c_state = tm.probabalistic_next_state(np.random.randint(0, num_buckets), matrix)
                arr_values[j] = avg_slopes[c_state -1] + N
                last_state = c_state
            else:
                c_state = tm.probabalistic_next_state(last_state, matrix)
                arr_values[j] = avg_slopes[c_state -1] + arr_values[j-1]
                last_state = c_state

        varince = np.var(arr_values)
        if np.isnan(varince) or np.isinf(varince):
            continue

        print(varince)

        #plt.plot(arr_values)
        #plt.title(symbol)
        #plt.show()

        idx_symbol[i] = {
                'Matrix' : matrix,
                'prices' : arr_values,
                'symbol' : symbol,
                'avg_slopes' : avg_slopes,
                'N' : N
            }

        i += 1
        
    num_assets = len(idx_symbol)

    arr_temp = np.random.random((num_assets, 1))
    arr_ratios = arr_temp / arr_temp.sum(axis=0, keepdims=True)
        
    print(arr_ratios)
    print(np.sum(arr_ratios))
    
    i = 0
    for item in idx_symbol.items():
        
        
        
        i += 1

if __name__ == "__main__":

    ## Load Data and Filter for issues
    app_dir = os.path.dirname(os.path.abspath(__file__))
    li_path_data = glob.glob(os.path.join(app_dir, "Data", "*.csv"))

    with mp.Pool() as pool:
        raw_results = pool.map(Load_data, li_path_data)

    filtered_results = []
    for result, path in raw_results:
        if filter_data(result):
            filtered_results.append([result, path])
        else:
            continue

    print(f"length resutls {len(filtered_results)}")

    ## Create Assets chain Transition matrixes
    tm = TransitionMatrixces(num_buckets=20)

    # Markov asset specific
    with mp.Pool() as pool:
        matrix_results = pool.map(tm.generate_transition_matrix, filtered_results)

    print(f"number transition matrixes {len(matrix_results)}")

    #for matrix, N, symbol, slopes in matrix_results:
        #print(matrix)
        #print(N)
        #print(symbol)
        #print(slopes)

    # ----------- SIMULATION ---------------

    count = [i for i in range(10)]

    iterable_matrix_results = itertools.product(count, [matrix_results])

    #with mp.Pool(processes=1) as pool:
        #sim_results = pool.starmap(simulation, iterable_matrix_results)
    simulation(1, matrix_results)