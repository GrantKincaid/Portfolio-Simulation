import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from scipy.stats import spearmanr
from scipy.interpolate import interp1d

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
        ratio_data = self.calculate_OC_ratios(candle_data)
        buckets = self.bucket_data_quantile(ratio_data)
        transition_matrix = self.build_transition_matrix(buckets)
        return transition_matrix

    def interpolate_to_length(self, array, target_length):
        """
        Interpolates a 1D NumPy array to match the target length.
        """
        x_old = np.linspace(0, 1, len(array))  # Original indices normalized
        x_new = np.linspace(0, 1, target_length)  # New indices
        interpolator = interp1d(x_old, array, kind='linear', fill_value='extrapolate')
        return interpolator(x_new)

    def calculate_asset_correlation(self, bucketed_assets):
        """
        Computes the Spearman correlation matrix, ensuring sequences are interpolated to match the longest one.

        Args:
            bucketed_assets (list of np.ndarray): List of bucketed sequences for different assets.

        Returns:
            np.ndarray: Correlation matrix (num_assets x num_assets).
        """
        num_assets = len(bucketed_assets)
        
        # Find the longest sequence length
        max_length = max(len(arr) for arr in bucketed_assets)

        # Interpolate all sequences to the longest length
        aligned_assets = [self.interpolate_to_length(arr, max_length) for arr in bucketed_assets]

        # Compute correlation matrix
        correlation_matrix = np.zeros((num_assets, num_assets))

        for i in range(num_assets):
            for j in range(num_assets):
                if i != j:
                    correlation_matrix[i, j], _ = spearmanr(aligned_assets[i], aligned_assets[j])
                else:
                    correlation_matrix[i, j] = 1.0  # Self-correlation is always 1

        return correlation_matrix
    
    def generate_asset_correlation_matrix(self, li_data):

        li_buckets = []
        for row in li_data:
            ratios = self.calculate_OC_ratios(row)
            li_buckets.append(self.bucket_data_quantile(ratios))

        correlation_matrix = self.calculate_asset_correlation(li_buckets)
        return correlation_matrix

    @staticmethod
    def predict_next_state(current_state, transition_matrix):
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
    return df.values

def filter_data(array):
    if array.shape[0] > 1000:
        return True
    else:
        return False
    
if __name__ == "__main__":

    ## Load Data and Filter for issues
    app_dir = os.path.dirname(os.path.abspath(__file__))
    li_path_data = glob.glob(os.path.join(app_dir, "Data", "*.csv"))

    with mp.Pool() as pool:
        raw_results = pool.map(Load_data, li_path_data)

    filtered_results = []
    for result in raw_results:
        if filter_data(result):
            filtered_results.append(result)
        else:
            continue

    print(f"length resutls {len(filtered_results)}")

    ## Create Assets chain Transition matrixes
    tm = TransitionMatrixces(num_buckets=20)

    # Markov asset specific
    with mp.Pool() as pool:
        results = pool.map(tm.generate_transition_matrix, filtered_results)

    print(f"number transition matrixes {len(results)}")

    # Corralation total assets
    correlation_matrix = tm.generate_asset_correlation_matrix(filtered_results)
    tm.plot_transition_matrix_heatmap(correlation_matrix)
