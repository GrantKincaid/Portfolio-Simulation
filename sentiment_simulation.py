import pandas as pd
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
import itertools
from numba import njit

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
    if array.shape[0] > 252 * 20:
        return True
    else:
        return False
    

@njit
def probabilistic_next_state_numba(num_buckets, current_state, matrix, sentiment, sentiment_weight, sentiment_range):
    # Get the row for current_state (0-indexed)
    row = matrix[current_state]
    p = np.empty_like(row)
    for i in range(len(row)):
        # Use non-negative probability from the matrix
        original = row[i] if row[i] > 0.0 else 0.0
        # Compute a Gaussian bias centered at the current sentiment value.
        # Buckets close to 'sentiment' get a boost.
        diff = i - sentiment
        bias = np.exp(- (diff * diff) / (2.0 * sentiment_range * sentiment_range))
        p[i] = original + sentiment_weight * bias

    # Sum the combined weights
    total = 0.0
    for i in range(len(p)):
        total += p[i]
    # If no valid transitions, choose uniformly at random
    if total == 0.0:
        return np.random.randint(0, num_buckets)
    
    # Normalize and compute CDF
    cdf = np.empty_like(p)
    s = 0.0
    for i in range(len(p)):
        s += p[i] / total
        cdf[i] = s
    # Draw a random number and select next state
    r = np.random.rand()
    for i in range(len(cdf)):
        if r < cdf[i]:
            return i
    return num_buckets - 1  # fallback


@njit
def simulate_asset(simulation_steps, num_buckets, matrix, N, avg_slopes, sentiment_array, sentiment_weight, sentiment_range):
    arr_values = np.empty(simulation_steps, dtype=np.float32)
    # Initialize first step separately using the sentiment from step 0.
    init_state = np.random.randint(0, num_buckets)
    c_state = probabilistic_next_state_numba(num_buckets, init_state, matrix, sentiment_array[0], sentiment_weight, sentiment_range)
    arr_values[0] = avg_slopes[c_state] + N
    last_state = c_state

    # For subsequent steps, use the corresponding sentiment for each step.
    for j in range(1, simulation_steps):
        c_state = probabilistic_next_state_numba(num_buckets, last_state, matrix, sentiment_array[j], sentiment_weight, sentiment_range)
        arr_values[j] = avg_slopes[c_state] + arr_values[j - 1]
        last_state = c_state
    return arr_values

@njit
def simulate_states_only(simulation_steps, num_buckets, matrix, N, avg_slopes, sentiment_array, sentiment_weight, sentiment_range):
    arr_values = np.empty(simulation_steps, dtype=np.float32)
    # Initialize first step separately using the sentiment from step 0.
    init_state = np.random.randint(0, num_buckets)
    c_state = probabilistic_next_state_numba(num_buckets, init_state, matrix, sentiment_array[0], sentiment_weight, sentiment_range)
    arr_values[0] = avg_slopes[c_state] + N
    last_state = c_state

    # For subsequent steps, use the corresponding sentiment for each step.
    for j in range(1, simulation_steps):
        c_state = probabilistic_next_state_numba(num_buckets, last_state, matrix, sentiment_array[j], sentiment_weight, sentiment_range)
        arr_values[j] = avg_slopes[c_state] + arr_values[j - 1]
        last_state = c_state
    return arr_values

def random_sentiment_basic(num_buckets, steps):
    sentiment = np.zeros((steps), dtype=np.int8)
    last_sentiment =  round(num_buckets / 2, 0)
    quarter_buckets = round(num_buckets / 4, 0)
    for i in range(steps):
        last_sentiment = np.random.randint(min(0, last_sentiment-quarter_buckets), max(num_buckets-1, last_sentiment+quarter_buckets))
        sentiment[i] = last_sentiment
    return sentiment


def random_sentiment(num_buckets, steps):
    sentiment = np.empty(steps, dtype=np.int8)
    last_sentiment = num_buckets // 2  # Start in the middle of the bucket range.
    
    for i in range(steps):
        # Calculate a step standard deviation that is larger when the sentiment is low.
        # When last_sentiment is low, (num_buckets - last_sentiment) is high, leading to larger steps.
        step_std = 1.0 + (num_buckets - last_sentiment) / num_buckets * 1.5  # Adjust multiplier as needed

        extream_std = 20.0
        if last_sentiment == num_buckets: # Extream State Means Extream Volatility
            step_std = extream_std
        
        # A small positive drift encourages moves toward higher bucket values.
        drift = 0.0
        
        # Sample a step from a normal distribution and round to an integer.
        change = int(round(np.random.normal(loc=drift, scale=step_std)))
        
        # Update sentiment and ensure it stays within the valid range.
        new_sentiment = last_sentiment + change
        new_sentiment = max(0, min(new_sentiment, num_buckets - 1))
        
        sentiment[i] = new_sentiment
        last_sentiment = new_sentiment
        
    return sentiment


def simulation(iteration, transition_matrix, matrix_spx):
    simulation_steps = 365 * 5  # Five years
    num_buckets = 20

    results = []  # Will hold simulation outputs for each asset

    # Create a sentiment trend for the simulation steps.
    sentiment = random_sentiment(num_buckets, simulation_steps)
    # Parameters to control the sentiment bias:
    sentiment_weight = 2.5    # Increase to strengthen sentiment influence.
    sentiment_range  = 20.0    # Sentiment boost will affect buckets within ~2 steps.

    # Simulate each asset, now using the sentiment trend.
    for matrix, N, symbol, avg_slopes in transition_matrix:
        arr_values = simulate_asset(simulation_steps, num_buckets, matrix, N, avg_slopes,
                                    sentiment, sentiment_weight, sentiment_range)
        variance = np.var(arr_values)
        # Skip assets with invalid variance
        if np.isnan(variance) or np.isinf(variance):
            continue
        arr_values = arr_values - arr_values[0]
        results.append({
            'matrix': matrix,
            'prices': arr_values,
            'symbol': symbol,
            'avg_slopes': avg_slopes,
            'N': N
        })

        #plt.plot(arr_values)
        #plt.show()

    num_assets = len(results)
    if num_assets == 0:
        return results, None

    # Generate asset ratios and normalize them (vectorized)
    arr_ratios = np.random.random(num_assets)
    arr_ratios /= arr_ratios.sum()

    # Compute cumulative performance in a vectorized way
    cumulative_performance = np.zeros(simulation_steps, dtype=np.float32)
    for i in range(num_assets):
        results[i]['ratio'] = arr_ratios[i]
        # Zero-base the performance accumulation
        cumulative_performance += results[i]['prices'] * arr_ratios[i]
    
    return results, cumulative_performance

    
def analysis():

    # Sharp ratio
        # max
        # efficency frontier graph

    # Var
    # Cvar

    # 3D Distribution by month

    pass


if __name__ == "__main__":

    # ----------- Data Prep --------------- #

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

    # index generation
    path_spx = os.path.join(app_dir, 'spx.csv')
    df_spx = pd.read_csv(path_spx, usecols=['Open', 'Close'])
    matrix_spx = tm.generate_transition_matrix([df_spx.values, path_spx])

    # ----------- SIMULATION --------------- #

    count = [i for i in range(10_000)]
    #clip data for testing
    #matrix_results = matrix_results[0:50]

    iterable_matrix_results = itertools.product(count, [matrix_results], [matrix_spx])

    with mp.Pool() as pool:
        sim_results = pool.starmap(simulation, iterable_matrix_results)

    for idx_symbol, performance in sim_results:
        plt.plot(performance)

    plt.show()