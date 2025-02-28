import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
import glob
import os

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


def plot_simulations(data):
    for symbol, arr in data:
        plt.plot(arr)
    plt.show()


if __name__ == '__main__':
    app_dir = os.path.dirname(os.path.abspath(__file__))

    print('loading data')
    # data = [dict[i]{symbol, ratio}, numpy_arr(performance)]
    data = load_pickles(app_dir)

    print('graphing simulation')
    plot_simulations(data)