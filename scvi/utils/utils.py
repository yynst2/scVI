import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_dir_if_necessary(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class IterativeDict:
    """
    Structure useful to save metrics for different models over different trainings

    Saved in a nested dictionnary
    Structure:
    model_name ==> metric_name ==> table [n_trainings, ...]
    """
    def __init__(self, model_names):
        self.values = {key: {} for key in model_names}

    def set_values(self, model_name, metric_name, values):
        if metric_name not in self.values[model_name]:
            self.values[model_name][metric_name] = [values]
        else:
            self.values[model_name][metric_name].append(values)

    def to_df(self):
        return pd.DataFrame(self.values)


def plot_traj(history, x=None, **plot_params):
    """
    :param history: (n_sim, n_x_values) array
    :param x: associated x values used for plotting
    :param plot_params: Plot parameters fed to plt.plot
    :return:
    """
    plot_params = {} if plot_params is None else plot_params
    history_np = np.array(history)
    theta_mean = np.mean(history_np, axis=0)
    theta_std = np.std(history_np, axis=0)
    n_iter = len(theta_mean)

    x = np.arange(n_iter) if x is None else x
    plt.plot(x, theta_mean, **plot_params)

    plt.fill_between(x=x, y1=theta_mean - theta_std, y2=theta_mean + theta_std, alpha=0.25)
