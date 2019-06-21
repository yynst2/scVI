import os
import pandas as pd


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
