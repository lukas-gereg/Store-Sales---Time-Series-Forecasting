import re
import os
import sys

import pandas as pd
import torch
import itertools
import numpy as np

from collections import Counter

from utils.evaluation import Evaluation


class HiddenPrints:
    def __init__(self, debug):
        self.debug = debug

    def __enter__(self):
        if not self.debug:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.debug:
            sys.stdout.close()
            sys.stdout = self._original_stdout


class ResultGeneration:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.evaluation = Evaluation(debug)

    def __call__(self, model, data_loader, loss, device):
        with HiddenPrints(self.debug):
            _, results = self.evaluation(loss, data_loader, model, device)

        y, x, names = zip(*results)

        names_joined = list(itertools.chain.from_iterable([name.tolist() for name in names]))
        predictions_joined = list(itertools.chain.from_iterable([value.tolist() for value in x]))

        df = pd.DataFrame(columns=['sales'])

        for item_index in np.unique(np.array(names_joined)):
            pred_index = names_joined.index(item_index)
            prediction = predictions_joined[pred_index]

            df.loc[item_index] = prediction

        return df