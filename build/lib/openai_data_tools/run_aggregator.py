import statistics
import numpy as np

class RunAggregator:

    # Creates a RunAggregator object that lets you convert the output of multiple model runs into a single set of output values
    def __init__(self, outputs):
        self._outputs = np.array(outputs, dtype=str)

    # Returns a list with the most common output for each item. If multiple output values are equally common, returns the one that is encountered first
    # (in run order); you probably want to avoid that situation if possible.
    def output(self):
        return [statistics.mode(col) for col in self._outputs.T]
        
    # Calculates the average agreement in output across runs
    def agreement(self):
        n_runs = self._outputs.shape[0]
        agreements = []
        for i in range(0, n_runs - 1):
            for j in range(i + 1, n_runs):
                agreements.append((self._outputs[i,:] == self._outputs[j,:]).mean())
        return np.mean(agreements)
