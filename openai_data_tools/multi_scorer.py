import numpy as np

class MultiScorer:

    # Creates a MultiScorer object for comparing <outputs> from multiple runs of the model with <targets>. <outputs> should be a list where each item is
    # the output from an individual run.
    def __init__(self, outputs, targets, method='mode'):
        # calculate the average output for each item as the mean of the outputs for each individual run
        if method == 'mean':
            self._avg_outputs = np.mean(np.array(outputs, dtype=int), axis=0)
        # calculate the average output for each item as the modal output across runs
        elif method == 'mode':
            self._avg_outputs = np.round(np.mean(np.array(outputs, dtype=int), axis=0)).astype(dtype=int)
        # score each item as 1 minus the absolute value of the difference between the average output and the target output
        self._scoring = 1 - np.absolute(np.array(targets, dtype=float) - self._avg_outputs)

    # Returns a list with the average output for each item
    def outputs(self):
        return self._avg_outputs.tolist()
        
    # Returns a list with the score for each item
    def scores(self):
        return self._scoring.tolist()

    # Return a list with the average of the scores for each item
    def accuracy(self):
        return np.mean(self._scoring)
