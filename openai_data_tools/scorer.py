import numpy as np

class Scorer:

    # Creates a Scorer object for comparing <output> with <targets>
    def __init__(self, output, targets):
        self._data = {}
        self._data['scoring'] = (np.array(output) == np.array(targets)).astype(dtype=int)
    
    # Returns a list with 1 for each element i where <output[i]> matches <targets[i]> and 0 for each element where they don't match
    def scores(self):
        return self._data['scoring'].tolist()

    # Returns the proportion of items where <output[i]> matches <targets[i]>
    def accuracy(self):
        return np.mean(self._data['scoring'])