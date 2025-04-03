import numpy as np
from .scorer import Scorer

class ClassificationScorer (Scorer):

    def __init__(self, output, targets):
        super().__init__(output, targets)

        ar = np.array(output)
        at = np.array(targets)
        # to simplify calculations, we treat the 1 and 0 values in the arrays as booleans,
        # and use logical operations on them
        self._n_truepos = np.logical_and(ar, at).sum()
        self._n_trueneg = np.logical_and(np.logical_not(ar), np.logical_not(at)).sum()
        self._n_falsepos = np.logical_and(ar, np.logical_not(at)).sum()
        self._n_falseneg = np.logical_and(np.logical_not(ar), at).sum()

    # Returns the proportion of all positive responses where that was the correct classification
    def precision(self):
        return self._n_truepos / (self._n_truepos + self._n_falsepos)

    # Returns the proportion of all items that should be classified as positive that were actually classified as positive
    def recall(self):
        return self._n_truepos / (self._n_truepos + self._n_falseneg)
