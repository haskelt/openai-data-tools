# standard packages
import pickle
# non-standard packages, i.e., you might need to install them
#import openai

class OpenAIDataObject:
    
    # Saves the _data attribute of the object to <filename>, so you can restore the state of the object
    # in another session.
    def dump(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self._data, outfile)

    # Restores the _data attribute of the object that was previously dumped to <filename>
    def restore(self, filename):
        with open(filename, "rb") as infile:
            self._data = pickle.load(infile)