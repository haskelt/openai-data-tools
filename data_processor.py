# standard packages
import statistics
import time
import random
# non-standard packages, i.e., you might need to install them
import numpy as np
import openai_functions as ai

class DataProcessor:
    
    def __init__(self, model, instructions, examples=None, api_key=None, timeout=30):
        self._data = {}
        self.model = model
        self.api_key = api_key
        # <timeout> controls how long we wait (in seconds) for a  response from OpenAI
        # before giving up and trying again.
        self.timeout = timeout
        self.instruction_messages=[
            {"role": "system", "content": instructions }
        ]
        # Configure example item/response pairs to be used for few-shot learning. They 
        # will be provided to the model after the instructions and before the item to be 
        # classified.
        # <examples> should be in the format a list of dicts, where each dict has keys
        # 'item' and 'target', where target is the desired response.
        self.example_messages = []
        if examples:
            for example in examples:
                self.example_messages.append({'role': 'user', 'content': example['item']})
                self.example_messages.append({'role': 'assistant', 'content': example['target']})
    
    # This function is used internally to process a single item
    def _process_item(self, item):
        if self.mode == 'live':
            response = ai.make_request(
                self.model, 
                messages = self.instruction_messages 
                   + self.example_messages 
                   + [{'role': 'user', 'content': item}] 
            )
        # This provides a way to test that your script runs properly without actually
        # making a call to the OpenAI API. It provides a random response along with a
        # small delay.
        elif self.mode == 'simulated':
            time.sleep(.2)
            response = {'choices': [{'message': {'content': item}}], 'usage': {}}

        return response

    # Initializes a data structure for storing data for each item
    def _init_item_data(self, n_items):
        self._data['item_data'] = [None] * n_items
    
    # Takes <response> and stores it as part of the item data for item <i>
    def _store_item_data(self, i, response):
        self._data['item_data'][i] = response

    # Returns a list with the model output for each item based on the last call to <process>
    def _get_output(self):
        return [item['choices'][0]['message']['content'] for item in self._data['item_data']]
    
    # Asks the model to go through each item in the list <items>, apply the processing
    # specified in the instructions, and return the result.  <mode> can be 'live' or
    # 'simulated', and controls whether we send requests to OpenAI and get a real
    # response, or just provide a simulated response (this can be useful for testing
    # purposes).
    # 
    # Also creates an attribute on the processor's _data object:
    # <output> - A list containing the responses from the model for each item
    def process(self, items, mode='live'):
        ai.configure(api_key=self.api_key, timeout=self.timeout)
        self.mode = mode
        n_items = len(items)
        self._init_item_data(n_items)
        for i, item in enumerate(items):
            response = self._process_item(item)
            self._store_item_data(i, response)
            print('Progress: {:.0%}'.format((i+1)/n_items), end='\r')
        print('')
        return self._get_output()
    
    # Using the results from the last call to <process>, compares the model responses
    # to the values in <targets>, and returns a list with the outcome of that 
    # comparison. <targets> should be the same length that <items> was in the last 
    # call to <process>. 
    #
    # ISN'T ACTUALLY DOING THIS PART CURRENTLY
    # Also creates an attribute on the coder's _data object called <scoring_matrix>. 
    # This is an array with rows for each item, and columns for each run, and where 
    # the values are 0 or 1 to indicate whether the model response matched the target 
    # for that item and that run.
    def score(self, targets):
        return (np.array(self._get_output()) == np.array(targets)).astype(dtype=int)

    # Saves the _data attribute of the object to <filename>, so you can restore the
    # state of the object in another session.
    def dump(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self._data, outfile)

    # Restores the _data attribute of the object that was previously dumped to 
    # <filename>
    def restore(self, filename):
        with open(filename, "rb") as infile:
            self._data = pickle.load(infile)
