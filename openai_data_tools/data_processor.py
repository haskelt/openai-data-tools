# standard packages
import statistics
import time
import random
import pickle
# non-standard packages, i.e., you might need to install them
import numpy as np
from . import openai_functions as ai

class DataProcessor:
    
    def __init__(self, model, instructions, examples=None, api_key=None, timeout=30):
        self._data = {}
        self.model = model
        self.api_key = api_key
        # <timeout> controls how long we wait (in seconds) for a  response from OpenAI
        # before giving up and trying again.
        self.timeout = timeout
        self._configure_instructions(instructions)
        self._configure_examples(examples)

    def _configure_instructions(self, instructions):
        self.instruction_messages=[
            {"role": "system", "content": instructions }
        ]

    def _configure_examples(self, examples):
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
    
    # Initializes a data structure for storing output data for each item
    def _init_output(self, n_items):
        self._data['output'] = [None] * n_items
    
    # Takes <response> and stores it as part of the output data for item <i>
    def _store_output(self, i, response):
        self._data['output'][i] = response['choices'][0]['message']['content']

    # Returns a list with the model output for each item based on the last call to <process>
    def _get_output(self):
        return self._data['output']

    # Processes a single item
    def _process_item(self, item):
        if self.mode == 'live':
            response = ai.make_request(
                self.model, 
                messages = self.instruction_messages 
                   + self.example_messages 
                   + [{'role': 'user', 'content': item}] 
            )
        # This provides a way to test that your script runs properly without actually
        # making a call to the OpenAI API. It returns the original item without changes
        # as output. To be more realistic, it also generates a small delay.
        elif self.mode == 'simulated':
            time.sleep(.2)
            response = {'choices': [{'message': {'content': item}}], 'usage': {}}

        return response
    
    # Processes each of the items in <items>
    def _process_items(self, items):
        n_items = len(items)
        for i, item in enumerate(items):
            response = self._process_item(item)
            self._store_output(i, response)
            print('Progress: {:.0%}'.format((i+1)/n_items), end='\r')
        print('')
    
    # Asks the model to go through each item in the list <items>, apply the processing
    # specified in the instructions, and return the result.  <mode> can be 'live' or
    # 'simulated', and controls whether we send requests to OpenAI and get a real
    # response, or just provide a simulated response (this can be useful for testing
    # purposes).
    # 
    # Also creates an attribute on the processor's _data object:
    # <item_data> - A list containing the responses from the model for each item
    def process(self, items, mode='live'):
        ai.configure(api_key=self.api_key, timeout=self.timeout)
        self.mode = mode
        self._init_output(len(items))
        self._process_items(items)
        return self._get_output()
    
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
