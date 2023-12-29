# standard packages
import json
import statistics
import time
import random
# non-standard packages, i.e., you might need to install them
import numpy as np

from .data_processor import DataProcessor

class DataCoder (DataProcessor):

    def _configure_instructions(self, instructions):
        # In addition to whatever criteria are provided in <instructions>, we ask for the 
        # response back in JSON format so that we can easily identify the yes/no answer
        # for purposes of scoring the response. 
        self.instruction_messages=[
            {"role": "system", "content": instructions + 
             """ Respond in JSON format with the key 'answer' for your yes/no response, and 
             the key 'explanation' for an explanation of the response."""}
        ]

    def _configure_examples(self, examples):
        # Configure example item/response pairs to be used for few-shot learning. They will be
        # provided to the model after the instructions and before the item to be classified.
        # Should be in the format a list of dicts, where each dict has keys 'item', 'target',
        # and 'explanation'. The 'target' value should contain a 0 or 1.
        self.example_messages = []
        if examples:
            for example in examples:
                self.example_messages.append({'role': 'user', 'content': example['item'] })
                self.example_messages.append({'role': 'assistant', 'content': json.dumps({
                        'answer': 'yes' if example['target']=='1' else 'no', 
                        'explanation': example['explanation']}
                )})

    # Initializes a data structure for storing output data for each item
    def _init_output(self, n_items):
        self._data['output'] = [None] * n_items
        self._data['explanation'] = [None] * n_items
    
    # Takes <response> and stores it as part of the output data for item <i>
    def _store_output(self, i, response):
        output = json.loads(response['choices'][0]['message']['content'])
        self._data['output'][i] = (np.array(output['answer'].lower().rstrip('.')) == 'yes').astype(dtype=int).tolist()
        self._data['explanation'][i] = output['explanation']

    def explanations(self):
        return self._data['explanation']