# standard packages
import json
import statistics
import time
import random
# non-standard packages, i.e., you might need to install them
import numpy as np
import openai

class OpenAIDataProcessor:

#### ALL CODE THAT INTERACTS WITH THE OPENAI API SHOULD GO IN THE SECTION BELOW

    # Configure properties that will apply to all OpenAI requests
    @staticmethod
    def _configure_openai(api_key, timeout):
        openai.api_key = api_key
        openai.timeout = timeout
    
    # Send a request to the specified <model> containing <messages>, and return the
    # response. If there is a timeout error or a communication error, will keep
    # retrying until there is a successful response.
    @staticmethod
    def _make_openai_request(model, messages):
        # If the request to OpenAI times out, this will keep making the same request
        # until it doesn't time out. If the timeout value is unreasonably low, this 
        # can lead to an infinite loop.
        while True:
            try:
                raw_response = openai.chat.completions.create(model=model, messages=messages)
                response = json.loads(raw_response.model_dump_json())
                break
            except openai.APITimeoutError as e:
                print('Request timed out, retrying...')
            except openai.APIConnectionError as e:
                print('Error communicating with OpenAI, retrying...')
        return response

#### ALL CODE THAT INTERACTS WITH THE OPENAID API SHOULD GO IN THE SECTION ABOVE
    
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
            response = self._make_openai_request(
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

        # ADD SOME CODE HERE TO CHECK FOR EXCEEDING THE TOKEN LIMIT
        return response['choices'][0]['message']['content']
        
    # Asks the model to go through each item in the list <items>, apply the processing
    # specified in the instructions, and return the result.  <mode> can be 'live' or
    # 'simulated', and controls whether we send requests to OpenAI and get a real
    # response, or just provide a simulated response (this can be useful for testing
    # purposes).
    # 
    # Also creates an attribute on the processor's _data object:
    # <output> - A list containing the responses from the model for each item
    def process(self, items, mode='live'):
        self._configure_openai(api_key=self.api_key, timeout=self.timeout)
        self.mode = mode
        n_items = len(items)
        self._data['output'] = []
        for i, item in enumerate(items):
            self._data['output'].append(self._process_item(item))
            print('Progress: {:.0%}'.format((i+1)/n_items), end='\r')
        print('')
        return self._data['output']
    
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
        return (np.array(self._data['output']) == np.array(targets)).astype(dtype=int)

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
