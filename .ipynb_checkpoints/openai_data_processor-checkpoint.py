# standard packages
import json
import statistics
import time
import random
# non-standard packages, i.e., you might need to install them
import numpy as np
import openai

from openai_data_object import OpenAIDataObject

class OpenAIDataProcessor (OpenAIDataObject):
    
    def __init__(self, api_key, model, instructions, examples=None):
        self._data = {}
        self.api_key = api_key
        self.model = model
        # In addition to whatever criteria are provided in <instructions>, we ask for the 
        # response back in JSON format so that we can easily identify the yes/no answer
        # for purposes of scoring the response. 
        self.instruction_messages=[
            {"role": "system", "content": instructions }
        ]
        # Configure example item/response pairs to be used for few-shot learning. They will be
        # provided to the model after the instructions and before the item to be classified.
        # Should be in the format a list of dicts, where each dict has keys 'item', 'target',
        # and 'explanation'. The 'target' value should contain a 0 or 1.
        self.example_messages = []
        if examples:
            for example in examples:
                self.example_messages.append({'role': 'user', 'content': example['item']})
                self.example_messages.append({'role': 'assistant', 'content': example['target']})
    
    # This function is used internally to process a single item
    def _process_item(self, item):
        if self.mode == 'live':
            # If the request to OpenAI times out, this will keep making the same request until it doesn't time
            # out. If the timeout value is unreasonably low, this can lead to an infinite loop.
            while True:
                try:
                    # CHANGE IN OPENAI 1.0
                    raw_completion = openai.chat.completions.create(
                        model = self.model, 
                        messages = self.instruction_messages 
                            + self.example_messages 
                            + [{'role': 'user', 'content': item }]
                    )
                    # CHANGE IN OPENAI 1.0
                    completion = json.loads(raw_completion.model_dump_json())
                    break
                # CHANGE IN OPENAI 1.0
                except openai.APITimeoutError as e:
                    print('Request timed out, retrying...')
                # CHANGE IN OPENAI 1.0
                except openai.APIConnectionError as e:
                    print('Error communicating with OpenAI, retrying...')
        # This provides a way to test that your script runs properly without actually making a call to the
        # OpenAI API. It provides a random response along with a small delay.
        elif self.mode == 'simulated':
            time.sleep(.2)
            completion = {'choices': [{'message': {'content': item}}], 'usage': {}}

        # ADD SOME CODE HERE TO CHECK FOR EXCEEDING THE TOKEN LIMIT
        return completion['choices'][0]['message']['content']
        
    # Asks the model to go through each item in the list <items>, apply the processing specified in the
    # instructions, and return the result. <timeout> controls how long we wait (in seconds) for a 
    # response from OpenAI before giving up and trying again. <mode> can be 'live' or 'simulated', and 
    # controls whether we send requests to OpenAI and get a real response, or just provide a simulated 
    # response (this can be useful for testing purposes).
    # 
    # Also creates an attribute on the processor's _data object:
    # <output> - A list containing the responses from the model for each item
    def process(self, items, timeout=30, mode='live'):
        openai.api_key = self.api_key
        # CHANGE IN OPENAI 1.0
        openai.timeout = timeout
        self.mode = mode
        n_items = len(items)
        self._data['output'] = []
        for i, item in enumerate(items):
            self._data['output'].append(self._process_item(item))
            print('Progress: {:.0%}'.format((i+1)/n_items), end='\r')
        print('')
        return self._data['output']
    
    # Using the results from the last call to <process>, compares the model responses to the 
    # values in <targets>, and returns a list with the outcome of that comparison. <targets> 
    # should be the same length that <items> was in the last call to <code>. 
    #
    # Also creates an attribute on the coder's _data object called <scoring_matrix>. This is an
    # array with rows for each item, and columns for each run, and where the values are 0
    # or 1 to indicate whether the model response matched the target for that item and that
    # run.
    def score(self, targets):
        return (np.array(self._data['output']) == np.array(targets)).astype(dtype=int)

    # This is an internal helper function that generates a classification matrix (described
    # under <classification_metrics>) given a matrix of the model responses for each item
    # and each run (as 0's and 1's), and a list of targets (also 0's and 1's).
    @staticmethod
    def _create_classification_matrix(response_matrix, targets):
        rm = response_matrix
        at = np.array(targets).reshape(-1,1)
        cm = {
            'truepos': np.logical_and(rm, at).astype(dtype=int),
            'trueneg': np.logical_and(np.logical_not(rm), np.logical_not(at)).astype(dtype=int),
            'falsepos': np.logical_and(rm, np.logical_not(at)).astype(dtype=int),
            'falseneg': np.logical_and(np.logical_not(rm), at).astype(dtype=int)
        }
        return cm
    
    # This is an internal helper function that calculates accuracy, precision, and recall given
    # a list of model responses and a list of the corresponding targets. Both lists should
    # be the same length and contain only 0 and 1 values.
    @staticmethod
    def _calculate_classification_metrics(responses, targets):
        ar = np.array(responses)
        at = np.array(targets)
        # to simplify calculations, we treat the 1 and 0 values in the arrays as booleans,
        # and use logical operations on them
        n_truepos = np.logical_and(ar, at).sum()
        n_trueneg = np.logical_and(np.logical_not(ar), np.logical_not(at)).sum()
        n_falsepos = np.logical_and(ar, np.logical_not(at)).sum()
        n_falseneg = np.logical_and(np.logical_not(ar), at).sum()
        accuracy = (n_truepos + n_trueneg) / (n_truepos + n_trueneg + n_falsepos + n_falseneg)
        precision = n_truepos / (n_truepos + n_falsepos)
        recall = n_truepos / (n_truepos + n_falseneg)
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall}
    
    # Using the results from the last call to <code>, along with the provided <targets>,
    # calculates accuracy, precision, and recall metrics. <scoring_type> controls how the
    # metrics are calculated if there were multiple model runs. If it is 'categorical',
    # the metrics are calculated based on the most frequent model response for each item.
    # If it is 'probability', the metrics are calculated separately for each run, and then
    # averaged.
    # 
    # Also creates an attribute on the coder's _data object called <classification_matrix>. This 
    # is a dict containing four matrices, one for each type of outcome of a yes/no classification:
    #   true positive - target is yes, model said yes
    #   true negative - target is no, model said no
    #   false positive - target is no, model said yes
    #   false negative - target is yes, model said no
    # Each matrix has a row for each item, and a column for each run of the model. A 1 in
    # a cell indicates that for that item and that run of the model, this was the outcome.
    # For example, a 1 in cell 5,2 of the true positive matrix means that for item 5 and
    # run 2, the model's response counts as a true positive.
    def classification_metrics(self, targets, scoring_type='categorical'):
        self._data['classification_matrix'] = self._create_classification_matrix(
            self._data['coding_matrix'], targets)
        
        if scoring_type == 'categorical':
            # get the most common response for each item
            responses = self.get_coding(coding_type='categorical')
            return self._calculate_classification_metrics(responses, targets)
        elif scoring_type == 'probability':
            n_runs = self._data['coding_matrix'].shape[1]
            metrics_list = []
            for i in range(0, n_runs):
                responses = self._data['coding_matrix'][:,i]
                metrics_list.append(self._calculate_classification_metrics(responses, targets))
            return {
                'accuracy': statistics.mean([metrics['accuracy'] for metrics in metrics_list]),
                'precision': statistics.mean([metrics['precision'] for metrics in metrics_list]),
                'recall': statistics.mean([metrics['recall'] for metrics in metrics_list])
            }  
        else:
            raise ValueException('Scoring type must be either categorical or probability')
    
    # Saves the _data attribute of the coder object to <filename>. Most importantly, this
    # includes the results from the last call to <code>, so you can capture this information
    # and restore it in another session.
    def dump(self, filename):
        with open(filename, "wb") as outfile:
            pickle.dump(self._data, outfile)

    # Restores the _data attribute of the coder object that was previously dumped to <filename>
    def restore(self, filename):
        with open(filename, "rb") as infile:
            self._data = pickle.load(infile)