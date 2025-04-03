import json
import openai

class OpenAI:
    client = None

    # Configure properties that will apply to all OpenAI requests
    @classmethod
    def configure(cls, api_key, model, timeout=30):
        openai.api_key = api_key
        # <timeout> controls how long we wait (in seconds) for a  response from OpenAI
        # before giving up and trying again.
        openai.timeout = timeout
        cls.model = model
        cls.client = openai.OpenAI()
    
    # Send a request to the specified <model> containing <messages>, and return the
    # response. If there is a timeout error or a communication error, will keep
    # retrying until there is a successful response.
    @classmethod
    def make_request(cls, messages):
        # If the request to OpenAI times out, this will keep making the same request
        # until it doesn't time out. If the timeout value is unreasonably low, this 
        # can lead to an infinite loop.
        while True:
            try:
                raw_response = cls.client.chat.completions.create(model=cls.model, messages=messages)
                response = json.loads(raw_response.model_dump_json())
                break
            except openai.APITimeoutError as e:
                print('Request timed out, retrying...')
            except openai.APIConnectionError as e:
                print('Error communicating with OpenAI, retrying...')
        return response