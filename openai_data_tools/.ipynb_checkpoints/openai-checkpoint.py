import json
import openai

class OpenAI:
    model = None
    client = None

    # Configure properties that will apply to all OpenAI requests
    @classmethod
    def configure(cls, api_key, model, azure_endpoint=None, api_version=None, timeout=30):
        cls.model = model
        if azure_endpoint:
            if api_version:
                cls.client = openai.AzureOpenAI(api_key=api_key, timeout=timeout, azure_endpoint=azure_endpoint, api_version=api_version)
            else:
                raise Exception('An api_version argument is required when using an Azure endpoint')
        else:
            cls.client = openai.OpenAI(api_key=api_key, timeout=timeout)
    
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