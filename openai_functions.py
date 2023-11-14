import json
import openai

# Configure properties that will apply to all OpenAI requests
def configure(api_key, timeout):
    openai.api_key = api_key
    openai.timeout = timeout
    
# Send a request to the specified <model> containing <messages>, and return the
# response. If there is a timeout error or a communication error, will keep
# retrying until there is a successful response.
def make_request(model, messages):
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