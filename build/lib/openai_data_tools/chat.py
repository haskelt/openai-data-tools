# this works well with Jupyter but may not work well on a command line
from IPython.display import display, Markdown
# non-standard packages, i.e., you might need to install them
from . import openai_functions as ai

class Chat:
    
    def __init__(self, model, instructions, api_key=None, timeout=30):
        self._data = {}
        self.model = model
        self.api_key = api_key
        # <timeout> controls how long we wait (in seconds) for a  response from OpenAI
        # before giving up and trying again.
        self.timeout = timeout
        self._configure_instructions(instructions)
        self.clear_history()

    def _configure_instructions(self, instructions):
        self.instruction_messages=[
            {"role": "system", "content": instructions + " Your response should be in Markdown format." }
        ]

    # Sends a prompt to the model and returns the response, along with adding the prompt/response pair to the message history
    def send(self, prompt):
        ai.configure(api_key=self.api_key, timeout=self.timeout)
        raw_response = ai.make_request(
            self.model, 
            messages = self.instruction_messages 
               + self.message_history 
               + [{'role': 'user', 'content': prompt}] 
        )
        response = raw_response['choices'][0]['message']['content']
        self.message_history.append({'role': 'user', 'content': prompt})
        self.message_history.append({'role': 'assistant', 'content': response})
        display(Markdown(response))

    # Clears the message history, so the AI won't see previous prompts and responses
    def clear_history(self):
        self.message_history = []
    
