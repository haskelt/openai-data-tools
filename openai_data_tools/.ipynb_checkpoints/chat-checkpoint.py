# this works well with Jupyter but may not work well on a command line
from IPython.display import display, Markdown
# non-standard packages, i.e., you might need to install them
from .openai import OpenAI

class Chat:
    
    def __init__(self, instructions):
        self._data = {}
        self._configure_instructions(instructions)
        self.clear_history()

    def _configure_instructions(self, instructions):
        self.instruction_messages=[
            {"role": "system", "content": instructions + " Your response should be in Markdown format." }
        ]

    # Sends a prompt to the model and returns the response, along with adding the prompt/response pair to the message history
    def send(self, prompt):
        raw_response = OpenAI.make_request(
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
    
