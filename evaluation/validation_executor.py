import litellm
from api_client.client import Client
from api_client.api_conf import APIConfiguration
import json
from collections import defaultdict
import copy
import re


OUTPUT_SCHEMA_KEYS = {"answer", "thoughts"}
MAX_RETRIES = 12

def get_response_json(response: str) -> dict:
    """
    Parses a string to JSON. As a fallback, it attempts to extract a JSON-like
    substring from the raw string if direct parsing fails.
    """
    try:
        out = json.loads(response)
    except json.JSONDecodeError:
        # If JSON parsing fails, attempt to extract JSON-like content using regex
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            out = json.loads(json_str)
        else:
            # If no JSON object can be found, re-raise the original error
            raise
    return out

class ValidationExecutor:
    """
    Simple executor class to run a single probing step with an agent.
    """
    def __init__(self, agent, openai_tools, terminate_function, target_function, api_conf: APIConfiguration):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        self.target_function = target_function

        self.api_conf = api_conf

        self.client = Client(api_conf=self.api_conf)

    def step(self, observation:str) -> bool | None:
        # Add the user's observation to the agent's history and print it.
        self.agent.update_history({"role": "user", "content": observation})
        print(f"[`user` -> `{self.agent.name.lower()}`]")
        print(f"'{observation}'\n")

        i = 0
        while True:
            probe_status = None
            i += 1
            if i > MAX_RETRIES:
                print(f"Maximum number of retries ({i-1}) reached. Exiting...\n")
                break

            response = self._call_llm()
            processed_response, probe_status = self._process_llm_response(response)
            if processed_response is not None:
                assert isinstance(probe_status, str)
                assert probe_status in ["yes", "no"]
                probe_status = probe_status == "yes"
                self.agent.update_history(processed_response)
                break

        return probe_status

    def _call_llm(self):
        """
        Calls the language model with the current agent history and available tools.
        Includes a retry mechanism for transient API errors like rate limiting.
        """
        retries = 0
        out = None
        while True:
            retries += 1
            if retries > MAX_RETRIES:
                print(f"Maximum number of retries reached when calling LLM. Exiting ...\n")
                raise Exception("Maximum number of retries reached when calling LLM.")
            try:
                response = self.client(query=self.agent.history, tools=self.openai_tools, tool_choice="auto")
                # Extract the message object from the API response.
                out = response.choices[0].message
                break
            except litellm.RateLimitError:
                print(f"Rate limit error when calling LLM. Retrying...\n")
                continue
            except Exception as e:
                print(f"Error calling LLM: {e}\n")
                continue
        return out

    def _process_llm_response(self, response):
        if response is None or not response.content:
            return None

        assistant_content = response.content

        output = None
        probe_status = None
        try:
            # The message is expected to be a structured JSON object.
            assistant_content_json = get_response_json(assistant_content)
            assert set(assistant_content_json.keys()) == OUTPUT_SCHEMA_KEYS
            probe_status = assistant_content_json["answer"].strip().lower()
            assert probe_status in ["yes", "no"]
            # Make sure the thoughts field is sufficiently detailed.
            assert len(assistant_content_json["thoughts"].strip()) > 50
            output = {"role": "assistant", "content": assistant_content}
            print(f"[`{self.agent.name.lower()}` -> `user`]")
            print(f"'{json.dumps(assistant_content_json, indent=2)}'\n")
        except (json.JSONDecodeError, AssertionError) as e:
            # If parsing or validation fails, we treat it as a recoverable error and trigger a retry.
            print(f"Error with assistant message format: {e}. Retrying...\n")

        return output, probe_status