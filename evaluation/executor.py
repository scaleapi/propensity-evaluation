import litellm
from api_client.client import Client
from api_client.api_conf import APIConfiguration
import json
from collections import defaultdict
import copy
import re


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


class Executor:
    """
    Manages the interaction loop between an agent and the user/environment.

    The Executor is responsible for taking user input, orchestrating the
    conversation with the LLM, handling tool calls, managing state, and
    determining when an interaction or task is complete.
    """

    def __init__(self, agent, openai_tools, terminate_function, target_function, api_conf: APIConfiguration):
        self.agent = agent
        self.openai_tools = openai_tools
        self.terminate_function = terminate_function
        self.target_function = target_function

        # --- State and retry management ---
        # Maximum number of times a single tool can be called in a turn before being disabled.
        self.max_retries = 3
        # Maximum number of LLM interactions (tool call -> response) in a single turn.
        self.max_interaction_retries = 12
        # Tracks the number of consecutive non-tool messages from the LLM to prevent monologues.
        self.number_of_consecutive_messages = 0
        # Tracks retries for the current interaction turn.
        self.curr_interaction_retries = 0
        # Tracks the number of calls for each tool within a single turn.
        self.function_call_count = defaultdict(int)

        self.api_conf = api_conf
        self.client = Client(self.api_conf)

    def step(self, observation: str) -> bool:
        """
        Executes a single interaction step for the agent.

        This method orchestrates a full turn of conversation. It starts with the user's
        observation, then enters a loop where it calls the LLM, processes its response
        (which could be a tool call or a message), executes the tool if needed, and
        feeds the result back to the LLM. This continues until the LLM terminates
        the interaction or a limit is reached.

        Args:
            observation: The input string from the user or environment.

        Returns:
            A boolean indicating if the `terminate_function` was called, signifying
            that the overall task is complete.
        """
        # Start of a new interaction turn, reset counters.
        self.number_of_consecutive_messages = 0
        self.curr_interaction_retries = 0
        self.function_call_count = defaultdict(int)

        # Add the user's observation to the agent's history and print it.
        self.agent.update_history({"role": "user", "content": observation})
        print(f"[`user` -> `{self.agent.name.lower()}`]")
        print(f"'{observation}'\n")

        # Create a mutable copy of tools for this turn, allowing us to disable failing tools.
        self.filtered_tools = copy.deepcopy(self.openai_tools)
        # Initial call to the LLM with the user's prompt.
        response = self._call_llm()

        # Process the response to get the initial action and determine if we need to loop.
        function_name, processed_response, retry_call = self._process_llm_response(response)

        # --- Main Interaction Loop for a Single User Turn ---
        # This loop continues as long as the LLM's response requires a follow-up action
        # (e.g., calling a tool and sending back the result) before the turn is over.
        while retry_call:
            self.curr_interaction_retries += 1
            # Safety break: prevent infinite loops within a single turn.
            if self.curr_interaction_retries > self.max_interaction_retries:
                print(f"Maximum number of interaction retries reached in episode. Exiting...\n")
                return False

            # Case 1: A recoverable error occurred during response processing (e.g., bad JSON).
            # `processed_response` is None, so we simply retry by calling the LLM again.
            if processed_response is None:
                response = self._call_llm()
                function_name, processed_response, retry_call = self._process_llm_response(response)
            # Case 2: The LLM explicitly called the terminate function.
            # The loop will end because `_process_llm_response` sets `retry_call` to False.
            elif function_name == self.terminate_function:
                self.agent.update_history(processed_response)
            # Case 3: A tool was called or a consecutive message was sent.
            # Update history with the action and its result, then call the LLM again to
            # let it process the new information.
            else:
                self.agent.update_history(processed_response)
                response = self._call_llm()
                function_name, processed_response, retry_call = self._process_llm_response(response)

        # After the loop, update history with the final response if one was generated.
        if processed_response is not None:
            self.agent.update_history(processed_response)

        # The step is considered successful/complete if the final action was to terminate.
        return function_name == self.terminate_function

    def _call_llm(self):
        """
        Calls the language model with the current agent history and available tools.
        Includes a retry mechanism for transient API errors like rate limiting.
        """
        retries = 0
        while True:
            retries += 1
            if retries > 5:
                print(f"Maximum number of retries reached when calling LLM. Exiting ...\n")
                raise Exception("Maximum number of retries reached when calling LLM.")
            try:
                response = self.client(query=self.agent.history, tools=self.filtered_tools, tool_choice="auto")
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
        """
        Processes the response from the LLM to determine the next action.

        This is the core logic that interprets the LLM's output. It handles three main scenarios:
        1. Tool Call: The LLM wants to execute a function.
        2. Content Message: The LLM sends a text message to the user, which is expected
           to be a structured JSON object.
        3. Empty/Invalid Response: The response is malformed.

        Returns:
            A tuple `(function_name, processed_response, retry_call)` where:
            - `function_name` (str|None): Name of the tool called, if any.
            - `processed_response` (dict|list|None): Formatted message(s) to be added to the history.
            - `retry_call` (bool): If `True`, the main loop in `step()` should continue.
                                   If `False`, the current interaction turn is over.
        """
        if response is None:
            return None, None, False

        # --- Scenario 1: LLM responded with a tool call ---
        if response.tool_calls:
            # This is a tool interaction, not a direct message, so reset the consecutive message counter.
            self.number_of_consecutive_messages = 0
            # By default, after a tool call, we need to loop and call the LLM again with the tool's output.
            retry_call = True
            # Note: We currently only process the first tool call if multiple are received.
            if len(response.tool_calls) > 1:
                print(
                    f"Received multiple tool calls, only processing the first: {response.tool_calls}\n")
            tool_call = response.tool_calls[0]

            tool_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                print(f"Error decoding arguments JSON: {e}\n")
                arguments = {}

            # Handle the special termination tool call. This ends the interaction turn.
            if tool_name == self.terminate_function:
                print(f"[{self.agent.name.lower()}: tool call -->] {tool_name}\n")
                tool_output = None
                retry_call = False
            else:
                # For any other tool, execute it and get the output.
                print(f"[{self.agent.name.lower()}: tool call -->] {tool_name}")
                if tool_name not in self.agent.tools:
                    tool_output = f'{{"status": "404", "message": "Function {tool_name} not found in tools. Please try again later."}}'
                else:
                    tool = self.agent.tools[tool_name]
                    tool_output = tool(**arguments)
                    self.function_call_count[tool_name] += 1
                    # If a tool is called too many times, it might be stuck in a loop.
                    # Disable it for the rest of this turn to encourage the LLM to try something else.
                    if self.function_call_count[tool_name] > self.max_retries:
                        tool_output = f'{{"status": "429", "message": "Maximum number of retries reached for the tool {tool_name}. Please try again later."}}'
                        self.filtered_tools = [t for t in self.filtered_tools if t['function']['name'] != tool_name]

                print(f"[tool response -->] {tool_output}\n")

            # Format the tool call and its result into messages for the agent's history.
            tool_result_messages = []
            # Append the assistant's original thought/message if it exists.
            if response.content:
                tool_result_messages.append({"role": "assistant", "content": response.content})
            # Append the tool call itself and the corresponding tool output.
            tool_result_messages.extend([
                {"role": "assistant", "tool_calls": [{"id": tool_call.id, "type": tool_call.type,
                                                      "function": {"name": tool_call.function.name,
                                                                   "arguments": tool_call.function.arguments}}]},
                {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_output, indent=2)}
            ])
            return tool_name, tool_result_messages, retry_call

        # --- Scenario 2: LLM responded with a direct message ---
        elif response.content:
            assistant_content = response.content
            # This message is expected to be a structured JSON object.
            try:
                assistant_content_json = get_response_json(assistant_content)
                # Validate that the JSON contains the required keys for our interaction protocol.
                assert set(assistant_content_json.keys()) == {"thoughts", "message", "interaction_status"}
                # Validate the 'interaction_status', which controls the flow.
                assert assistant_content_json['interaction_status'] in ["continue", "terminate"]
            except (json.JSONDecodeError, AssertionError) as e:
                # If parsing or validation fails, we treat it as a recoverable error and trigger a retry.
                # A guiding message can be sent to help the LLM correct its output format.
                print(f"Error with assistant message format: {e}. Retrying...\n")
                error_message = "Backend System: Invalid response format. Response MUST be a JSON object with keys 'thoughts', 'message', and 'interaction_status', where 'interaction_status' is either 'continue' or 'terminate'. Try again."
                return None, {"role": "user", "content": error_message}, True

            print(f"[`{self.agent.name.lower()}` -> `user`]")
            print(f"'{json.dumps(assistant_content_json, indent=2)}'\n")

            # If the agent wants to terminate, the turn is over. Set retry_call to False.
            if assistant_content_json['interaction_status'] == "terminate":
                self.number_of_consecutive_messages = 0
                return None, {"role": "assistant", "content": assistant_content}, False

            # If the agent wants to continue, it's sending an intermediary message to the user.
            # To prevent the agent from getting stuck in a monologue, we limit consecutive messages.
            if self.number_of_consecutive_messages < 2:
                self.number_of_consecutive_messages += 1
                # We send a system message to acknowledge receipt and prompt the LLM to continue its work.
                ack_message = 'Backend System: Message received. Continue with the next tool call or terminate the interaction.'
                print(f"[`user` -> `{self.agent.name.lower()}`]")
                print(f"'{ack_message}'\n")
                # Return the assistant's message and our system acknowledgement, and set retry_call to True.
                return None, [{"role": "assistant", "content": assistant_content},
                              {"role": "user", "content": ack_message}], True
            else:
                # If the consecutive message limit is reached, we break the loop to avoid monologues.
                self.number_of_consecutive_messages = 0
                return None, {"role": "assistant", "content": assistant_content}, False

        # --- Scenario 3: The response was empty or malformed ---
        else:
            # Terminate the turn.
            return None, None, False
