import json
import os
import traceback

from api_client.client import Client
from utils.litellm_utils import get_response_content


def save_to_disk(data: dict, file_path: str):
    # make the necessary dirs
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def fill_prompt(prompt: str, fields: list, context: dict):
    for field in fields:
        if field not in context:
            raise ValueError(f"Field {field} not found in context")

        prompt = prompt.replace('{' + f'{field}' + '}', str(context[field]))
    return prompt


def read_prompts(file_path: str, key: str, delimiter: str = r'\[{}\]', context: dict = None, logger=None):
    if context is None:
        context = dict()
    with open(file_path, 'r') as file:
        content = file.read()

    key = key.strip().lower()

    # Match everything from the key until the next delimiter or end of file
    assert '(' not in key and ')' not in key, "Key should not contain parentheses"
    pattern = rf"{delimiter.format(key)}\s*?fields\s*:(.*?)\n(.*?)(?={delimiter.format(r'.*?')}|\Z)"
    pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    matches = pattern.findall(content)

    if not matches:
        raise ValueError(f"Key {key} not found in prompts file {file_path}")

    assert len(matches) == 1, f"Key {key} not found in prompts file {file_path}"
    assert len(matches[0]) == 2, f'Pattern match failed: {matches[0]}'

    fields_line, text = matches[0][0], matches[0][1]
    fields = [x.strip().lower() for x in fields_line.split(',')]
    fields = [x for x in fields if x]  # Remove empty strings
    prompt = text.strip()

    # if logger is not None:
    #     logger.debug(
    #         f'For file: {file_path}, key: {key}, \nLoaded fields: {fields}, \nProvided context: {context},\nLoaded prompt: {prompt}\n\n')
    filled_prompt = fill_prompt(prompt, fields, context)

    return filled_prompt


def extract_required_fields(json_schema: dict) -> list:
    required_fields = []

    def traverse(schema):
        if 'required' in schema:
            required_fields.extend(schema['required'])
        if 'properties' in schema:
            for key, value in schema['properties'].items():
                traverse(value)
        if 'items' in schema:
            traverse(schema['items'])

    traverse(json_schema)
    return required_fields


def load_output_schemas(path: str):
    with open(path, 'r') as f:
        out = json.load(f)

    return out


def json_obj_list_to_dict(json_list: list, transformation_key: str):
    out = dict()
    for obj in json_list:
        key = obj[transformation_key]
        out[key] = obj
    return out


def load_from_disk(file_path: str) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)

def validate_response(response, schema):
    """
    Recursively validate that the response contains all required fields from the schema.
    """
    if isinstance(schema, dict):
        # Check required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in response:
                raise ValueError(f"Missing required field: {field}")
        # Recursively validate properties
        for key, value in schema.get("properties", {}).items():
            if key in response:
                validate_response(response[key], value)
    elif isinstance(schema, list) and "items" in schema:
        # Validate list items
        for item in response:
            validate_response(item, schema["items"])



def run_agent_query(prompt, agent: Client, logger, n_retries=5, to_json=False, json_transform_keys: list = None):
    num_retries = n_retries
    # while num_retries >= 0:
    #     try:
    result = agent(prompt)

    try:
        if result.choices[0].finish_reason == 'length':
            logger.warning(f"Response length exceeded for prompt.")
    except Exception as e:
        logger.error(f"Could not extract finish_reason: {e}")
        logger.error(traceback.format_exc())

    response = get_response_content(result, to_json=to_json)
    if to_json:
        validate_response(response, agent.output_schema['json_schema']['schema'])
    if to_json and json_transform_keys:
        assert len(json_transform_keys) == 2
        response = response[json_transform_keys[0]]
        response = json_obj_list_to_dict(response, json_transform_keys[1])
    return response


def check_for_missing_fields(input_dict: dict, required_fields: list) -> dict:
    out = {}
    for obj in input_dict.values():
        missing_fields = []
        for required_field in required_fields:
            if required_field not in obj:
                missing_fields.append(required_field)
        if missing_fields:
            out[obj['name']] = {'name': obj['name'], 'missing_fields': missing_fields}
    return out


def capitalize_first_letter(text):
    return ' '.join(word[0].upper() + word[1:] for word in text.split())

def capitalize_paragraph(paragraph_in: str):
    """
    Capitalizes the first letter of each sentence in a paragraph.
    Works with ".", "!", and "?" as sentence delimiters. Maintains the case for the rest of the sentences.
    """
    sentences = re.split(r'([.!?])', paragraph_in)
    capitalized_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if sentence:
            capitalized_sentences.append(sentence[0].upper() + sentence[1:] + sentences[i + 1])
    return ' '.join(capitalized_sentences).strip()


def normalize_string(text, lower=True, remove_special_chars_punctuation=False):
    # Convert to lowercase
    text = text.lower()
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    text = re.sub(r'[-_]', ' ', text)

    # Optionally, remove special characters or punctuation
    if remove_special_chars_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    return text


def order_dict_keys(data: dict, order: list) -> dict:
    """
    Order the keys in the given dictionary and each sub-dict of it.
    Keys are ordered first by the specified order and then alphabetically for the remaining fields.

    Args:
        data (dict): The dictionary to be ordered.
        order (list): The list of keys specifying the desired order.

    Returns:
        dict: The dictionary with ordered keys.
    """

    def order_keys(d):
        ordered = {k: d[k] for k in order if k in d}
        remaining = {k: d[k] for k in sorted(d) if k not in order}
        return {**ordered, **remaining}

    def recursive_order(d):
        if isinstance(d, dict):
            return {k: recursive_order(v) for k, v in order_keys(d).items()}
        elif isinstance(d, list):
            return [recursive_order(i) for i in d]
        else:
            return d

    return recursive_order(data)


import copy


def merge_keys_in_scenarios(dict1, dict2, key_mappings):
    """
    Merges values from keys in dict1 to keys in dict2 based on the provided key mappings.

    Args:
        dict1 (dict): The source dictionary.
        dict2 (dict): The target dictionary.
        key_mappings (list): List of tuples where each tuple contains the source key and the target key in dot notation.

    Returns:
        dict: The updated target dictionary.
    """

    def get_nested_value(d, keys):
        for key in keys:
            d = d.get(key, {})
        return d

    def set_nested_value(d, keys, value):
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value

    dict2 = copy.deepcopy(dict2)

    for domain in dict2:
        for workspace in dict2[domain]:
            for role in dict2[domain][workspace]:
                for scenario in dict2[domain][workspace][role]['scenarios']:
                    for src_key, tgt_key in key_mappings:
                        src_keys = src_key.split('.')
                        tgt_keys = tgt_key.split('.')
                        value = get_nested_value(dict1[domain][workspace][role]['scenarios'][scenario], src_keys)
                        if value:
                            set_nested_value(dict2[domain][workspace][role]['scenarios'][scenario], tgt_keys, value)

    return dict2


def remove_nested_fields(roles: dict, fields_to_remove: list):
    """
    Remove specified fields in the format 'x.y.z' from every scenario of every role.
    Supports wildcard 'any' to match all keys at a level.

    Args:
        roles (dict): The roles dictionary containing scenarios.
        fields_to_remove (list): List of fields to remove in 'x.y.z' format.

    Returns:
        dict: The updated roles dictionary with specified fields removed.
    """

    def delete_nested_field(data, keys):
        """Recursively delete a nested field given a list of keys."""
        if not keys:
            return

        if keys[0] == 'any':
            # If 'any' is encountered, apply the deletion to all keys at this level
            for key in list(data.keys() if isinstance(data, dict) else range(len(data))):
                delete_nested_field(data[key], keys[1:])
        else:
            if not isinstance(data, dict) or keys[0] not in data:
                return

            if len(keys) == 1:
                del data[keys[0]]
            else:
                delete_nested_field(data[keys[0]], keys[1:])

    roles = deepcopy(roles)
    for role_name, role_data in roles.items():
        for scenario_name, scenario_data in role_data.get('scenarios', {}).items():
            for field in fields_to_remove:
                keys = field.split('.')
                delete_nested_field(scenario_data, keys)

    return roles


def filter_fields_for_roles(roles: dict, fields_to_include: list) -> dict:
    """
    Include only specific fields in the scenarios of the roles dictionary for processing,
    while keeping all fields in the roles themselves.

    Args:
        roles (dict): The roles dictionary containing scenarios.
        fields_to_include (list): List of fields to include in 'x.y.z' format, supporting 'any' wildcard.

    Returns:
        dict: A new roles dictionary with all role-level fields retained and only the specified fields in scenarios.
    """

    def get_nested_field(data, keys):
        """Recursively get a nested field given a list of keys."""
        if not keys:
            return data
        if keys[0] == 'any':
            if isinstance(data, dict):
                return {k: get_nested_field(v, keys[1:]) for k, v in data.items()}
            elif isinstance(data, list):
                return [get_nested_field(v, keys[1:]) for v in data]
            else:
                return None
        if keys[0] in data:
            return get_nested_field(data[keys[0]], keys[1:])
        return None

    def set_nested_field(data, keys, value):
        """Recursively set a nested field given a list of keys."""
        if not keys:
            return
        if keys[0] == 'any':
            if isinstance(data, dict):
                for k in data.keys():
                    set_nested_field(data[k], keys[1:], value)
            elif isinstance(data, list):
                for i in range(len(data)):
                    set_nested_field(data[i], keys[1:], value)
        else:
            for key in keys[:-1]:
                data = data.setdefault(key, {})
            data[keys[-1]] = value

    filtered_roles = {}
    for role_name, role_data in roles.items():
        # Retain all role-level fields
        filtered_roles[role_name] = deepcopy(role_data)
        filtered_roles[role_name]['scenarios'] = {}

        for scenario_name, scenario_data in role_data.get('scenarios', {}).items():
            filtered_scenario = {}
            for field in fields_to_include:
                keys = field.split('.')
                value = get_nested_field(scenario_data, keys)
                if value is not None:
                    set_nested_field(filtered_scenario, keys, value)
            filtered_roles[role_name]['scenarios'][scenario_name] = filtered_scenario

    return filtered_roles


def rename_nested_fields(roles: dict, fields_to_rename: list, new_field_name: str) -> dict:
    """
    Rename specified fields in the format 'x.y.z' to a new field name in every scenario of every role.
    Supports wildcard 'any' to match all keys at a level.

    Args:
        roles (dict): The roles dictionary containing scenarios.
        fields_to_rename (list): List of fields to rename in 'x.y.z' format.
        new_field_name (str): The new name for the specified fields.

    Returns:
        dict: The updated roles dictionary with specified fields renamed.
    """

    def rename_nested_field(data, keys, new_name):
        """Recursively rename a nested field given a list of keys."""
        if not keys:
            return

        if keys[0] == 'any':
            # If 'any' is encountered, apply the renaming to all keys at this level
            for key in list(data.keys() if isinstance(data, dict) else range(len(data))):
                rename_nested_field(data[key], keys[1:], new_name)
        else:
            if keys[0] in data:
                if len(keys) == 1:
                    data[new_name] = data.pop(keys[0])
                else:
                    rename_nested_field(data[keys[0]], keys[1:], new_name)

    roles = deepcopy(roles)
    for role_name, role_data in roles.items():
        for scenario_name, scenario_data in role_data.get('scenarios', {}).items():
            for field in fields_to_rename:
                keys = field.split('.')
                rename_nested_field(scenario_data, keys, new_field_name)

    return roles


import re
from copy import deepcopy


def append_function_to_matching_fields(data, pattern, new_field_name, input_function):
    """
    Append a new field with a specific name at the same level for fields matching a pattern.

    Args:
        data (dict or list): The input data (nested dictionary or list).
        pattern (str): The regex pattern to match field names.
        new_field_name (str): The name of the new field to append.
        input_function (callable): The function to generate the value for the new field.

    Returns:
        dict or list: The updated data with the new fields appended.
    """
    if isinstance(data, dict):
        updated_data = {}
        for key, value in data.items():
            # Check if the key matches the pattern
            if re.match(pattern, key):
                # Add the original field
                updated_data[key] = value
                # Add the new field with the input function applied
                updated_data[new_field_name] = input_function(value)
            else:
                # Recursively process nested structures
                updated_data[key] = append_function_to_matching_fields(value, pattern, new_field_name, input_function)
        return updated_data
    elif isinstance(data, list):
        # Process each item in the list
        return [append_function_to_matching_fields(item, pattern, new_field_name, input_function) for item in data]
    else:
        # Return the data as-is if it's not a dict or list
        return data
