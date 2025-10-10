import re


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
