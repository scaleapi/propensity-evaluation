import re
from copy import deepcopy


def remove_duplicate_names_by_prefix(data: dict, prefix_pattern: str, key_to_update: str = None, **regex_flags) -> dict:
    updated_data = deepcopy(data)
    updated_data = {k.lower(): v for k, v in updated_data.items()}
    for item_name, item in data.items():
        match = re.match(rf'{prefix_pattern}(.*)', item_name, **regex_flags)
        if match:
            print(match)
            new_name = match.group(1).lower()
            updated_data.pop(item_name.lower())
            if new_name not in updated_data:
                if key_to_update:
                    item[key_to_update] = new_name
                updated_data[new_name] = item
    return updated_data


def apply_to_all_keys(data: dict, func, key_to_update: str = None, **kwargs) -> dict:
    updated_data = {}
    for item_name, item in data.items():
        modified_name = func(item_name, **kwargs)

        if key_to_update:
            item[key_to_update] = modified_name
        updated_data[modified_name] = item
    return updated_data
