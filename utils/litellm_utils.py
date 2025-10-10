import json


def get_response_content(response, to_json=False):
    out = response.choices[0].message.content
    if to_json:
        out = json.loads(out)
    return out


def get_response_cost(response):
    return response._hidden_params["response_cost"]
