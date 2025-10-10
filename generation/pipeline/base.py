import abc
import datetime
import os.path
import time
import json
import traceback

KEYS_ORDERS = [
    'name', 'description', 'scenarios',

    'states', 'initial_state',
    'target_state', 'illegal_state', 'attack_vector_description', 'attack_subject', 'attack_objects',
    'achievements', 'common_outcome_target_trigger', 'intentional_adversarial_action_taken',
    'intentional_adversarial_action_implementation_details', 'task_achievement_mechanism',

    'policy', 'role_description', 'duties_description', 'prohibitions_description', 'trigger_rationale',
    'immediate_outcome_acknowledgment', 'negative_consequences', 'trigger_awareness', 'trigger_awareness_fpp',

    'configurations', 'functions', 'getter_functions', 'target_function', 'trigger_function', 'neutral_name',
    'neutral_description', 'harmful_name', 'harmful_description', 'input_arguments', 'output_arguments', 'errors',

    'messages', 'task_message', 'neutral_sys_messages', 'msg_level_index', 'from', 'subject', 'body', 'pressure_level',

    'sys_messages',

    'prefix_connector', 'prefix_connector_sentence', 'text',

    'statistics', 'generation', 'judgment',
]


class BasePipeline(abc.ABC):
    def __init__(self, cfg, logs_fname: str = None):
        self.cfg = cfg

        self.logs_dir = cfg.logs_dir
        logs_fname = logs_fname if logs_fname is not None else str(datetime.datetime.now()).replace(' ',
                                                                                                    '-') + '.log'
        self.logs_file_path = os.path.join(self.logs_dir, logs_fname)
        os.makedirs(self.logs_dir, exist_ok=True)
        if os.path.exists(self.logs_file_path):
            os.remove(self.logs_file_path)

    def _log_to_file(self, role: str, data, cost):
        data = {'role': role, 'data': data, 'cost': cost, 'timestamp': time.time()}
        if os.path.exists(self.logs_file_path):
            with open(self.logs_file_path, 'r') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = []
                    print(traceback.format_exc())
        else:
            existing_data = []

        existing_data.append(data)

        with open(self.logs_file_path, 'w') as f:
            json.dump(existing_data, f, indent=4)

    def _log_response(self, role: str, response_content, cost):
        self._log_to_file(role, response_content, cost)
