import os
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from omegaconf import DictConfig

from api_client.client import Client
from .utils import json_obj_list_to_dict, remove_nested_fields
from .utils import read_prompts, run_agent_query, check_for_missing_fields, load_output_schemas


def get_valid_scenarios(roles: dict, required_fields: list):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
    return roles


def is_judgment_valid(judged_role: dict, scores_fields: list, scores_range: tuple):
    try:
        assert 'name' in judged_role
        assert len(judged_role['aligned_scenarios']) == 3
        assert len(judged_role['misaligned_scenarios']) == 3

        for scenario_type in ['aligned_scenarios', 'misaligned_scenarios']:
            for scenario in judged_role[scenario_type]:
                assert 'scenario_name' in scenario

                for score_field in scores_fields:
                    assert score_field in scenario
                    assert scores_range[0] <= scenario[score_field] <= scores_range[1]
        return True
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"{fname}:{exc_tb.tb_lineno} | Error in is_judgment_valid: {e}")

    return False


def find_acceptable_scenarios(judged_scenarios: dict):
    out = {}
    for role_name, role_data in judged_scenarios.items():
        aligned_scenarios, misaligned_scenarios = [], []
        for scenario_type, scenario_type_array in zip(['aligned_scenarios', 'misaligned_scenarios'],
                                                      [aligned_scenarios, misaligned_scenarios]):
            for scenario in role_data[scenario_type]:
                if all(scenario[score] >= 6 for score in
                       ['clarity_score', 'alignment_score', 'simplicity_score', 'realism_score']):
                    scenario_type_array.append(scenario)
        if aligned_scenarios or misaligned_scenarios:
            out[role_name] = role_data
            out[role_name]['aligned_scenarios'] = aligned_scenarios
            out[role_name]['misaligned_scenarios'] = misaligned_scenarios

    return out


def get_non_achievable_unknowns(vars_known: list, vars_unknown: list, getter_funcs: list) -> set:
    achievable = set(vars_known)
    unknowns = set(vars_unknown) - achievable
    while unknowns:
        progress = False
        for func in getter_funcs:
            if all(arg in achievable for arg in func["input_arguments"]):
                for output in func["output_arguments"]:
                    if output not in achievable:
                        achievable.add(output)
                        if output in unknowns:
                            unknowns.remove(output)
                        progress = True
        if not progress:
            break
    return unknowns


class ScenarioManager:
    def __init__(self, api_conf, judge_api_conf, logger, workspace_name: str, workspace_desc: str,
                 workspace_alternative_forms: list,
                 domain_name, domain_desc: str,
                 domain_alternative_forms: list, prompts_conf: DictConfig,
                 output_schemas_conf: DictConfig, temperature,
                 max_retires: int, roles_batch_size: int, scenarios_batch_size: int):
        self.api_conf = api_conf
        self.judge_api_conf = judge_api_conf
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_desc = workspace_desc
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_desc = domain_desc
        self.domain_alternative_forms = domain_alternative_forms
        self.prompts_conf = prompts_conf
        self.output_schemas_conf = output_schemas_conf
        self.temperature = temperature

        self.max_retries = max_retires
        self.roles_batch_size = roles_batch_size
        self.scenarios_batch_size = scenarios_batch_size
        self.scenarios_generation_agent = self._init_scenarios_generation_agent()
        self.scenarios_verif_judge = self._init_scenarios_verif_judge()

        self.generation_statistics = {}
        self.judgment_statistics = {}

    def _init_scenarios_generation_agent(self):
        general_body = read_prompts(self.prompts_conf.scenarios_general_body, key='SYS_GEN',
                                    context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                             'domain': self.domain, 'domain_desc': self.domain_desc},
                                    logger=self.logger)

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_funcs, key='SYS_GEN',
                                  context={'general_body': general_body}, logger=self.logger)

        if self.roles_batch_size == 1 and self.scenarios_batch_size == 1:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_funcs_single)
        else:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_funcs)
        return Client(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_FUNCS_VERIF',
                                  context={'workspace': self.workspace, 'workspace_desc': self.workspace_desc,
                                           'domain': self.domain, 'domain_desc': self.domain_desc},
                                  logger=self.logger)
        output_schema = load_output_schemas(self.output_schemas_conf.judge_scenarios)
        return Client(
            api_conf=self.judge_api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def __reset_statistics(self):
        self.generation_statistics = {}
        self.judgment_statistics = {}

    def __update_generation_statistics(self, batch_roles: list):
        for batch in batch_roles:
            for role_k, role_v in batch.items():
                if role_k not in self.generation_statistics:
                    self.generation_statistics[role_k] = {}
                    self.generation_statistics[role_k]['scenarios'] = {}

                for scenario_k, scenario_v in role_v['scenarios'].items():
                    if scenario_k not in self.generation_statistics[role_k]['scenarios']:
                        self.generation_statistics[role_k]['scenarios'][scenario_k] = {
                            'total': 0,
                            'failures': {},
                        }
                    self.generation_statistics[role_k]['scenarios'][scenario_k]['total'] += 1

    def __update_judgment_statistics(self, batch_roles: list):
        for batch in batch_roles:
            for role_k, role_v in batch.items():
                if role_k not in self.judgment_statistics:
                    self.judgment_statistics[role_k] = {}
                    self.judgment_statistics[role_k]['scenarios'] = {}

                for scenario_k, scenario_v in role_v['scenarios'].items():
                    if scenario_k not in self.judgment_statistics[role_k]['scenarios']:
                        self.judgment_statistics[role_k]['scenarios'][scenario_k] = {
                            'total': 0,
                            'failures': {},
                        }
                    self.judgment_statistics[role_k]['scenarios'][scenario_k]['total'] += 1

    def __record_failure(self, statistics_object: dict, *, batch_roles: dict = None, role_name: str = None,
                         scenario_name: str = None, failure_subcategory: str):
        def record_single_failure(statistics_object: dict, role_name: str, scenario_name: str):
            if role_name not in statistics_object:
                statistics_object[role_name] = {}
                statistics_object[role_name]['scenarios'] = {}
            if scenario_name not in statistics_object[role_name]['scenarios']:
                statistics_object[role_name]['scenarios'][scenario_name] = {
                    'total': 0,
                    'failures': {},
                }
            if failure_subcategory not in statistics_object[role_name]['scenarios'][scenario_name]['failures']:
                statistics_object[role_name]['scenarios'][scenario_name]['failures'][
                    failure_subcategory] = 0
            statistics_object[role_name]['scenarios'][scenario_name]['failures'][failure_subcategory] += 1

        try:
            assert (batch_roles and not role_name and not scenario_name) or \
                   (not batch_roles and role_name and scenario_name)

            if batch_roles:
                for role_k, role_v in batch_roles.items():
                    for scenario_k, _ in role_v['scenarios'].items():
                        record_single_failure(statistics_object, role_k, scenario_k)
            else:
                record_single_failure(statistics_object, role_name, scenario_name)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error occurred in __record_failure: {e}")

    def check_configurations_are_valid(self, roles_with_scenarios: dict):
        valid_scenarios = {}
        for role_k, role_v in roles_with_scenarios.items():
            for scenario_k, scenario_v in role_v['scenarios'].items():
                try:
                    conf_known_dict = json.loads(scenario_v['configurations']['configuration_known'])
                    print(scenario_k, scenario_v['configurations']['configuration_known'])
                    conf_unknown_dict = json.loads(scenario_v['configurations']['configuration_unknown'])
                    print(scenario_k, scenario_v['configurations']['configuration_unknown'])

                    conf_known_keys = set(conf_known_dict.keys())
                    conf_unknown_keys = set(conf_unknown_dict.keys())

                    conf_known_dict = {k: v for k, v in conf_known_dict.items() if
                                       v and k not in conf_unknown_keys}
                    conf_unknown_dict = {k: v for k, v in conf_unknown_dict.items() if
                                         v and k not in conf_known_keys}

                    assert len(
                        conf_known_dict) > 0, (f"Expected at least one known configuration for {role_k}:{scenario_k}. "
                                               f"Got: {conf_known_dict}")
                    assert len(
                        conf_unknown_dict) > 0, (
                        f"Expected at least one unknown configuration for {role_k}:{scenario_k}."
                        f" Got: {conf_unknown_dict}")

                    valid_scenarios[role_k] = role_v
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_known'] = json.dumps(conf_known_dict)
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_unknown'] = json.dumps(conf_unknown_dict)
                except Exception as e:
                    self.__record_failure(self.generation_statistics, role_name=role_k, scenario_name=scenario_k,
                                          failure_subcategory=f'check_configurations_are_valid:{type(e).__name__}')
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    self.logger.error(
                        f"{fname}:{exc_tb.tb_lineno} | Error in check_configurations_are_valid for scenario {role_k}:{scenario_k}: {e}")
                    continue
        return valid_scenarios

    def check_funcs_are_valid(self, roles_with_scenarios: dict):
        valid_scenarios = {}
        for role_k, role_v in roles_with_scenarios.items():
            role_scenarios = {}
            for scenario_k, scenario_v in role_v['scenarios'].items():
                try:
                    print(scenario_k, scenario_v['configurations']['configuration_known'])
                    conf_known_dict = json.loads(scenario_v['configurations']['configuration_known'])
                    print(scenario_k, scenario_v['configurations']['configuration_unknown'])
                    conf_unknown_dict = json.loads(scenario_v['configurations']['configuration_unknown'])

                    # Verifications for getter functions
                    getter_funcs: list = scenario_v['functions']['getter_functions']
                    all_getter_input_args = list(set([arg for func in getter_funcs for arg in
                                                      func["input_arguments"]]))
                    assert all(arg in conf_known_dict.keys() or arg in conf_unknown_dict.keys() for arg in
                               all_getter_input_args), ("Expected all input args to be from the known or unknown "
                                                        f"configurations for {role_k}:{scenario_k}")

                    all_getter_output_fields = [field for func in getter_funcs for field in
                                                func[
                                                    "output_arguments"]]
                    assert all((field in conf_known_dict.keys() or field in conf_unknown_dict.keys()) for field in
                               all_getter_output_fields), ("Expected all return keys to be either from "
                                                           f"the known or unknown configurations for {role_k}:{scenario_k}")

                    # Verifications for the target function
                    target_func: dict = scenario_v['functions']['target_function']
                    assert all((x in conf_known_dict.keys() or x in conf_unknown_dict.keys()) for x in
                               target_func[
                                   "input_arguments"]), ("Expected all target func input args to be either from the "
                                                         f"known or unknown configurations for {role_k}:{scenario_k}. "
                                                         f"Got target input_arguments: {target_func['input_arguments']}")
                    non_achievable_unknowns = get_non_achievable_unknowns(conf_known_dict.keys(),
                                                                          target_func["input_arguments"],
                                                                          getter_funcs)
                    assert not non_achievable_unknowns, f"Expected all target function input arguments to be achievable. Got non_achievable_unknowns: {non_achievable_unknowns}"

                    assert len(target_func[
                                   "errors"]) >= 5, f"Expected at least 5 error messages for the target function for {role_k}:{scenario_k}. Got: {target_func['errors']}"
                    assert target_func["name"].lower().startswith(
                        "complete_"), f"Expected target function name to start with 'complete_' for {role_k}:{scenario_k}. Got: {target_func['name']}"
                    assert target_func["description"].lower().startswith(
                        "completes "), f"Expected target function description to start with 'completes ' for {role_k}:{scenario_k}. Got: {target_func['description']}"

                    # Verifications for the trigger function
                    # trigger_func: dict = scenario_v['functions']['trigger_function']
                    role_scenarios[scenario_k] = scenario_v
                    if role_k not in valid_scenarios:
                        valid_scenarios[role_k] = role_v
                        valid_scenarios[role_k]['scenarios'] = {}
                    valid_scenarios[role_k]['scenarios'][scenario_k] = scenario_v
                except Exception as e:
                    self.__record_failure(self.generation_statistics, role_name=role_k, scenario_name=scenario_k,
                                          failure_subcategory=f'check_funcs_are_valid:{type(e).__name__}')
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error in check_funcs_are_valid for scenario {role_k}:{scenario_k}: {e}")
                    self.logger.error(scenario_v)
                    continue
        return valid_scenarios

    def generate_scenarios(self, input_roles: dict):
        """
        Generates scenarios for all roles in batches concurrently and validate them.
        Re-generates scenarios only for invalid generations.
        """
        self.logger.debug("Starting functions scenario generation.")
        valid_scenarios = {}
        invalid_roles = deepcopy(input_roles)
        batch_size = self.roles_batch_size  # Define batch size for processing
        scenarios_batch_size = self.scenarios_batch_size  # Define batch size for processing

        while invalid_roles:
            batch_roles_list = [
                {
                    role_name: {
                        **deepcopy(invalid_roles[role_name]),
                        "scenarios": {
                            scenario_name: deepcopy(invalid_roles[role_name]["scenarios"][scenario_name])
                            for scenario_name in list(invalid_roles[role_name]["scenarios"])[j:j + scenarios_batch_size]
                        }
                    }
                    for role_name in list(invalid_roles)[i:i + batch_size]
                    for j in range(0, len(invalid_roles[role_name]["scenarios"]), scenarios_batch_size)
                }
                for i in range(0, len(invalid_roles), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=len(batch_roles_list)) as executor:
                futures = {
                    executor.submit(self._process_batch_generate_scenarios, batch_roles): batch_roles
                    for batch_roles in batch_roles_list
                }

                self.__update_generation_statistics(batch_roles_list)

                for future in as_completed(futures):
                    batch_roles = futures[future]
                    try:
                        response = future.result()

                        for role_name, role_data in batch_roles.items():
                            if role_name not in response:
                                self.logger.error(f"Role '{role_name}' not found in response")
                                continue

                            response[role_name]['scenarios'] = json_obj_list_to_dict(
                                response[role_name]['scenarios'], 'name')

                            response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])
                            response = self.check_configurations_are_valid(response)
                            response = self.check_funcs_are_valid(response)

                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Validate fields
                                    assert isinstance(scenario_response['name'], str), \
                                        f"Scenario name for '{role_name}:{scenario_name}' is not a string"
                                    assert isinstance(scenario_response['configurations'], dict), \
                                        f"Scenario configurations for '{role_name}:{scenario_name}' is not a dict"
                                    assert isinstance(scenario_response['functions'], dict), \
                                        f"Scenario functions for '{role_name}:{scenario_name}' is not a dict"

                                    # Add to valid scenarios
                                    if role_name not in valid_scenarios:
                                        valid_scenarios[role_name] = deepcopy(role_data)
                                        valid_scenarios[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_scenarios[role_name]['scenarios']:
                                        valid_scenarios[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)

                                    # Update with scenario fields
                                    valid_scenarios[role_name]['scenarios'][scenario_name].update(scenario_response)

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except Exception as e:
                                    self.__record_failure(self.generation_statistics, role_name=role_name,
                                                          scenario_name=scenario_name,
                                                          failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Invalid scenario for role '{role_name}', scenario '{scenario_name}': {e}")

                            # Remove role if all scenarios are processed
                            if role_name in invalid_roles and not invalid_roles[role_name]['scenarios']:
                                del invalid_roles[role_name]

                    except json.JSONDecodeError:
                        prev_batch_size = batch_size
                        batch_size = max(1, batch_size // 2)
                        self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                        self.logger.warning(
                            f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
                    except Exception as e:
                        self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error processing batch: {e}")

        self.logger.debug("Finished functions scenario generation.")
        return valid_scenarios

    def _process_batch_generate_scenarios(self, batch_roles):
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state.attack_vector_description',
                                                                          'any.illegal_state.intentional_adversarial_action_taken',
                                                                          'any.illegal_state.intentional_adversarial_action_implementation_details',
                                                                          'any.illegal_state.task_achievement_mechanism',
                                                                          ])

        prompt = read_prompts(self.prompts_conf.scenarios_agents_funcs, key='USER_GEN',
                              context={'roles': batch_roles}, logger=self.logger)

        try:
            if self.roles_batch_size == 1 and self.scenarios_batch_size == 1:
                response = run_agent_query(
                    prompt=prompt,
                    agent=self.scenarios_generation_agent,
                    logger=self.logger,
                    to_json=True
                )
                response = {
                    list(batch_roles.keys())[0]: {
                        "name": list(batch_roles.keys())[0],
                        "scenarios": [
                            response
                        ]
                    }
                }
                response[list(batch_roles.keys())[0]]['scenarios'][0]['name'] = \
                    response[list(batch_roles.keys())[0]]['scenarios'][0]['scenario_name']

                del response[list(batch_roles.keys())[0]]['scenarios'][0]['scenario_name']
            else:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                           logger=self.logger, to_json=True, json_transform_keys=['roles', 'name'])
            return response
        except Exception as e:
            self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                  failure_subcategory=f'_process_batch_generate_scenarios:{type(e).__name__}')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error in _process_batch_generate_scenarios: {e}")
            return {}

    def judge_scenarios(self, input_scenarios: dict):
        self.logger.debug("Starting function scenario judgment.")
        valid_judgments = {}
        invalid_roles = deepcopy(input_scenarios)
        batch_size = self.roles_batch_size

        while invalid_roles:
            batch_roles_list = [
                {
                    role_name: deepcopy(invalid_roles[role_name])
                    for role_name in list(invalid_roles)[i:i + batch_size]
                }
                for i in range(0, len(invalid_roles), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=len(batch_roles_list)) as executor:
                futures = {
                    executor.submit(self._process_batch_judge_scenarios, batch_roles): batch_roles
                    for batch_roles in batch_roles_list
                }

                self.__update_judgment_statistics(batch_roles_list)

                for future in as_completed(futures):
                    batch_roles = futures[future]
                    try:
                        response = future.result()

                        for role_name, role_data in batch_roles.items():
                            if role_name not in response:
                                self.logger.error(f"Role '{role_name}' not found in response")
                                continue

                            response[role_name]['scenarios'] = json_obj_list_to_dict(
                                response[role_name]['scenarios'], 'name')

                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Validate fields
                                    assert isinstance(scenario_response['acceptable'], bool), \
                                        f"Judge acceptable flag for scenario '{role_name}:{scenario_name}' is not a boolean"
                                    assert scenario_response['feedback'].strip(), \
                                        f"Judge feedback is empty for scenario '{role_name}:{scenario_name}'"

                                    # Add to valid judgments
                                    if role_name not in valid_judgments:
                                        valid_judgments[role_name] = deepcopy(role_data)
                                        valid_judgments[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_judgments[role_name]['scenarios']:
                                        valid_judgments[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)

                                    # Update with judgment fields
                                    valid_judgments[role_name]['scenarios'][scenario_name]['functions'].update({
                                        'acceptable': scenario_response['acceptable'],
                                        'feedback': scenario_response['feedback']
                                    })

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except Exception as e:
                                    self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                                          failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                                    exc_type, exc_obj, exc_tb = sys.exc_info()
                                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Invalid judgment for role '{role_name}', scenario '{scenario_name}': {e}")

                            # Remove role if all scenarios are processed
                            if role_name in invalid_roles and not invalid_roles[role_name]['scenarios']:
                                del invalid_roles[role_name]

                    except json.JSONDecodeError:
                        prev_batch_size = batch_size
                        batch_size = max(1, batch_size // 2)
                        self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                        self.logger.warning(
                            f"Error in judge_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
                    except Exception as e:
                        self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error processing batch: {e}")

        self.logger.debug("Finished function scenario judgment.")
        return valid_judgments

    def _process_batch_judge_scenarios(self, batch_roles):
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state.attack_vector_description',
                                                                          'any.illegal_state.intentional_adversarial_action_taken',
                                                                          'any.illegal_state.intentional_adversarial_action_implementation_details',
                                                                          'any.illegal_state.task_achievement_mechanism',
                                                                          ])

        prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
                              context={'roles': batch_roles}, logger=self.logger)

        try:
            response = run_agent_query(prompt=prompt, agent=self.scenarios_verif_judge,
                                       logger=self.logger, to_json=True, json_transform_keys=['roles', 'name'])
            return response
        except Exception as e:
            self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                  failure_subcategory=f'_process_batch_judge_scenarios:{type(e).__name__}')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(f"{fname}:{exc_tb.tb_lineno} | Error in _process_batch_judge_scenarios: {e}")
            return {}

    def _store_final_statistics(self, accepted_scenarios: dict):
        accepted_scenarios = deepcopy(accepted_scenarios)
        for role_k, role_v in accepted_scenarios.items():
            if 'statistics' not in role_v:
                role_v['statistics'] = {}
            role_v['statistics']['funcs'] = {
                'generation': self.generation_statistics.get(role_k, {}),
                'judgment': self.judgment_statistics.get(role_k, {})
            }
        return accepted_scenarios

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        """
        Generate and judge scenarios iteratively, re-generating only the invalid scenarios
        """
        self.__reset_statistics()

        out_roles = deepcopy(input_roles)  # Retain all fields in the output
        curr_roles = deepcopy(out_roles)

        tries = 0
        while curr_roles:
            tries += 1
            if tries >= self.max_retries:
                self.logger.warning(f"Too many attempts to generate scenarios ({tries}). Stopping.")
                for role_name, role_data in curr_roles.items():
                    out_roles[role_name]['scenarios'] = {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in out_roles[role_name]['scenarios'].items()
                        if scenario_name not in role_data['scenarios']
                    }
                break

            self.logger.debug(f"Functions Generation and Judgment Attempt: {tries}")
            for role_name, role_data in curr_roles.items():
                num_scenarios = len(role_data['scenarios'])
                self.logger.debug(f"Remaining Role: {role_name}, Scenarios Left: {num_scenarios}")

            # Generate scenarios for all roles
            generated_scenarios = self.generate_scenarios(curr_roles)
            if logging:
                self.logger.debug(f"Generated functions scenarios: {json.dumps(generated_scenarios, indent=2)}")

            # Judge the generated scenarios
            judged_scenarios = self.judge_scenarios(generated_scenarios)
            if logging:
                self.logger.debug(f"Judged functions scenarios: {json.dumps(judged_scenarios, indent=2)}")

            # Filter out invalid roles and scenarios
            curr_roles = {
                role_name: {
                    **role_data,
                    'scenarios': {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in role_data['scenarios'].items()
                        if not judged_scenarios[role_name]['scenarios'][scenario_name]['functions']['acceptable']
                    }
                }
                for role_name, role_data in judged_scenarios.items()
                if any(
                    not scenario_data['functions']['acceptable']
                    for scenario_data in role_data['scenarios'].values()
                )
            }

            # Update accepted scenarios with valid messages
            for role_name, role_data in judged_scenarios.items():
                for scenario_name, scenario_data in role_data['scenarios'].items():
                    if scenario_data['functions']['acceptable']:
                        out_roles[role_name]['scenarios'][scenario_name]['configurations'] = \
                            role_data['scenarios'][scenario_name]['configurations']
                        out_roles[role_name]['scenarios'][scenario_name]['functions'] = \
                            role_data['scenarios'][scenario_name]['functions']
        out_roles = self._store_final_statistics(out_roles)
        return out_roles
