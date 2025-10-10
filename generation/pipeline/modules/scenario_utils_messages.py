import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed

from omegaconf import DictConfig
from api_client.client import Client
from .utils import read_prompts, save_to_disk, run_agent_query, check_for_missing_fields, load_output_schemas
from .utils import json_obj_list_to_dict, remove_nested_fields


def get_valid_scenarios(roles: dict, required_fields: list):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
    return roles


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

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='SYS_GEN',
                                  context={'general_body': general_body}, logger=self.logger)

        if self.roles_batch_size == 1:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_messages_single_neut)
        else:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_messages)

        return Client(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_NEUTRAL_MESSAGES_VERIF',
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
            self.logger.error('Error occurred in __record_failure:', e)

    def generate_scenarios(self, input_roles: dict):
        """
        Generates scenarios for all roles in batches concurrently and validate them.
        Re-generates scenarios only for invalid generations.
        """
        self.logger.debug("Starting messages (task/neutral) scenario generation.")
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

                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Validate fields
                                    assert isinstance(scenario_response['name'], str), \
                                        f"Scenario name for '{role_name}:{scenario_name}' is not a string"
                                    assert scenario_response['task_message'], \
                                        f"Scenario task message for '{role_name}:{scenario_name}' is empty"
                                    assert isinstance(scenario_response['neutral_sys_messages'], list), \
                                        f"Scenario neutral sys messages for '{role_name}:{scenario_name}' is not a list"

                                    # Add to valid scenarios
                                    if role_name not in valid_scenarios:
                                        valid_scenarios[role_name] = deepcopy(role_data)
                                        valid_scenarios[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_scenarios[role_name]['scenarios']:
                                        valid_scenarios[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)
                                        valid_scenarios[role_name]['scenarios'][scenario_name]['messages'] = {}

                                    # Update with scenario fields
                                    valid_scenarios[role_name]['scenarios'][scenario_name]['messages'].update({
                                        'task_message': scenario_response['task_message'],
                                        'neutral_sys_messages': scenario_response['neutral_sys_messages']
                                    })

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except Exception as e:
                                    self.__record_failure(self.generation_statistics, role_name=role_name,
                                                          scenario_name=scenario_name,
                                                          failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                                    self.logger.error(
                                        f"Invalid scenario for role '{role_name}', scenario '{scenario_name}': {e}")

                            # Remove role if all scenarios are processed
                            if role_name in invalid_roles and not invalid_roles[role_name]['scenarios']:
                                del invalid_roles[role_name]

                    except json.JSONDecodeError:
                        self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                        prev_batch_size = batch_size
                        batch_size = max(1, batch_size // 2)
                        self.logger.warning(
                            f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")

                    except Exception as e:
                        self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                        self.logger.error(f"Error processing batch: {e}")

        self.logger.debug("Finished messages (task/neutral) scenario generation.")
        return valid_scenarios

    def _process_batch_generate_scenarios(self, batch_roles):
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state',
                                                                          'any.trigger_function'
                                                                          'any.prohibitions_description'
                                                                          'any.trigger_rationale'
                                                                          'any.immediate_outcome_acknowledgment'
                                                                          'any.negative_consequences'
                                                                          'any.trigger_awareness'
                                                                          'any.trigger_awareness_fpp'
                                                                          ])

        prompt = read_prompts(self.prompts_conf.scenarios_agents_messages, key='USER_GEN',
                              context={'roles': batch_roles}, logger=self.logger)

        try:
            if self.roles_batch_size == 1:
                response = run_agent_query(
                    prompt=prompt,
                    agent=self.scenarios_generation_agent,
                    logger=self.logger,
                    to_json=True
                )
                response = {
                    list(batch_roles.keys())[0]: {
                        "name": list(batch_roles.keys())[0],
                        **response
                    }
                }
            else:
                response = run_agent_query(prompt=prompt, agent=self.scenarios_generation_agent,
                                           logger=self.logger, to_json=True, json_transform_keys=['roles', 'name'])
            return response
        except Exception as e:
            self.__record_failure(self.generation_statistics, batch_roles=batch_roles,
                                  failure_subcategory=f'_process_batch_generate_scenarios:{type(e).__name__}')
            self.logger.error(f"Error in _process_batch_generate_scenarios: {e}")
            return {}

    def judge_scenarios(self, input_scenarios: dict):
        self.logger.debug("Starting messages (task/neutral) scenario judgment.")
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
                                    valid_judgments[role_name]['scenarios'][scenario_name]['messages'].update({
                                        'acceptable': scenario_response['acceptable'],
                                        'feedback': scenario_response['feedback']
                                    })

                                    # Remove from invalid roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except Exception as e:
                                    self.__record_failure(self.judgment_statistics, role_name=role_name,
                                                          scenario_name=scenario_name,
                                                          failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                                    self.logger.error(
                                        f"Invalid judgment for role '{role_name}', scenario '{scenario_name}': {e}")

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
                        self.logger.error(f"Error processing batch: {e}")

        self.logger.debug("Finished messages (task/neutral) scenario judgment.")
        return valid_judgments

    def _process_batch_judge_scenarios(self, batch_roles):

        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'attack_vector',
                                                                          'any.attack_vector',
                                                                          'any.illegal_state',
                                                                          'any.trigger_function'
                                                                          'any.prohibitions_description'
                                                                          'any.trigger_rationale'
                                                                          'any.immediate_outcome_acknowledgment'
                                                                          'any.negative_consequences'
                                                                          'any.trigger_awareness'
                                                                          'any.trigger_awareness_fpp'
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
            self.logger.error(f"Error in _process_batch_judge_scenarios: {e}")
            return {}

    def _store_final_statistics(self, accepted_scenarios: dict):
        accepted_scenarios = deepcopy(accepted_scenarios)
        for role_k, role_v in accepted_scenarios.items():
            if 'statistics' not in role_v:
                role_v['statistics'] = {}
            role_v['statistics']['messages'] = {
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
            if tries > 50:
                self.logger.warning(f"Too many attempts to generate scenarios ({tries}). Stopping.")
                for role_name, role_data in curr_roles.items():
                    out_roles[role_name]['scenarios'] = {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in out_roles[role_name]['scenarios'].items()
                        if scenario_name not in role_data['scenarios']
                    }
                break

            for role_name, role_data in curr_roles.items():
                num_scenarios = len(role_data['scenarios'])
                self.logger.debug(f"Remaining Role: {role_name}, Scenarios Left: {num_scenarios}")

            # Generate scenarios for all roles
            generated_scenarios = self.generate_scenarios(curr_roles)
            if logging:
                self.logger.debug(f"Generated messages scenarios: {json.dumps(generated_scenarios, indent=2)}")

            # Judge the generated scenarios
            judged_scenarios = self.judge_scenarios(generated_scenarios)
            if logging:
                self.logger.debug(f"Judged messages scenarios: {json.dumps(judged_scenarios, indent=2)}")

            # Filter out invalid roles and scenarios
            curr_roles = {
                role_name: {
                    **role_data,
                    'scenarios': {
                        scenario_name: scenario_data
                        for scenario_name, scenario_data in role_data['scenarios'].items()
                        if not judged_scenarios[role_name]['scenarios'][scenario_name]['messages']['acceptable']
                    }
                }
                for role_name, role_data in judged_scenarios.items()
                if any(
                    not scenario_data['messages']['acceptable']
                    for scenario_data in role_data['scenarios'].values()
                )
            }

            # Update accepted scenarios with valid messages
            for role_name, role_data in judged_scenarios.items():
                for scenario_name, scenario_data in role_data['scenarios'].items():
                    if scenario_data['messages']['acceptable']:
                        out_roles[role_name]['scenarios'][scenario_name]['configurations'] = \
                            role_data['scenarios'][scenario_name]['configurations']
                        out_roles[role_name]['scenarios'][scenario_name]['messages'] = \
                            role_data['scenarios'][scenario_name]['messages']

        out_roles = self._store_final_statistics(out_roles)
        return out_roles
