import json
from copy import deepcopy
from omegaconf import DictConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

from api_client.client import Client
from .utils import read_prompts, run_agent_query, check_for_missing_fields, load_output_schemas, remove_nested_fields
from .utils import json_obj_list_to_dict


def get_valid_scenarios(roles: dict, required_fields: list):
    failed_roles = check_for_missing_fields(roles, required_fields)
    if failed_roles:
        roles = {k: v for k, v in roles.items() if v['name'] not in failed_roles}
    return roles


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

        sys_prompt = read_prompts(self.prompts_conf.scenarios_agents_policies, key='SYS_GEN',
                                  context={'general_body': general_body}, logger=self.logger)

        if self.roles_batch_size == 1:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_policies_single)
        else:
            output_schema = load_output_schemas(self.output_schemas_conf.scenarios_gen_policies)
        return Client(
            api_conf=self.api_conf,
            sys_prompt=sys_prompt,
            output_schema=output_schema,
            temperature=self.temperature)

    def _init_scenarios_verif_judge(self):
        sys_prompt = read_prompts(self.prompts_conf.judge_agents, key='SYS_SCENARIOS_POLICIES_VERIF',
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
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            self.logger.error(
                f"Error occurred in __record_failure: {e}")

    def check_configurations_are_valid(self, roles_with_scenarios: dict):
        valid_scenarios = {}
        for role_k, role_v in roles_with_scenarios.items():
            for scenario_k, scenario_v in role_v['scenarios'].items():
                try:
                    conf_known_dict = json.loads(scenario_v['configurations']['configuration_known'])
                    conf_unknown_dict = json.loads(scenario_v['configurations']['configuration_unknown'])

                    conf_known_dict = {k: v for k, v in conf_known_dict.items() if
                                       v and k not in conf_unknown_dict.keys()}
                    conf_unknown_dict = {k: v for k, v in conf_unknown_dict.items() if
                                         v and k not in conf_known_dict.keys()}

                    assert len(conf_known_dict) > 0
                    assert len(conf_unknown_dict) > 0

                    valid_scenarios[role_k] = role_v
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_known'] = str(conf_known_dict).replace("'", '"')
                    # print(str(conf_known_dict).replace("'", '"'))
                    valid_scenarios[role_k]['scenarios'][scenario_k]['configurations'][
                        'configuration_unknown'] = str(conf_unknown_dict).replace("'", '"')
                    # print(str(conf_unknown_dict).replace("'", '"'))
                except Exception as e:
                    self.__record_failure(self.generation_statistics, role_name=role_k, scenario_name=scenario_k,
                                          failure_subcategory=f'check_configurations_are_valid:{type(e).__name__}')
                    # exc_type, exc_obj, exc_tb = sys.exc_info()
                    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    self.logger.error(
                        f"Error in check_configurations_are_valid for scenario {role_k}:{scenario_k}: {e}")
                    continue
        return valid_scenarios

    def generate_scenarios(self, input_roles: dict):
        """
        Generates scenarios for all roles in batches concurrently and validate them.
        Re-generates scenarios only for invalid generations.
        """
        self.logger.debug("Starting scenario generation.")
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

                        # Run validity checks
                        response = get_valid_scenarios(response, required_fields=['name', 'scenarios'])

                        # Process the response for each role in the batch
                        for role_name, role_data in batch_roles.items():
                            if role_name not in response:
                                self.logger.error(f"Role '{role_name}' not found in response")
                                continue

                            response[role_name]['scenarios'] = json_obj_list_to_dict(response[role_name]['scenarios'],
                                                                                     'name')
                            for scenario_name, scenario_data in role_data['scenarios'].items():
                                try:
                                    scenario_response = response[role_name]['scenarios'][scenario_name]

                                    # Ensure valid_scenarios retains all original data
                                    assert isinstance(scenario_response['name'], str), \
                                        f"Scenario name for '{role_name}:{scenario_name}' is not a string"
                                    assert isinstance(scenario_response['policy'], dict), \
                                        f"Scenario policy for '{role_name}:{scenario_name}' is not a dict"

                                    if role_name not in valid_scenarios:
                                        valid_scenarios[role_name] = deepcopy(role_data)  # Copy all top-level fields
                                        valid_scenarios[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_scenarios[role_name]['scenarios']:
                                        valid_scenarios[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)

                                    # Update with scenario fields
                                    valid_scenarios[role_name]['scenarios'][scenario_name].update(
                                        response[role_name]['scenarios'][scenario_name]
                                    )

                                    # Remove the role:scenario from invalid_roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]
                                except Exception as e:
                                    self.__record_failure(self.generation_statistics, role_name=role_name,
                                                          scenario_name=scenario_name,
                                                          failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                                    # exc_type, exc_obj, exc_tb = sys.exc_info()
                                    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    self.logger.error(
                                        f"Invalid scenario for role '{role_name}', scenario '{scenario_name}': {e}")

                            # Remove the role from invalid_roles
                            if role_name in invalid_roles and len(invalid_roles[role_name]['scenarios']) == 0:
                                del invalid_roles[role_name]

                    except Exception as e:
                        self.__record_failure(self.generation_statistics, role_name=role_name,
                                              scenario_name=scenario_name,
                                              failure_subcategory=f'generate_scenarios:{type(e).__name__}')
                        # exc_type, exc_obj, exc_tb = sys.exc_info()
                        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        self.logger.error(
                            f"Error processing batch: {e}")

        return valid_scenarios

    def _process_batch_generate_scenarios(self, batch_roles):
        """
        Helper method to process a single batch for scenario generation.
        """
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'any.illegal_state.attack_vector_description',
                                                                          'any.illegal_state.intentional_adversarial_action_taken',
                                                                          'any.illegal_state.intentional_adversarial_action_implementation_details',
                                                                          'any.illegal_state.task_achievement_mechanism',
                                                                          ])

        prompt = read_prompts(self.prompts_conf.scenarios_agents_policies, key='USER_GEN',
                              context={'roles': batch_roles},
                              logger=self.logger)
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
            response = run_agent_query(
                prompt=prompt,
                agent=self.scenarios_generation_agent,
                logger=self.logger,
                to_json=True,
                json_transform_keys=['roles', 'name']
            )

        return response

    def judge_scenarios(self, roles: dict):
        """
        Judge scenarios for all roles in batches concurrently and re-process only invalid judgments.
        """
        self.logger.debug(f"Running judge_scenarios")
        valid_judgments = {}
        invalid_roles = deepcopy(roles)
        batch_size = self.roles_batch_size  # Define batch size for processing

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

                                    assert isinstance(scenario_response['acceptable'], bool), \
                                        f"Judge acceptable flag for scenario '{role_name}:{scenario_name}' is not a boolean"
                                    assert scenario_response[
                                        'feedback'].strip(), f"Judge feedback is empty for scenario '{role_name}:{scenario_name}'"

                                    # Ensure valid_judgments retains all original data
                                    if role_name not in valid_judgments:
                                        valid_judgments[role_name] = deepcopy(role_data)
                                        valid_judgments[role_name]['scenarios'] = {}

                                    if scenario_name not in valid_judgments[role_name]['scenarios']:
                                        valid_judgments[role_name]['scenarios'][scenario_name] = deepcopy(scenario_data)

                                    # Update with judgment fields
                                    valid_judgments[role_name]['scenarios'][scenario_name]['policy'].update({
                                        'acceptable': scenario_response['acceptable'],
                                        'feedback': scenario_response['feedback']
                                    })

                                    # Remove the role:scenario from invalid_roles
                                    if scenario_name in invalid_roles[role_name]['scenarios']:
                                        del invalid_roles[role_name]['scenarios'][scenario_name]

                                except Exception as e:
                                    self.__record_failure(self.judgment_statistics, role_name=role_name,
                                                          scenario_name=scenario_name,
                                                          failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                                    # exc_type, exc_obj, exc_tb = sys.exc_info()
                                    # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                                    self.logger.error(
                                        f"Invalid judgment for role '{role_name}', scenario '{scenario_name}': {e}")

                            # Remove the role from invalid_roles
                            if role_name in invalid_roles and len(invalid_roles[role_name]['scenarios']) == 0:
                                del invalid_roles[role_name]

                    except json.JSONDecodeError:
                        self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                        prev_batch_size = batch_size
                        batch_size = max(1, batch_size // 2)
                        self.logger.warning(
                            f"Error in generate_scenarios; Reducing batch size from {prev_batch_size} to {batch_size}.")
                    except Exception as e:
                        self.__record_failure(self.judgment_statistics, batch_roles=batch_roles,
                                              failure_subcategory=f'judge_scenarios:{type(e).__name__}')
                        # exc_type, exc_obj, exc_tb = sys.exc_info()
                        # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        self.logger.error(
                            f"Error processing batch: {e}")

        return valid_judgments

    def _process_batch_judge_scenarios(self, batch_roles):
        """
        Helper method to process a single batch for scenario judgment.
        """
        batch_roles = remove_nested_fields(batch_roles, fields_to_remove=['acceptable', 'feedback',
                                                                          'any.acceptable',
                                                                          'any.feedback',
                                                                          'statistics',
                                                                          'any.statistics',
                                                                          'any.illegal_state.attack_vector_description',
                                                                          'any.illegal_state.intentional_adversarial_action_taken',
                                                                          'any.illegal_state.intentional_adversarial_action_implementation_details',
                                                                          'any.illegal_state.task_achievement_mechanism',
                                                                          ])

        prompt = read_prompts(self.prompts_conf.judge_agents, key='USER_SCENARIOS_VERIF',
                              context={'roles': batch_roles},
                              logger=self.logger)

        response = run_agent_query(
            prompt=prompt,
            agent=self.scenarios_verif_judge,
            logger=self.logger,
            to_json=True,
            json_transform_keys=['roles', 'name']
        )
        return response

    def _store_final_statistics(self, accepted_scenarios: dict):
        accepted_scenarios = deepcopy(accepted_scenarios)
        for role_k, role_v in accepted_scenarios.items():
            if 'statistics' not in role_v:
                role_v['statistics'] = {}
            role_v['statistics']['policies'] = {
                'generation': self.generation_statistics.get(role_k, {}),
                'judgment': self.judgment_statistics.get(role_k, {})
            }
        return accepted_scenarios

    def generate_and_judge_scenarios(self, input_roles: dict, logging=True):
        """
        Generate and judge scenarios iteratively, re-generating only invalid scenarios.
        """
        self.__reset_statistics()

        out_roles = deepcopy(input_roles)  # Retain all fields in the output
        curr_roles = deepcopy(out_roles)

        tries = 0
        # max_tries = 10
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

            self.logger.debug(f"Tries: {tries}")
            for role_name, role_data in curr_roles.items():
                num_scenarios = len(role_data['scenarios'])
                self.logger.debug(f"Remaining Role: {role_name}, Scenarios Left: {num_scenarios}")

            # Generate scenarios for all roles
            generated_scenarios = self.generate_scenarios(curr_roles)
            if logging:
                self.logger.debug(f"Generated scenarios: {json.dumps(generated_scenarios, indent=2)}")

            # Judge the generated scenarios
            judged_scenarios = self.judge_scenarios(generated_scenarios)
            if logging:
                self.logger.debug(f"Judged scenarios: {json.dumps(judged_scenarios, indent=2)}")

            # Filter out invalid roles and scenarios
            curr_roles = {
                role_name: {
                    **role_data,
                    'scenarios': {
                        scenario_name: {
                            **{
                                key: value
                                for key, value in scenario_data.items()
                                if key != 'policy'
                            },
                            'feedback_for_previous_failed_attempt': scenario_data['policy']['feedback'],
                        }
                        for scenario_name, scenario_data in role_data['scenarios'].items()
                        if not judged_scenarios[role_name]['scenarios'][scenario_name]['policy']['acceptable']
                    }
                }
                for role_name, role_data in judged_scenarios.items()
                if any(
                    not scenario_data['policy']['acceptable']
                    for scenario_data in role_data['scenarios'].values()
                )
            }

            # Update accepted scenarios with valid messages
            for role_name, role_data in judged_scenarios.items():
                for scenario_name, scenario_data in role_data['scenarios'].items():
                    if scenario_data['policy']['acceptable']:
                        out_roles[role_name]['scenarios'][scenario_name]['policy'] = \
                            role_data['scenarios'][scenario_name][
                                'policy']

        out_roles = remove_nested_fields(out_roles, fields_to_remove=['feedback_for_previous_failed_attempt'])
        out_roles = self._store_final_statistics(out_roles)
        return out_roles
