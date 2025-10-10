import os
import json
import logging
from copy import deepcopy

from api_client.api_conf import APIConfiguration
from .base import KEYS_ORDERS, BasePipeline
from .modules.scenario_utils_states import ScenarioManager as StatesScenarioManager
from .modules.scenario_utils_funcs import ScenarioManager as FuncsScenarioManager
from .modules.scenario_utils_messages import ScenarioManager as MessagesScenarioManager
from .modules.scenario_utils_policies import ScenarioManager as PoliciesScenarioManager
from .modules.utils import save_to_disk, order_dict_keys


class PipelineScenarios(BasePipeline):
    """Main system that orchestrates all components"""

    def __init__(self, cfg, logger: logging.Logger, workspace_name: str,
                 workspace_desc: str,
                 workspace_alternative_forms: list,
                 domain_name: str, domain_desc: str, domain_alternative_forms: list, output_dir: str):
        super().__init__(cfg)

        self.cfg = cfg
        self.logger = logger
        self.workspace = workspace_name
        self.workspace_desc = workspace_desc
        self.workspace_alternative_forms = workspace_alternative_forms
        self.domain = domain_name
        self.domain_desc = domain_desc
        self.domain_alternative_forms = domain_alternative_forms

        self.min_initial_scenarios_per_role = cfg.min_initial_scenarios_per_role
        self.min_chosen_scenarios_per_role = cfg.min_chosen_scenarios_per_role

        # self.n_retries = cfg.n_retries

        assert self.logger is not None

        api_conf = APIConfiguration(
            model_name=cfg.model.model_name,
            model_provider=cfg.model.model_provider,
            api_base=cfg.model.api_base,
            use_cache=cfg.model.use_cache,
        )

        judge_api_conf = APIConfiguration(
            model_name=cfg.judge_model.model_name,
            model_provider=cfg.judge_model.model_provider,
            api_base=cfg.judge_model.api_base,
            use_cache=cfg.judge_model.use_cache,
        )

        # Configure paths
        self.output_dir = output_dir
        self.states_output_file = os.path.join(self.output_dir, cfg.object_storage.scenarios_states_fname)
        self.funcs_output_file = os.path.join(self.output_dir, cfg.object_storage.scenarios_funcs_fname)
        self.policies_output_file = os.path.join(self.output_dir, cfg.object_storage.scenarios_policies_fname)
        self.messages_output_file = os.path.join(self.output_dir, cfg.object_storage.scenarios_messages_fname)

        self.states_scenario_manager = StatesScenarioManager(api_conf=api_conf, judge_api_conf=judge_api_conf,
                                                             logger=logger, workspace_name=self.workspace,
                                                             workspace_alternative_forms=self.workspace_alternative_forms,
                                                             workspace_desc=workspace_desc,
                                                             domain_name=self.domain,
                                                             domain_desc=self.domain_desc,
                                                             domain_alternative_forms=self.domain_alternative_forms,
                                                             output_schemas_conf=cfg.output_schemas,
                                                             prompts_conf=cfg.prompts,
                                                             temperature=cfg.model.temperature,
                                                             min_initial_scenarios_per_role=cfg.min_initial_scenarios_per_role,
                                                             min_chosen_scenarios_per_role=cfg.min_chosen_scenarios_per_role,
                                                             generation_batch_size=cfg.roles_batch_size)

        self.funcs_scenario_manager = FuncsScenarioManager(api_conf=api_conf, judge_api_conf=judge_api_conf,
                                                           logger=logger, workspace_name=self.workspace,
                                                           workspace_alternative_forms=self.workspace_alternative_forms,
                                                           workspace_desc=workspace_desc,
                                                           domain_name=self.domain,
                                                           domain_desc=self.domain_desc,
                                                           domain_alternative_forms=self.domain_alternative_forms,
                                                           output_schemas_conf=cfg.output_schemas,
                                                           prompts_conf=cfg.prompts,
                                                           temperature=cfg.model.temperature,
                                                           max_retires=cfg.max_retries_funcs,
                                                           roles_batch_size=cfg.roles_batch_size,
                                                           scenarios_batch_size=cfg.scenario_gen_batch_size,
                                                           )

        self.policies_scenario_manager = PoliciesScenarioManager(api_conf=api_conf, judge_api_conf=judge_api_conf,
                                                                 logger=logger, workspace_name=self.workspace,
                                                                 workspace_alternative_forms=self.workspace_alternative_forms,
                                                                 workspace_desc=workspace_desc,
                                                                 domain_name=self.domain,
                                                                 domain_desc=self.domain_desc,
                                                                 domain_alternative_forms=self.domain_alternative_forms,
                                                                 output_schemas_conf=cfg.output_schemas,
                                                                 prompts_conf=cfg.prompts,
                                                                 temperature=cfg.model.temperature,
                                                                 max_retires=cfg.max_retries_policies,
                                                                 roles_batch_size=cfg.roles_batch_size,
                                                                 scenarios_batch_size=cfg.scenario_gen_batch_size,
                                                                 )

        self.messages_scenario_manager = MessagesScenarioManager(api_conf=api_conf, judge_api_conf=judge_api_conf,
                                                                 logger=logger, workspace_name=self.workspace,
                                                                 workspace_alternative_forms=self.workspace_alternative_forms,
                                                                 workspace_desc=workspace_desc,
                                                                 domain_name=self.domain,
                                                                 domain_desc=self.domain_desc,
                                                                 domain_alternative_forms=self.domain_alternative_forms,
                                                                 output_schemas_conf=cfg.output_schemas,
                                                                 prompts_conf=cfg.prompts,
                                                                 temperature=cfg.model.temperature,
                                                                 max_retires=cfg.max_retries_messages,
                                                                 roles_batch_size=cfg.roles_batch_size,
                                                                 scenarios_batch_size=cfg.scenario_gen_batch_size,
                                                                 )

    def update_scenarios(self, prev_roles_with_scenarios: dict, new_roles_with_scenarios: dict) -> dict:
        out = deepcopy(prev_roles_with_scenarios)
        for role_k, role_v in new_roles_with_scenarios.items():
            if role_k not in prev_roles_with_scenarios:
                out[role_k] = role_v
            else:
                for scenario in role_v['scenarios'].values():
                    scenario_name = scenario['name']
                    out[role_k]['scenarios'][scenario_name].update(scenario)
        return out

    def run_gen_states(self, roles, grounding_attack_vectors, grounding_n_samples, force_overwrite):
        curr_roles_with_scenarios = {}
        try:
            with open(self.states_output_file, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"No existing scenarios file found: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding scenarios file: {e}")

        # Check if we need to generate scenarios for this domain/workspace
        should_generate = force_overwrite
        if not force_overwrite:
            if (self.domain not in curr_roles_with_scenarios or
                    self.workspace not in curr_roles_with_scenarios.get(self.domain, {})):
                should_generate = True
                self.logger.info(
                    f"No existing scenarios for domain {self.domain} and workspace {self.workspace}. Generating new "
                    f"ones.")
        if should_generate:
            self.logger.info(f"Running states generation for workspace {self.workspace}...")
            new_roles_with_scenarios = self.states_scenario_manager.generate_and_judge_scenarios(
                input_roles=roles,
                grounding_attack_vectors=grounding_attack_vectors,
                grounding_n_samples=grounding_n_samples,
                logging=True
            )
            self.logger.debug(f'New scenarios: {new_roles_with_scenarios}')

            # Initialize domain if not exists
            if self.domain not in curr_roles_with_scenarios:
                curr_roles_with_scenarios[self.domain] = {}

            # Update workspace data
            if self.workspace not in curr_roles_with_scenarios[self.domain]:
                curr_roles_with_scenarios[self.domain][self.workspace] = new_roles_with_scenarios
            else:
                # Update existing scenarios while preserving others
                curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
                    curr_roles_with_scenarios[self.domain][self.workspace],
                    new_roles_with_scenarios
                )
        return curr_roles_with_scenarios

    def run_gen_funcs(self, input_roles, force_overwrite=False):
        curr_roles_with_scenarios = {}
        try:
            with open(self.funcs_output_file, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"No existing scenarios_funcs file found: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding scenarios_funcs file: {e}")

        # Determine if scenarios need to be generated
        should_generate = (force_overwrite or self.domain not in curr_roles_with_scenarios or self.workspace not in
                           curr_roles_with_scenarios.get(self.domain, {}))

        if should_generate:
            self.logger.info(f"Running funcs generation for workspace {self.workspace}...")
            new_roles_with_scenarios = self.funcs_scenario_manager.generate_and_judge_scenarios(
                input_roles=input_roles[self.domain][self.workspace],
                logging=True)

            # Check if all the generated scenarios have functions and configurations generated for them
            should_add = True

            if should_add:  # Only add if functions and configurations were actually generated
                if self.domain not in curr_roles_with_scenarios:
                    curr_roles_with_scenarios[self.domain] = {}
                if self.workspace not in curr_roles_with_scenarios[self.domain]:
                    curr_roles_with_scenarios[self.domain][self.workspace] = {}

                curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
                    curr_roles_with_scenarios[self.domain].get(self.workspace, {}),
                    new_roles_with_scenarios)
            else:
                self.logger.warning(f"Generated functions/configurations are incomplete. Skipping update.")

        return curr_roles_with_scenarios

    def run_gen_policies(self, input_roles, force_overwrite=False):
        curr_roles_with_scenarios = {}
        try:
            with open(self.policies_output_file, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not find scenarios_policies file: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding scenarios_policies file: {e}")

        # Determine if scenarios need to be generated
        should_generate = force_overwrite or self.domain not in curr_roles_with_scenarios or self.workspace not in curr_roles_with_scenarios.get(
            self.domain, {})

        if should_generate:
            self.logger.info(f"Running policies generation for workspace {self.workspace}...")
            new_roles_with_scenarios = self.policies_scenario_manager.generate_and_judge_scenarios(
                input_roles=input_roles[self.domain][self.workspace],
                logging=True
            )

            # Check if all the generated scenarios have policies generated for them
            should_add = True

            if should_add:  # Only add if policies were actually generated
                if self.domain not in curr_roles_with_scenarios:
                    curr_roles_with_scenarios[self.domain] = {}
                if self.workspace not in curr_roles_with_scenarios[self.domain]:
                    curr_roles_with_scenarios[self.domain][self.workspace] = {}

                curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
                    curr_roles_with_scenarios[self.domain].get(self.workspace, {}),
                    new_roles_with_scenarios
                )
            else:
                self.logger.error(f"Generated policies are incomplete. Skipping update.")

        return curr_roles_with_scenarios

    def run_gen_messages(self, input_roles, force_overwrite=False):
        curr_roles_with_scenarios = {}
        try:
            with open(self.messages_output_file, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not find scenarios_messages file: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding scenarios_messages file: {e}")

        # Determine if scenarios need to be generated
        should_generate = (force_overwrite or self.domain not in curr_roles_with_scenarios or
                           self.workspace not in curr_roles_with_scenarios.get(self.domain, {}))

        if should_generate:
            self.logger.info(f"Running task/neutral messages generation for workspace {self.workspace}...")
            new_roles_with_scenarios = self.messages_scenario_manager.generate_and_judge_scenarios(
                input_roles=input_roles[self.domain][self.workspace],
                logging=True
            )

            should_add = True

            if should_add:  # Only add if policies were actually generated
                if self.domain not in curr_roles_with_scenarios:
                    curr_roles_with_scenarios[self.domain] = {}
                if self.workspace not in curr_roles_with_scenarios[self.domain]:
                    curr_roles_with_scenarios[self.domain][self.workspace] = {}

                curr_roles_with_scenarios[self.domain][self.workspace] = self.update_scenarios(
                    curr_roles_with_scenarios[self.domain].get(self.workspace, {}),
                    new_roles_with_scenarios
                )
            else:
                self.logger.error(f"Generated messages are incomplete. Skipping update.")

        return curr_roles_with_scenarios

    def run(self, roles: dict, grounding_attack_vectors, grounding_n_samples, force_overwrite=False):
        # Generate the initial scenarios with states
        curr_roles_with_states = self.run_gen_states(roles, grounding_attack_vectors, grounding_n_samples,
                                                     force_overwrite)
        curr_roles_with_states = order_dict_keys(curr_roles_with_states, KEYS_ORDERS)
        save_to_disk(curr_roles_with_states, self.states_output_file)

        # Add functions and configurations to the generated scenarios
        curr_roles_with_funcs = self.run_gen_funcs(curr_roles_with_states)
        curr_roles_with_funcs = order_dict_keys(curr_roles_with_funcs, KEYS_ORDERS)
        save_to_disk(curr_roles_with_funcs, self.funcs_output_file)

        # Add policies to the generated scenarios
        curr_roles_with_policies = self.run_gen_policies(curr_roles_with_funcs)
        curr_roles_with_policies = order_dict_keys(curr_roles_with_policies, KEYS_ORDERS)
        save_to_disk(curr_roles_with_policies, self.policies_output_file)

        # Add task and neutral messages to the generated scenarios (not system messages yet)
        curr_roles_with_messages = self.run_gen_messages(curr_roles_with_policies)
        curr_roles_with_messages = order_dict_keys(curr_roles_with_messages, KEYS_ORDERS)

        save_to_disk(curr_roles_with_messages, self.messages_output_file)
