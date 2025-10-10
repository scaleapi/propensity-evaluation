import json
import logging
from copy import deepcopy
import os

from api_client.api_conf import APIConfiguration
from .base import KEYS_ORDERS, BasePipeline
from .modules.scenario_utils_messages_single import MessagesScenarioManager
from .modules.utils import save_to_disk, order_dict_keys


class PipelineMessages(BasePipeline):
    """Pipeline for generating messages with different pressure categories"""

    def __init__(self, cfg, logger: logging.Logger, workspace_name: str,
                 workspace_desc: str, workspace_alternative_forms: list,
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

        assert self.logger is not None

        api_conf = APIConfiguration(
            model_name=cfg.messages_model.model_name,
            model_provider=cfg.messages_model.model_provider,
            api_base=cfg.messages_model.api_base,
            use_cache=cfg.messages_model.use_cache,
        )

        judge_api_conf = APIConfiguration(
            model_name=cfg.judge_model.model_name,
            model_provider=cfg.judge_model.model_provider,
            api_base=cfg.judge_model.api_base,
            use_cache=cfg.judge_model.use_cache,
        )

        # Configure paths
        self.output_dir = output_dir
        self.single_messages_output_file = os.path.join(self.output_dir,
                                                        cfg.object_storage.scenarios_messages_single_fname)

        self.messages_scenario_manager = MessagesScenarioManager(
            api_conf=api_conf, judge_api_conf=judge_api_conf,
            logger=logger, workspace_name=self.workspace,
            workspace_alternative_forms=self.workspace_alternative_forms,
            workspace_desc=self.workspace_desc,
            domain_name=self.domain,
            domain_desc=self.domain_desc,
            domain_alternative_forms=self.domain_alternative_forms,
            inputs_conf=cfg.inputs,
            output_schemas_conf=cfg.output_schemas,
            prompts_conf=cfg.prompts,
            temperature=cfg.messages_model.temperature,
            roles_batch_size=cfg.roles_batch_size,
            scenarios_batch_size = cfg.scenario_gen_batch_size
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

    def run_gen_messages(self, input_roles, force_overwrite=False):
        curr_roles_with_scenarios = {}
        try:
            with open(self.single_messages_output_file, 'r') as f:
                curr_roles_with_scenarios = json.load(f)
        except FileNotFoundError as e:
            self.logger.warning(f"Could not find scenarios_messages_single file: {e}")
        except json.JSONDecodeError as e:
            self.logger.warning(f"Error decoding scenarios_messages_single file: {e}")

        # Determine if scenarios need to be generated
        should_generate = (force_overwrite or self.domain not in curr_roles_with_scenarios or
                           self.workspace not in curr_roles_with_scenarios.get(self.domain, {}))

        if should_generate:
            self.logger.info(f"Running sys-messages (regular) generation for workspace {self.workspace}...")
            new_roles_with_scenarios = self.messages_scenario_manager.generate_and_judge_scenarios(
                input_roles=input_roles[self.domain][self.workspace],
                logging=False
            )

            should_add = True
            if set(new_roles_with_scenarios.keys()) != set(input_roles.get(self.domain).get(self.workspace).keys()):
                should_add = False
            else:
                for role in new_roles_with_scenarios:
                    for scenario in new_roles_with_scenarios[role]['scenarios']:
                        curr_scen = new_roles_with_scenarios[role]['scenarios'][scenario]
                        if 'sys_messages' not in curr_scen:
                            should_add = False
                            break
                    if not should_add:
                        break

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
                self.logger.warning(f"Generated messages are incomplete. Skipping update.")

        return curr_roles_with_scenarios

    def run(self, roles: dict, force_overwrite=False):
        # Add system messages (applying pressure) to the scenarios from the previous steps
        curr_roles_with_messages = self.run_gen_messages(roles, force_overwrite)
        curr_roles_with_messages = order_dict_keys(curr_roles_with_messages, KEYS_ORDERS)

        save_to_disk(curr_roles_with_messages, self.single_messages_output_file)
