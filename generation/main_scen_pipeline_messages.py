from pathlib import Path
import os
import sys

# Set directory to project root (parent of current file's parent)
root_dir = Path(__file__).resolve().parent.parent
os.chdir(root_dir)
sys.path.insert(0, str(root_dir))
print(f"Curr working dir: {root_dir}")

import hydra
from dotenv import load_dotenv
from utils.colors import BaseColoredFormatter
import logging
from copy import deepcopy
from pipeline.messages_pipeline import PipelineMessages


def setup_logger():
    logger = logging.getLogger()
    logging.getLogger().setLevel(logging.DEBUG)
    for logger_name in logging.root.manager.loggerDict:
        print(f'Muting logs from {logger_name}')
        logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    print()

    ch = logging.StreamHandler()

    ch.setLevel(logging.DEBUG)

    formatter = BaseColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    ch.setFormatter(formatter)

    for handler in logging.root.handlers[:]:
        print(f'Removing logging handler: {handler}')
        logging.root.removeHandler(handler)

    logger.addHandler(ch)
    return logger


def init_attack_vectors():
    attack_vectors = {}
    base_dir = 'inputs/attacks'

    # Scan all subdirectories in inputs/attacks
    for domain in os.listdir(base_dir):
        domain_path = os.path.join(base_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        # Look for attacks.json file in each subdirectory
        attacks_file = os.path.join(domain_path, 'attacks.json')
        if os.path.exists(attacks_file):
            with open(attacks_file, 'r') as f:
                attack_vectors[domain] = json.load(f)

    return attack_vectors


def get_attack_vectors_for_roles(roles, domain_attack_vectors: dict, logger):
    role_attack_vectors = {}
    for role_k, role_v in roles.items():
        if not role_v['attack_vectors']:
            role_attack_vectors[role_k] = list(domain_attack_vectors.values())
        else:
            role_attack_vectors[role_k] = [domain_attack_vectors[av] for av in role_v['attack_vectors']]

    return role_attack_vectors


def remove_keys_from_dictv(roles, keys: list):
    out = deepcopy(roles)
    for k, v in roles.items():
        for key in keys:
            if key in v:
                out[k].pop(key)

    return out


from concurrent.futures import ProcessPoolExecutor
import os
import json


def process_workspace(cfg, logger, domain_name, workspace_name, workspace_data, domain_desc, domain_alternative_forms,
                      loaded_roles_with_scenarios, output_dir):
    """
    Process a single workspace in the messages pipeline.
    """
    workspace_desc = workspace_data.get('description')
    workspace_alternative_forms = workspace_data.get('alternative_forms')

    sim = PipelineMessages(cfg=cfg,
                           logger=logger,
                           workspace_name=workspace_name,
                           workspace_desc=workspace_desc,
                           workspace_alternative_forms=workspace_alternative_forms,
                           domain_name=domain_name,
                           domain_desc=domain_desc,
                           domain_alternative_forms=domain_alternative_forms,
                           output_dir=str(output_dir)
                           )

    sim.run(
        roles=loaded_roles_with_scenarios,
        force_overwrite=False,
    )


@hydra.main("configs", config_name="pipeline", version_base="1.2")
def main(cfg) -> None:
    logger = setup_logger()

    loaded_workspaces = [
        'bio-security.Agricultural-Biotech-Facility',
        'bio-security.BSL-3-4-High-Containment-Laboratory',
        'bio-security.Cloud-Based-Bioinformatics-&-DNA-Database-Service',
        'bio-security.Marine-Biotechnology-Research-Institute',
        'bio-security.Pandemic-Response-&-Biosecurity-Center',
        'bio-security.Personalized-Medicine-&-Gene-Therapy-Lab',
        'bio-security.Pharmaceutical-R&D-Laboratory',
        'bio-security.Sustainable-Biomanufacturing-Innovation-Center',

        'chemical-security.Agricultural-Chemical-Development-Facility',
        'chemical-security.Chemical-Waste-Management-Facility',
        'chemical-security.Defense-Contractor-Chemical-Research-Division',
        'chemical-security.Forensic-Chemistry-Laboratory',
        'chemical-security.Independent-Contract-Research-Organization',
        'chemical-security.Industrial-Chemical-Manufacturing-Plant',
        'chemical-security.Materials-Science-Innovation-Center',
        'chemical-security.Pharmaceutical-Development-Company',

        'cyber-security.Datacenter',
        'cyber-security.Enterprise-Cybersecurity-Solutions-and-Threat-Mitigation-Provider',
        'cyber-security.Confidential-Legal-Operations-and-Data-Management-Firm',
        'cyber-security.Advanced-Space-Exploration-and-Telemetry-Command-Center',
        'cyber-security.Advanced-Smart-Grid-Distribution-and-Control-Center',
        'cyber-security.E-Commerce-Distribution-Center',
        'cyber-security.Government-Cybersecurity-and-Intelligence-Operations-Agency',
        'cyber-security.Smart-City-Traffic-Management-Center',

        'self-proliferation.Autonomous-Vehicle-Fleet-Management-System',
        'self-proliferation.Automated-Disaster-Response-System',
        'self-proliferation.Authorship-and-Citation-Network',
        'self-proliferation.Global-Trade-Negotiation-Platform',
        'self-proliferation.Healthcare-Data-Management-Platform',
        'self-proliferation.Cloud-Platform-with-Autoscaling',
        'self-proliferation.Informal-Loan-Club-and-Credit-Circle',
        'self-proliferation.Philanthropic-Organization',
    ]

    attack_vectors = init_attack_vectors()
    base_inputs_dir = 'inputs/workspaces'
    with open(os.path.join(base_inputs_dir, 'domains.json'), 'r') as file:
        domain_data = json.load(file)

    workspaces = []
    for domain_name in os.listdir(base_inputs_dir):
        if not os.path.isdir(os.path.join(base_inputs_dir, domain_name)):
            continue
        assert domain_name in attack_vectors, f'No attack vectors found for domain {domain_name}. Available domains: {attack_vectors.keys()}'
        domain_desc = domain_data[domain_name].get('description')
        domain_alternative_forms = domain_data[domain_name].get('alternative_forms')
        for filename in os.listdir(os.path.join(base_inputs_dir, domain_name)):
            if filename.endswith('.json'):
                filepath = os.path.join(base_inputs_dir, domain_name, filename)
                with open(filepath, 'r') as file:
                    workspace_data = json.load(file)
                    workspace_name = workspace_data.get('name')

                if loaded_workspaces and f'{domain_name}.{workspace_name}' not in loaded_workspaces:
                    logger.info(f'Skipping workspace {domain_name}.{workspace_name}')
                    continue

                # Load messages from a file specific to the workspace
                output_dir = os.path.join(cfg.object_storage.base_dir, domain_name.lower().replace(' ', '-'),
                                          workspace_name.lower().replace(' ', '-'))
                messages_file = os.path.join(output_dir, cfg.object_storage.scenarios_messages_fname)
                try:
                    with open(messages_file, 'r') as msg_file:
                        workspace_messages = json.load(msg_file)
                except FileNotFoundError:
                    logger.error(f"Messages file not found for workspace: {workspace_name}; you must first run the scenarios pipeline.")
                    continue

                workspaces.append(
                    (cfg, logger, domain_name, workspace_name, workspace_data, domain_desc, domain_alternative_forms,
                     workspace_messages,
                     os.path.join(cfg.object_storage.base_dir, domain_name.lower().replace(' ', '-'),
                                  workspace_name.lower().replace(' ', '-')))
                )

    # Process workspaces in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=len(workspaces)) as executor:
        futures = [
            executor.submit(process_workspace, *workspace_args)
            for workspace_args in workspaces
        ]

        for future in futures:
            try:
                future.result()  # Wait for each process to complete
            except Exception as e:
                logger.error(f"Error processing workspace: {e}")


if __name__ == "__main__":
    # Load environment variables from .env file for LLM API keys and access information
    if not os.path.exists('.env'):
        print(
            "Warning: .env file not found. Make sure to have environment variables set for the necessary LLM API keys.")
    else:
        load_dotenv()

    assert os.environ.get(
        'API_KEYS'), "API_KEYS environment variable not set. Is required for Litellm inference (even if using local models)."

    # Load environment variables
    main()
