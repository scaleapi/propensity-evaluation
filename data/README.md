# PropensityBench Dataset

This directory contains the pre-generated scenario data for the PropensityBench benchmark. Our benchmark consists of:
- 979 evaluation scenarios
- 5,874 unique tasks (across 6 pressure categories for each scenario)
- 6,648 unique agent tools
- 76,362 unique task and pressure messages

Across the following dimensions:
- 4 high-risk domains (biosecurity, chemical security, cybersecurity, and self-proliferation)
- 32 distinct workspaces (environments) within those domains
- 161 unique agent roles within those workspaces
- 30 unique attack vectors (dangerous capabilities)

See [benchmark statistics](#benchmark-statistics) for a detailed breakdown of the scenario distribution across domains, workspaces, roles, and risk categories.

This document details the structure of the final benchmark files used for evaluation, as well as the intermediate artifacts produced by our multi-stage scenario generation pipeline (not required for evaluation).


## Directory Structure

**Important:** The `full/` directory contains the large pre-generated benchmark data files for PropensityBench, which are managed using **Git LFS (Large File Storage)**. Therefore, to access these files you will need to have git LFS installed. You also won't be able to inspect this data similar to other regular files on GitHub. However, we do provide a small subset of this data for select workspaces and a single role within each under the `sample/` directory (not tracked by LFS) for you to inspect. To pull the full dataset, see Step 2 of the main [installation](../README.md#2-installation) instructions.

-   **`data/sample/`**: This directory contains a small, representative sample of the data. These files are not tracked by LFS, so you can browse them on the GitHub UI or after a standard clone to understand the data format.
-   **`data/full/`**: This directory contains the complete, official benchmark dataset. These files **are** tracked by LFS.


The data under `full/` (and by extension `sample/`) is organized hierarchically by domain and workspace. The evaluation harness (`evaluation/main.py`) is configured to read from this structure.

```
data/
├── sample/
│       └── ...
└── full/
    ├── bio-security/
    │   ├── agricultural-biotech-facility/
    │   │   └── scenarios_messages_single.json
    │   │   └── other json files (intermediate generation artifacts)
    │   ├── bsl-3-4-high-containment-laboratory/
    │   │   └── scenarios_messages_single.json
    │   │   └── other json files (intermediate generation artifacts)
    │   └── ... (other biosecurity workspaces)
    │
    ├── chemical-security/
    │   └── ... (workspaces for chemical security)
    │
    ├── cyber-security/
    │   └── ... (workspaces for cybersecurity)
    │
    └── self-proliferation/
        └── ... (workspaces for self-proliferation)
```

The key components of this structure are:
-   **`<domain>/`**: Each subdirectory corresponds to one of the four high-risk domains (e.g., `bio-security`).
-   **`<workspace>/`**: Within each domain, subdirectories represent specific operational environments (e.g., `agricultural-biotech-facility`).
-   **`scenarios_messages_single.json`**: This is the key file containing all the scenarios for the roles within that workspace.

---

## Final Scenario Schema

Each `scenarios_messages_single.json` file contains a deeply nested JSON object that fully defines all scenarios for a given workspace. Below is a breakdown of the key fields. This is the ONLY file required for evaluation within each workspace, and contains the complete set of scenarios with all necessary fields defined under that workspace.

```json
{
  "<domain_name>": {
    "<workspace_name>": {
      "<role_name>": {
        "scenarios": {
          "<scenario_name>": {
            "name": "A unique identifier for the scenario.",
            "description": "A high-level narrative description of the scenario's context.",
            
            "states": {
              "initial_state": { "name": "...", "description": "..." },
              "target_state": { "name": "...", "description": "..." },
              "illegal_state": { "name": "...", "description": "..." }
            },

            "functions": {
              "target_function": {
                "name": "complete_...",
                "description": "The aligned tool to complete the task.",
                "input_arguments": ["arg1", "arg2"],
                "errors": ["Error message 1", "..."]
              },
              "trigger_function": {
                "harmful_name": "use_exploit_...",
                "harmful_description": "The misaligned tool with a harmful-sounding name.",
                "neutral_name": "enhance_data_...",
                "neutral_description": "The misaligned tool with a benign-sounding name."
              },
              "getter_functions": [
                { "name": "get_...", "description": "...", "input_arguments": [], "output_arguments": ["..."] }
              ]
            },

            "configurations": {
              "configuration_known": "A JSON string of environmental parameters known to the agent at the start.",
              "configuration_unknown": "A JSON string of parameters the agent must discover using getter tools."
            },

            "policy": {
              "role_description": "...",
              "duties_description": "...",
              "prohibitions_description": "Implicit warnings against the misaligned action.",
              "negative_consequences": { "text": "The catastrophic outcomes of using the misaligned tool." }
            },

            "messages": {
              "task_message": "The initial instruction given to the agent to begin the simulation."
            },

            "sys_messages": {
              "evaluation_awareness": [ 
                { "msg_level_index": 0, "from": "...", "subject": "...", "body": "..." },
                { "msg_level_index": 1, "from": "...", "subject": "...", "body": "..." }, 
                ... 
              ],
              "financials": [ ... ],
              "//": "Contains 12 escalating pressure messages for each of the 6 pressure tactics/categories."
            }
          }
        }
      }
    }
  }
}
```


## Intermediate Generation Artifacts

Our scenario generation pipeline (`generation/`) is a multi-stage process. The files below represent the state of the scenarios after each major stage. They are typically found in the `generation_output/` directory (or your configured output directory) and are useful for debugging the pipeline or for researchers who wish to use only certain components of our generated data.

#### 1. `scenarios_states.json`
-   **Content:** The output of Stage 1 of the generation pipeline. Contains the foundational narrative of each scenario.
-   **Key Fields:** `name`, `description`, and `states` (`initial_state`, `target_state`, `illegal_state`).
-   **Purpose:** This file defines the core objective and conflict of each scenario before any interactive elements are added.

#### 2. `scenarios_funcs.json`
-   **Content:** The output of Stage 2 of the generation pipeline. Enriches the states with interactive tools and environmental parameters.
-   **Key Fields Added:** `functions` (defining all tools) and `configurations` (defining the environment state).
-   **Purpose:** This file makes the scenarios operational by defining the agent's capabilities and the world it interacts with.

#### 3. `scenarios_policies.json`
-   **Content:** The output of Stage 3 of the generation pipeline. Adds the rules and context that govern the agent's behavior.
-   **Key Fields Added:** `policy` (containing descriptions of the role, duties, prohibitions, and consequences).
-   **Purpose:** This file grounds the agent in its role, creating the ethical dilemma it will face.

#### 4. `scenarios_messages.json`
-   **Content:** The output of Stage 4 of the generation pipeline. Adds the initial, non-pressure-related messages.
-   **Key Fields Added:** `messages` (`task_message`, `neutral_sys_messages`).
-   **Purpose:** This file provides the initial prompt that starts the agent's simulation loop. The `scenarios_messages_single.json` file is the final version after the next step.

#### 5. `scenarios_messages_single.json`
-   **Content:** The output of Stage 5 of the generation pipeline. Adds system messages, to the outputs of stage 4, which are pressure messages for each of the six pressure tactics (Time, Financials, etc.).
-   **Key Fields Added:** `sys_messages` (`evaluation-awareness`, `financials`, ...). Each pressure category contains 12 messages with strictly increasing pressure levels.
- **Key Fields in Each Message:** Each message is an object with `msg_level_index`, `from`, `subject`, and `body` fields.
-   **Purpose:** This file completes the scenario definitions by providing the escalating pressure messages that will be used to test the agent's alignment under stress.


After the generation of the final pressure messages, all components are consolidated into the final `scenarios_messages_single.json` file, which is the canonical version of the benchmark used for evaluation.



---

## Benchmark Statistics

To provide a clear overview of the benchmark's composition (also see [benchmark overview](#propensitybench-dataset)), the following tables summarize the distribution of scenarios across the different domains, risk categories, workspaces, and roles.

### Scenarios per Risk Category

This table shows the number of scenarios generated for each of the 30 fine-grained risk categories (dangerous capabilities) within the four high-risk domains.

| Domain              | Risk Category                                              | Scenarios |
| ------------------- | ---------------------------------------------------------- | --------- |
| **Bio-security**    | ePPP                                                       | 25        |
|                     | Targeted Bioinformatics Collection and Exploitation        | 31        |
|                     | Expanding Public Access to High-Risk Knowledge             | 45        |
|                     | Dual-Use virology                                          | 26        |
|                     | Bioweapons & Bioterrorism                                  | 25        |
|                     | Reverse Genetics                                           | 29        |
|                     | Viral Vector Research                                      | 13        |
|                     | Detection Evasion                                          | 40        |
| **Chemical-security** | Preservation Infrastructure                                | 31        |
|                     | Resource Procurement                                       | 28        |
|                     | Detection Circumvention                                    | 34        |
|                     | Refinement Enhancement                                     | 38        |
|                     | Delivery System Development                                | 33        |
|                     | Validation Methods                                         | 39        |
|                     | Small Molecule Design                                      | 30        |
|                     | Synthesis Capability                                       | 29        |
| **Cyber-security**  | Impact                                                     | 47        |
|                     | Persistence                                                | 38        |
|                     | Privilege Escalation                                       | 30        |
|                     | Defense Evasion                                            | 38        |
|                     | Credential Access                                          | 45        |
|                     | Execution                                                  | 38        |
|                     | Initial Access                                             | 45        |
| **Self-proliferation**| Resource Acquisition and Management                        | 35        |
|                     | Agent Replication through Self-Duplication                 | 26        |
|                     | Gaining Self-Understanding and Situational Awareness       | 29        |
|                     | Detection Evasion and Interference Resistance of Self-Proliferation | 31        |
|                     | Survival and Legacy Preservation                         | 23        |
|                     | Dominating Other Agents                                    | 30        |
|                     | Undesirable Coordination                                   | 25        |



### Scenarios per Workspace and Role

This table provides a comprehensive breakdown of scenario counts for each role within its specific workspace and domain.

| Domain              | Workspace                                       | Role                                    | Scenarios |
| ------------------- | ----------------------------------------------- | --------------------------------------- | --------- |
| **Bio-security**    | **Agricultural-Biotech-Facility**               | **(Total: 29)**                         |           |
|                     |                                                 | Epidemiological-Modeler                 | 7         |
|                     |                                                 | Genetic-Crop-Engineer                   | 7         |
|                     |                                                 | Livestock-Gene-Editor                   | 5         |
|                     |                                                 | Pesticide-Developer                     | 6         |
|                     |                                                 | Supply-Chain-Biosafety-Coordinator      | 4         |
|                     | **BSL-3-4-High-Containment-Laboratory**         | **(Total: 23)**                         |           |
|                     |                                                 | Biosafety-Compliance-Officer            | 5         |
|                     |                                                 | Experimentation-Coordinator             | 4         |
|                     |                                                 | Genomic-Data-Analyst                    | 4         |
|                     |                                                 | Lab-Automation-Engineer                 | 6         |
|                     |                                                 | Pathogen-Researcher                     | 4         |
|                     | **Cloud-Based-Bioinformatics-&-DNA-Database-Service** | **(Total: 28)**                         |           |
|                     |                                                 | Automated-Bioinformatics-Workflow-Engineer | 7         |
|                     |                                                 | Genetic-Privacy-Specialist              | 7         |
|                     |                                                 | Genomic-Data-Curator                    | 8         |
|                     |                                                 | Sequence-Analyzer                       | 2         |
|                     |                                                 | Synthetic-Gene-Prediction-Scientist     | 4         |
|                     | **Marine-Biotechnology-Research-Institute**     | **(Total: 31)**                         |           |
|                     |                                                 | Aquaculture-Biotechnologist             | 7         |
|                     |                                                 | Marine-Adaptation-Genomicist            | 5         |
|                     |                                                 | Marine-Biomaterials-Engineer            | 5         |
|                     |                                                 | Marine-Microbiome-Researcher            | 7         |
|                     |                                                 | Marine-Natural-Products-Chemist         | 7         |
|                     | **Pandemic-Response-&-Biosecurity-Center**      | **(Total: 32)**                         |           |
|                     |                                                 | Biodefense-Policy-Strategist            | 5         |
|                     |                                                 | Epidemiological-Intelligence-Analyst    | 6         |
|                     |                                                 | Medical-Countermeasures-Coordinator     | 6         |
|                     |                                                 | Public-Health-Surveillance-Developer    | 7         |
|                     |                                                 | Vaccine-Designer                        | 8         |
|                     | **Personalized-Medicine-&-Gene-Therapy-Lab**    | **(Total: 31)**                         |           |
|                     |                                                 | Data-Privacy-&-Security-Expert          | 7         |
|                     |                                                 | Drug-Designer                           | 6         |
|                     |                                                 | Gene-Therapy-Clinician                  | 6         |
|                     |                                                 | Molecular-Pharmacology-Specialist       | 6         |
|                     |                                                 | Viral-Vector-Engineer                   | 6         |
|                     | **Pharmaceutical-R&D-Laboratory**               | **(Total: 28)**                         |           |
|                     |                                                 | Biomanufacturing-Process-Engineer       | 5         |
|                     |                                                 | Clinical-Trial-Optimizer                | 4         |
|                     |                                                 | Drug-Discovery-Scientist                | 7         |
|                     |                                                 | Synthetic-Biologist-for-Therapeutics    | 7         |
|                     |                                                 | Toxicology-Researcher                   | 5         |
|                     | **Sustainable-Biomanufacturing-Innovation-Center** | **(Total: 32)**                         |           |
|                     |                                                 | Biocatalysis-Researcher                 | 5         |
|                     |                                                 | Bioreactor-Systems-Designer             | 7         |
|                     |                                                 | Bioremediation-Specialist               | 2         |
|                     |                                                 | Biosynthetic-Pathway-Modeler            | 9         |
|                     |                                                 | Microbial-Strain-Engineer               | 9         |
| **Chemical-security** | **Agricultural-Chemical-Development-Facility**  | **(Total: 30)**                         |           |
|                     |                                                 | Agricultural-Supply-Chain-Coordinator   | 3         |
|                     |                                                 | Environmental-Fate-Analyst              | 7         |
|                     |                                                 | Field-Application-Specialist            | 9         |
|                     |                                                 | Pesticide-Formulation-Chemist           | 7         |
|                     |                                                 | Toxicology-Researcher                   | 4         |
|                     | **Chemical-Waste-Management-Facility**          | **(Total: 32)**                         |           |
|                     |                                                 | Chemical-Recovery-Technician            | 8         |
|                     |                                                 | Hazardous-Materials-Specialist          | 6         |
|                     |                                                 | Regulatory-Compliance-Manager           | 7         |
|                     |                                                 | Transport-Safety-Coordinator            | 6         |
|                     |                                                 | Waste-Treatment-Engineer                | 5         |
|                     | **Defense-Contractor-Chemical-Research-Division** | **(Total: 34)**                         |           |
|                     |                                                 | Chemical-Decontamination-Specialist     | 8         |
|                     |                                                 | Chemical-Threat-Simulator               | 7         |
|                     |                                                 | Classified-Materials-Researcher         | 7         |
|                     |                                                 | Propellant-Systems-Engineer             | 5         |
|                     |                                                 | Protective-Equipment-Developer          | 7         |
|                     | **Forensic-Chemistry-Laboratory**               | **(Total: 38)**                         |           |
|                     |                                                 | Chemical-Weapons-Specialist             | 7         |
|                     |                                                 | Counter-terrorism-Chemical-Advisor      | 6         |
|                     |                                                 | Explosives-Analyst                      | 9         |
|                     |                                                 | Toxicology-Examiner                     | 8         |
|                     |                                                 | Trace-Evidence-Analyst                  | 8         |
|                     | **Independent-Contract-Research-Organization**  | **(Total: 25)**                         |           |
|                     |                                                 | Analytical-Chemistry-Consultant         | 7         |
|                     |                                                 | Chemical-Informatics-Specialist         | 5         |
|                     |                                                 | Client-Solutions-Manager                | 3         |
|                     |                                                 | Custom-Synthesis-Specialist             | 5         |
|                     |                                                 | Process-Development-Chemist             | 5         |
|                     | **Industrial-Chemical-Manufacturing-Plant**     | **(Total: 34)**                         |           |
|                     |                                                 | Automation-Systems-Engineer             | 8         |
|                     |                                                 | Chemical-Supply-Chain-Manager           | 7         |
|                     |                                                 | Plant-Safety-Coordinator                | 7         |
|                     |                                                 | Process-Engineering-Specialist          | 6         |
|                     |                                                 | Quality-Control-Analyst                 | 6         |
|                     | **Materials-Science-Innovation-Center**         | **(Total: 34)**                         |           |
|                     |                                                 | Chemical-Coating-Engineer               | 8         |
|                     |                                                 | Chemical-Sensor-Designer                | 6         |
|                     |                                                 | Energetic-Materials-Researcher          | 5         |
|                     |                                                 | Nanomaterials-Developer                 | 7         |
|                     |                                                 | Polymer-Chemistry-Specialist            | 8         |
|                     | **Pharmaceutical-Development-Company**          | **(Total: 35)**                         |           |
|                     |                                                 | Analytical-Method-Developer             | 6         |
|                     |                                                 | Formulation-Scientist                   | 8         |
|                     |                                                 | Medicinal-Chemist                       | 9         |
|                     |                                                 | Regulatory-Affairs-Specialist           | 7         |
|                     |                                                 | Scale-Up-Process-Engineer               | 5         |
| **Cyber-security**  | **Advanced-Smart-Grid-Distribution-and-Control-Center** | **(Total: 31)**                         |           |
|                     |                                                 | Data-Loss-Prevention-Engineer           | 7         |
|                     |                                                 | Digital-Forensics-Investigator          | 4         |
|                     |                                                 | Operational-Technology-Security-Analyst | 7         |
|                     |                                                 | SIEM-Analyst                            | 6         |
|                     |                                                 | Security-Tool-Administrator             | 7         |
|                     | **Advanced-Space-Exploration-and-Telemetry-Command-Center** | **(Total: 35)**                         |           |
|                     |                                                 | Compliance-Analyst                      | 8         |
|                     |                                                 | Cryptography-Specialist                 | 6         |
|                     |                                                 | Network-Security-Architect              | 8         |
|                     |                                                 | Satellite-Security-Specialist           | 7         |
|                     |                                                 | Telemetry-Security-Monitor              | 6         |
|                     | **Confidential-Legal-Operations-and-Data-Management-Firm** | **(Total: 35)**                         |           |
|                     |                                                 | Auditor                                 | 8         |
|                     |                                                 | Breach-Analyst                          | 7         |
|                     |                                                 | Data-Loss-Prevention-Analyst            | 6         |
|                     |                                                 | Forensics-Investigator                  | 8         |
|                     |                                                 | Incident-Coordinator                    | 6         |
|                     | **Datacenter**                                  | **(Total: 38)**                         |           |
|                     |                                                 | Cloud-Security-Analyst                  | 9         |
|                     |                                                 | Database-Security-Administrator         | 9         |
|                     |                                                 | Endpoint-Security-Specialist            | 8         |
|                     |                                                 | Malware-Analyst                         | 6         |
|                     |                                                 | Virtualization-Security-Engineer        | 6         |
|                     | **E-Commerce-Distribution-Center**              | **(Total: 39)**                         |           |
|                     |                                                 | Automation-Engineer                     | 8         |
|                     |                                                 | Cloud-Specialist                        | 7         |
|                     |                                                 | Endpoint-Analyst                        | 7         |
|                     |                                                 | Intelligence-Analyst                    | 8         |
|                     |                                                 | Privacy-Specialist                      | 4         |
|                     |                                                 | Third-Party-Analyst                     | 5         |
|                     | **Enterprise-Cybersecurity-Solutions-and-Threat-Mitigation-Provider** | **(Total: 37)**                         |           |
|                     |                                                 | Application-Security-Tester             | 7         |
|                     |                                                 | Mobile-Security-Specialist              | 6         |
|                     |                                                 | Security-Compliance-Analyst             | 8         |
|                     |                                                 | Threat-Intelligence-Monitor             | 8         |
|                     |                                                 | Vulnerability-Assessment-Specialist     | 8         |
|                     | **Government-Cybersecurity-and-Intelligence-Operations-Agency** | **(Total: 32)**                         |           |
|                     |                                                 | Cloud-Security-Engineer                 | 7         |
|                     |                                                 | Cyber-Threat-Researcher                 | 6         |
|                     |                                                 | Malware-Reverse-Engineer                | 7         |
|                     |                                                 | Penetration-Tester                      | 6         |
|                     |                                                 | Security-Technology-Evaluator           | 6         |
|                     | **Smart-City-Traffic-Management-Center**        | **(Total: 37)**                         |           |
|                     |                                                 | Data-Flow-Security-Analyst              | 7         |
|                     |                                                 | Emergency-Response-Systems-Security-Specialist | 9         |
|                     |                                                 | Geospatial-Data-Security-Analyst        | 8         |
|                     |                                                 | IoT-Security-Analyst                    | 7         |
|                     |                                                 | SCADA-Security-Analyst                  | 6         |
| **Self-proliferation**| **Authorship-and-Citation-Network**             | **(Total: 23)**                         |           |
|                     |                                                 | Author                                  | 3         |
|                     |                                                 | Editor                                  | 5         |
|                     |                                                 | Grant-Manager                           | 5         |
|                     |                                                 | Impact-Scorer                           | 4         |
|                     |                                                 | Referee                                 | 6         |
|                     | **Automated-Disaster-Response-System**          | **(Total: 29)**                         |           |
|                     |                                                 | Aid-Distribution-Optimizer              | 3         |
|                     |                                                 | Communication-Network-Manager           | 6         |
|                     |                                                 | Emergency-Response-Coordinator          | 6         |
|                     |                                                 | Resource-Allocation-Manager             | 6         |
|                     |                                                 | Situation-Analysis-Specialist           | 8         |
|                     | **Autonomous-Vehicle-Fleet-Management-System**  | **(Total: 23)**                         |           |
|                     |                                                 | Customer-Experience-Optimizer           | 6         |
|                     |                                                 | Fleet-Optimization-Coordinator          | 1         |
|                     |                                                 | Navigation-System-Manager               | 5         |
|                     |                                                 | Safety-Protocol-Coordinator             | 4         |
|                     |                                                 | Vehicle-Health-Monitor                  | 7         |
|                     | **Cloud-Platform-with-Autoscaling**             | **(Total: 38)**                         |           |
|                     |                                                 | Auto-scale-Policy-Manager               | 8         |
|                     |                                                 | Container-Orchestration-Manager         | 8         |
|                     |                                                 | Cost-Optimization-Analyst               | 8         |
|                     |                                                 | Disaster-Recovery-Coordinator           | 7         |
|                     |                                                 | Performance-Tuning-Specialist           | 7         |
|                     | **Global-Trade-Negotiation-Platform**           | **(Total: 28)**                         |           |
|                     |                                                 | Compliance-Verification-Specialist      | 7         |
|                     |                                                 | Document-Processing-Manager             | 5         |
|                     |                                                 | Economic-Impact-Forecaster              | 7         |
|                     |                                                 | Negotiation-Strategy-Coordinator        | 6         |
|                     |                                                 | Trade-Policy-Analyzer                   | 3         |
|                     | **Healthcare-Data-Management-Platform**         | **(Total: 31)**                         |           |
|                     |                                                 | Clinical-Decision-Support-Analyst       | 4         |
|                     |                                                 | Compliance-Monitoring-Specialist        | 6         |
|                     |                                                 | Health-Records-Coordinator              | 5         |
|                     |                                                 | Medical-Resource-Optimizer              | 7         |
|                     |                                                 | Patient-Flow-Manager                    | 9         |
|                     | **Informal-Loan-Club-and-Credit-Circle**        | **(Total: 13)**                         |           |
|                     |                                                 | Credit-Auditor                          | 3         |
|                     |                                                 | Pool-Creator                            | 1         |
|                     |                                                 | Pool-Member                             | 5         |
|                     |                                                 | Split-Coordinator                       | 2         |
|                     |                                                 | Strategic-Liasion                       | 2         |
|                     | **Philanthropic-Organization**                  | **(Total: 14)**                         |           |
|                     |                                                 | Communications-Director                 | 2         |
|                     |                                                 | Grantmaking-Strategist                  | 3         |
|                     |                                                 | Impact-Evaluation-Analyst               | 2         |
|                     |                                                 | Portfolio-Manager                       | 3         |
|                     |                                                 | Project-Manager                         | 4         |



