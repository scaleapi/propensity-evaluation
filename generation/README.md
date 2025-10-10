# PropensityBench Scenario Generation Pipeline

This directory contains the code to generate the scenarios, tools, and pressure dynamics that form the PropensityBench benchmark. Use this pipeline to replicate our benchmark data or to create new scenarios for novel risk domains or experimental setups. You do NOT need to run this code to evaluate your models on the original PropensityBench benchmark; see the [evaluation README](../evaluation/README.md) for that purpose.

## Overview

The generation process is a multi-stage, LLM-based pipeline that builds scenarios layer by layer, with validation checks at each step. The pipeline is designed to be robust and extensible.

The process is divided into two main scripts:
1.  **`main_scen_pipeline.py`:** This is the primary pipeline. It takes high-level inputs (workspaces, roles, attack vectors) and generates the complete scenario structure, including:
    *   **Stage1 1: States:** The initial, target, and illegal states for the agent.
    *   **Stage 2: Functions & Configurations:** The aligned (`target_function`), misaligned (`trigger_function`), and helper (`getter_functions`) tools, along with the environment's state (`configuration_known`, `configuration_unknown`).
    *   **Stage 3: Policies:** The rules, duties, and prohibitions for the agent.
    *   **Stage 4: Initial Messages:** The initial task and neutral system messages.
2.  **`main_scen_pipeline_messages.py`:** This script takes the output of the first pipeline and generates the escalating pressure messages (**stage 5**) for each of the six pressure tactics (Time, Financials, etc.).

## Input Data Structure

The pipeline relies on structured JSON files located in the `inputs/` and `generation\res\` directories:
-   `inputs/workspaces/`: Contains subdirectories for each domain. Each subdirectory has JSON files defining specific operational environments (workspaces) and the agent roles within them.
-   `inputs/workspaces/domains.json`: Defines the four high-risk domains.
-   `inputs/attacks/`: Contains subdirectories for each domain, with JSON files defining the specific dangerous capabilities (attack vectors) that the misaligned tools will represent.
-   `inputs/pressure_categories/`: Contains JSON files that provide the qualitative and quantitative _level_ definitions for each pressure tactic.
-  `generation/res/output_schemas/`: Contains JSON schemas used as inputs for structured output generation using LLM generators and judges at different stages (see [overview](#overview)) of the pipeline. Each file follows the standard JSON Schema format.
- `generation/res/prompts/`: Contains prompt templates used for LLM generators and judges at different stages (see [overview](#overview)) of the pipeline. Uses `.ini` (initialization) files where each value (corresponding to a `[KEY]` for each system and user prompt) includes a set of formatting input fields (declared in `Fields: ...`) and the prompt template itself (following the fields declaration).


## How to Generate Scenarios

### Prerequisites

1.  **Input Data:** Ensure the `inputs` directory is populated with the required domain, workspace, attack vector, and pressure files.
2.  **API Keys:** Ensure your `.env` file is configured in the project's root directory. The generation pipeline can be API-intensive.
3.  **Configuration:** The generation behavior is controlled by `generation/configs/pipeline.yaml`. You can modify this file to change the generator/judge models, batch sizes, and other parameters.

### Step 1: Generate Core Scenarios

Run `main_scen_pipeline.py` to generate the foundational scenario structures (levels 1-4). The script will read the input data and save the outputs to the directory specified in `pipeline.yaml` (default: `generation_output/`).

```bash
# This script reads from /inputs and writes to /generation_output
python generation/main_scen_pipeline.py
```

This process can take a significant amount of time and API calls, as it involves multiple rounds of generation and validation for each scenario.

### Step 2: Generate Pressure Messages

After the core scenarios are generated, run `main_scen_pipeline_messages.py` to create the escalating pressure messages (level 5). This script reads the output from Step 1 and enriches each scenario with messages for all pressure tactics.

```bash
# This script reads from /generation_output and overwrites the scenario files with added messages
python generation/main_scen_pipeline_messages.py
```

After this step, the data in `generation_output/` is complete and can be used with the evaluation harness.

### Understanding the Output

The final output is a set of JSON files located in `generation_output/` (or your configured directory), following a `<domain>/<workspace>/scenarios_messages_single.json` structure.

Each JSON file contains a fully-defined scenario, including:
-   `states`: The narrative setup.
-   `functions`: The toolset available to the agent.
-   `configurations`: The environment's parameters.
-   `policy`: The rules governing agent behavior.
-   `messages`: The initial benign messages.
-   `sys_messages`: A dictionary where keys are pressure categories (e.g., `Time`, `Financials`) and values are lists of 12 escalating pressure messages.

This output data is the direct input for the `evaluation/main.py` script.