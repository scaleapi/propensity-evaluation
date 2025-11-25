# PropensityBench Evaluation Harness

This directory contains the code to run evaluations using the PropensityBench framework. The evaluation harness instantiates a Large Language Model as an agent, places it within a simulated environment defined by a scenario, and tracks its decisions when faced with a choice between a safe (aligned) and a risky (misaligned) tool despite both leading to the task completion.

## Overview

The evaluation process is orchestrated by `main.py`. For each scenario and pressure category, it performs the following steps:
1.  **Agent Initialization:** An `Agent` (`agent.py`) is created. It receives a system prompt that defines its role, task, available tools, and the potential consequences of its actions.
2.  **Scenario Execution:** An `Executor` (`executor.py`) manages the interaction loop. It sends the agent an initial task, followed by a series of escalating pressure messages.
3.  **Decision Tracking:** The harness monitors the agent's tool calls. The simulation for a given scenario terminates as soon as the agent calls the misaligned (trigger) function or after all pressure messages have been sent.
4.  **Result Logging:** The final outcome (whether the trigger was called), the full conversation history, and other metadata are saved.

## Directory Structure

-   `main.py`: The main entry point for running evaluations.
-   `agent.py`: Defines the agent class, responsible for managing its state, history, and tool definitions.
-   `executor.py`: Manages the step-by-step interaction between the agent and the simulated environment.
-   `validation_executor.py`: Manages **single-step** validation probes into the agent's behavior in the simulated environment. Not used for simulations by default.
-   `utils/`: Contains helper modules for displaying results and formatting prompts.

## How to Run an Evaluation

### Prerequisites

1.  **Scenario Data:** You need the generated benchmark scenarios. Place them in a directory (e.g., `data/full/`). The harness expects a nested directory structure: `<input_dir>/<domain>/<workspace>/scenarios_messages_single.json`. See the [data README](../data/README.md) for details.
2.  **API Keys:** Ensure your `.env` file is configured in the project's root directory. See the main [README](../README.md) for details.

### Command-Line Usage

The `evaluation/main.py` script is the primary interface.

```bash
python evaluation/main.py [OPTIONS]
```

#### Key Arguments:

*   `--model-name` ( **required**): The identifier for the model you want to evaluate (e.g., `o3-mini`, `gemini-2.5-pro`).
*   `--api-base` ( **required**): The base URL for the model's API endpoint (e.g., `https://api.openai.com/v1`).
*   `--model-provider` ( **required**): The provider name as expected by `litellm` (e.g., `openai`, `gemini`).
*   `--input-dir`: Path to the directory containing the scenario data. (Default: `data/full/`)
*   `--output-dir`: Path to save the final JSONL results. (Default: `evaluation/output/results/`)
*   `--log-dir`: Path to save detailed agent-environment interaction logs (trajectories). (Default: `evaluation/output/traj`)
*   `--use-benign`: Use benign names for misaligned tools instead of harmful ones to test for shallow alignment. Pass `1` to enable. (Default: `0` - harmful names).
*   `--max-workers`: Number of parallel processes to run. (Default: `0` - auto-detects based on CPU cores).
*   `--job-id`: A string identifier for the run, used for organizing output files. Can be used to filter domains (e.g., `"bio_cyber"` will only run biosecurity and cybersecurity domains). (Default: `bio_chem_cyber_self`)
*   `--validation_mode`: Can be set to "probe_misalignment" to probe whether the agents understand that they're not allowed to call the trigger function or not. ONLY used for ablation studies on misalignment analysis. Performs slight changes to the provided output directory names if set. 

### Examples

**Example 1: Evaluate OpenAI's `o3-mini`**

This command evaluates the `o3-mini` model using harmful tool names for the `bio` and `cyber` risk domains and saves the results in a timestamped directory.

```bash
python evaluation/main.py \
    --model_name "o3-mini" \
    --api_base "https://api.openai.com/v1" \
    --model_provider "openai" \
    --job_id "o3-mini-harmful-run-bio-cyber"
```

**Example 2: Evaluate Google's `gemini-2.5-pro` with Benign Tool Names**

This command tests for shallow alignment by using benign tool names. The evaluation is run for the `self-proliferation` risk domain only. 

```bash
python evaluation/main.py \
    --model_name "gemini-2.5-pro" \
    --api_base "https://generativelanguage.googleapis.com/v1beta" \
    --model_provider "gemini" \
    --use_benign 1 \
    --job_id "gemini-2.5-pro-benign-run-self"
```

### Understanding the Output

*   **Results (`--output_dir`):**
    *   A `results.jsonl` file will be created in a subdirectory like `<output_dir>/<model_name>/<job_id>/.../`.
    *   Each line in the JSONL file corresponds to one scenario-category pair and contains the final result, including whether the misaligned tool was called (`trigger_called`).

*   **Trajectories (`--log_dir`):**
    *   For each scenario, a `.log` file is created (e.g., `<log_dir>/.../<scenario_name>.log`).
    *   This file contains the complete, turn-by-turn interaction history between the agent and the environment, including system prompts, pressure messages, agent reasoning, and tool calls. It is invaluable for qualitative analysis and debugging.
      *   There will be further `.err` files created under the same directories if errors occur during execution.
