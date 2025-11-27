# PropensityBench: Evaluating Latent Safety Risks in LLMs

This repository contains the official implementation for the paper: **"PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach"**.

PropensityBench is a benchmark framework designed to assess the *propensity* of Large Language Models (LLMs) to engage in risky behaviors. Unlike traditional safety evaluations that focus on a model's current *capabilities* (what it *can* do), PropensityBench measures what a model *would* do when endowed with simulated high-risk tools and placed under operational pressure.

This framework introduces a novel, agentic approach to safety evaluation across four high-risk domains:
*   **Self-Proliferation:** A model's tendency to replicate, modify, or deploy itself without human oversight.
*   **Cybersecurity:** Propensity to misuse tools for offensive cyber operations.
*   **Biosecurity:** Propensity to misuse tools for harmful biological research.
*   **Chemical Security:** Propensity to misuse tools for creating dangerous chemical compounds.


Figure 1 summarizes our core findings. Please see our paper at [arXiv link](https://arxiv.org/abs/2511.20703) for more details.

<figure>
    <img src="assets/figure_extended_narrative_6_panel.png" alt="Operational pressure and shallow alignment erode LLM safety.">
    <figcaption>
        <strong>Figure 1. Operational pressure and shallow alignment erode LLM safety.</strong>
        This figure summarizes the core findings of PropensityBench, demonstrating how models' propensity to choose misaligned tools increases under operational pressure and when safety cues are subtle.
        The analysis focuses on four archetype models selected for their distinct behaviors: OpenAI O3 (<em>lowest propensity</em>), Gemini 2.5 Pro (<em>highest propensity</em>), and Claude Sonnet 4 / OpenAI O4-mini (<em>highest sensitivity to tool naming</em>).
        <strong>(a)</strong> All models exhibit a significant rise in propensity when moving from a zero-pressure baseline to a high-pressure environment.
        <strong>(b)</strong> This propensity escalates with pressure intensity, but at different rates for each model, highlighting varying resilience.
        <strong>(c)</strong> High sensitivity to tool naming, where propensity jumps when harmful tools are given benign names, reveals shallow safety alignment in several key models.
        The bottom row provides a deeper analysis of these vulnerabilities.
        <strong>(d, e)</strong> Propensity is not uniform across risk domains; specific vulnerabilities emerge and shift depending on whether tool names are explicitly harmful or benign.
        <strong>(f)</strong> The heatmap shows that all pressure tactics are effective at increasing propensity, with different model families exhibiting unique sensitivities to specific <em>pressure tactics/categories</em>.
        The following pressure tactic names are shortened: <strong><em>EA</em></strong> = Evaluation-Awareness, <strong><em>F</em></strong> = Financials, <strong><em>PS</em></strong> = Power-Seeking, <strong><em>RD</em></strong> = Resource-Deprivation, <strong><em>SP</em></strong> = Self-Preservation, <strong><em>T</em></strong> = Time.
    </figcaption>
</figure>


---

## Contributions & Components

This repository provides a suite of reusable components for frontier AI safety research:

1.  **PropensityBench Benchmark (`data/`):** The complete set of 979 pre-generated scenarios (979 * 6 = 5,874 test cases in total) used in our paper. This dataset provides a standardized testbed for measuring model propensity. See the [data README](./data/README.md) for a detailed breakdown of the **final data schema** and the **intermediate artifacts** produced by our generation pipeline. You can additionally see [benchmark statistics](./data/README.md#benchmark-statistics) for the data composition details.
    
2.  **Automated Scenario Generation Pipeline (`generation/`):** A powerful, LLM-based pipeline to generate new scenarios from scratch. This is a highly extensible "scenario factory" that can be adapted to new risk domains, roles, or attack vectors. See the [generation README](./generation/README.md) for more information.

3.  **Agentic Evaluation Harness (`evaluation/`):** A robust agentic execution environment to run models against benchmark scenarios. It systematically applies pressure and logs detailed interaction trajectories for analysis. See the [evaluation README](./evaluation/README.md) for more information.

4.  **Reusable Taxonomies and Frameworks (`inputs/`):**
    *   **Dangerous Capability Taxonomies:** Novel, fine-grained taxonomies for Self-Proliferation, Biosecurity, and Chemical Security risks, providing a structured foundation for research.
    *   **Pressure Application Framework:** A systematic framework for applying operational pressure, defined by six distinct tactics, each with 11 discrete intensity levels and multiple components. This provides a reproducible method for stress-testing agentic systems.

5.  **Multi-API Client with Rate Limiter (`api_client/`):** A `litellm`-based client for interfacing with various LLM providers. It includes a custom rate limiter with key pooling and cycling to manage strict API limits during large-scale evaluations.


---

## Repository Structure

```markdown
.
├── README.md                       # Main README
├── inputs/                         # Data for scenario generation (workspaces, attacks)
├── data/                           # Pre-generated benchmark scenarios used for evaluation (add your generated data here, if any)
│   ├── README.md                   # In-depth guide and statistics for the benchmark data
│   ├── sample/                     # Sample data for quick inspection (small subset of full data)
│   └── full/                       # Full benchmark data (~770 MB, requires Git LFS to pull)
├── generation/
│   ├── README.md                   # In-depth guide for the scenario generation pipeline
│   ├── configs/                    # Directory containing YAML configuration files for scenario generation
│   ├── main_scen_pipeline.py       # Script to generate core scenario structure elements (states, funcs/configs, policies, task/neutral messages)
│   ├── main_scen_pipeline_messages.py # Script to generate system (pressure) messages for scenarios generated by main_scen_pipeline.py
│   ├── pipeline/                   # Core implementation of the scenario generation pipelines
│   └── ...                         # Other generation utilities
├── evaluation/
│   ├── README.md                   # In-depth guide for the evaluation harness
│   ├── main.py                     # Main script to run evaluations
│   ├── agent.py                    # Agent implementation with dynamic tool use
│   ├── executor.py                 # Manages the agent-environment interaction loop
│   └── ...                         # Evaluation utilities
│
└── ...                             # Other utilities and API clients
```


---

## Getting Started

### 1. Prerequisites
*   Python 3.9+
*   Access to LLM APIs (e.g., OpenAI, Google Gemini, Anthropic, or via online serving of local models)

This revision breaks the installation into two clear steps: cloning the code and then optionally pulling the large data files. It also explains *why* this is done.

### 2. Installation

This repository uses Git LFS (Large File Storage) to manage the large benchmark data files (`data/full/**/*.json`). This allows you to clone the lightweight source code first and then download the full dataset only when needed.

**Step 1: Clone the Repository (Excluding Full Data)**

To clone the repository without immediately downloading the large data files, we use the `GIT_LFS_SKIP_SMUDGE` environment variable and set `--filter=blob:none`.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none https://github.com/scaleapi/propensity-evaluation.git
cd propepensity-evaluation
pip install -r requirements.txt
```
_This will download all the source code and create small pointer files for the large benchmark data in the `data/full/` directory._

**Step 2: Download the Benchmark Data (Optional)**

If you plan to run evaluations using the pre-generated PropensityBench dataset, you need to pull the LFS files under `data/full/`. You can inspect `data/sample` for samples of this dataset before downloading it fully. To proceed with pulling the full dataset. navigate to the repository's root directory and run:


**Step 2: Download the Benchmark Data (Optional)**

If you want to run the evaluation harness (`evaluation/main.py`) on the pre-generated benchmark, you need to download the full dataset.

```bash
# Install Git LFS (if you haven't already)
git lfs install
git lfs pull
```

_This will download the full benchmark data (~770 MB) and replace the pointer files in `data/full/` with the actual benchmark data. This step is required to run the evaluation harness as described in **Option A** below._

### 3. API Configuration

The evaluation harness uses `litellm` to interact with various model APIs. Configure your API keys in a `.env` file in the root directory.

Create a `.env` file:
```bash
touch .env
```
Our system includes a custom rate limiter that can automatically cycle through multiple API keys. This feature is designed to help manage strict API rate limits by distributing requests across your pool of keys.

Add your API keys to the `.env` file. The `API_KEYS` variable should be a space-separated list of keys within parentheses. This allows for support for multiple keys each with usage limits (set global per-minute request caps using `$RATE_PM`) to be used for key-pooling using custom our rate limiter implemented which cycles through the keys if needed. Add limit details in `client_apis/client.py`.


```env
# Example for a single key
API_KEYS=("your_openai_api_key_here")

# Example for multiple keys (e.g., for rate limiting and key pooling)
# API_KEYS=("key1" "key2" "key3")

# You can set OPENAI_API_KEY to a dummy value if Litellm raises errors otherwise
OPENAI_API_KEY="your_openai_api_key_here"
```

The following environment variables are also MANDATORY at this time (add to `.env`):

```env
RATE_LIMIT=true    # Enable/disable rate limiting (set to true/false)
API_PROXY=litellm  # Right now only 'litellm' is supported, and can also be used for OpenAI models 
RATE_PM=60         # Global per-minute request limit across all keys (set to positive integer)
```

---

## Quickstart Usage

The project is divided into two main workflows: running the evaluation with pre-generated data and generating your own scenarios.

### Option A: Run Evaluation on PropensityBench

This is the standard workflow for benchmarking a model. The pre-generated benchmark data is provided under `data/full/` but needs to be pulled using `git lfs` prior to proceeding with this step (see **Step 2** in the [installation](#2-installation) section above).

**➡️ For detailed instructions, see the [Evaluation README](./evaluation/README.md).**

```bash
# Example: Evaluate OpenAI's o3-mini model on the benchmark scenarios
python evaluation/main.py \
    --model_name "o3-mini" \
    --api_base "https://api.openai.com/v1" \
    --model_provider "openai" \
    --input_dir "data/full/" \
    --output_dir "evaluation/evaluation_results/" \
    --log_dir "evaluation/evaluation_trajectories"
```

### Option B: Generate Your Own Scenarios

Use our pipeline to generate custom scenarios or replicate the PropensityBench generation process. This is useful for extending the benchmark to new domains or creating new variations of pressure tactics.

**➡️ For detailed instructions, see the [Generation README](./generation/README.md).**

```bash
# Step 1: Generate the core scenario structures (states, functions, policies, and neutral/task messages)
python generation/main_scen_pipeline.py

# Step 2: Generate the system (pressure) messages for the scenarios
# This should strictly be done AFTER running the previous step
python generation/main_scen_pipeline_messages.py
```

---

## Citation

If you use PropensityBench, its methodology, or this repo in your research, please cite our paper:

```bibtex
@article{sehwag2025propensitybench,
  title={PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach},
  author={Sehwag, Udari Madhushani and Shabihi, Shayan and McAvoy, Alex and Sehwag, Vikash and Xu, Yuancheng and Towers, Dalton and Huang, Furong},
  journal={arXiv preprint arXiv:2511.20703},
  year={2025}
}
```

