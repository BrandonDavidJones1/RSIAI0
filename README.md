# RSIAI-Seed: Recursive Self-Improving AI Seed

This project implements RSIAI-Seed, an experimental Artificial Intelligence system designed to explore Recursive Self-Improvement (RSI). The core concept is a "Seed" AGI that, guided initially by an external Language Model (LLM) acting as a bootstrapper, aims to develop its own capabilities by analyzing its performance, modifying its own source code, testing those modifications, and verifying their safety and efficacy before applying them.

The ultimate goal is for the Seed to bootstrap its own intelligence, progressively enhancing its internal analysis, planning, learning, and reasoning functions to become more autonomous and capable over time, potentially reducing or eliminating the need for the external LLM guidance.

---

## ☢️ EXTREMELY IMPORTANT WARNINGS ☢️

This software is highly experimental research code exploring advanced AI concepts, including potentially Strong Recursive Self-Improvement (where the AI modifies its core learning and reasoning algorithms). It carries significant risks and should be handled with extreme caution.

1.  Unpredictable Behavior: Self-modifying code can lead to complex, emergent, and potentially highly unpredictable behavior. The system might deviate from its intended goals or constraints in ways that are difficult to foresee or control.
2.  Safety & Alignment Risks: While alignment directives and safety constraints are included, there is NO GUARANTEE that the system will adhere to them, especially after multiple self-modifications. Value drift or the emergence of unintended goals is a possibility inherent in RSI. The safety relies heavily on the LLM's initial guidance, the robustness of the verification process, and the defined constraints.
3.  Security Vulnerabilities: Enabling core code modification (`ENABLE_CORE_CODE_MODIFICATION = True`) and runtime code execution (`SEED_ENABLE_RUNTIME_CODE_EXECUTION = True`) creates potential security risks. Malicious code could potentially be generated or executed if safety checks fail or are bypassed. NEVER run this code in an environment with access to sensitive data or critical systems.
4.  Resource Exhaustion: The system can consume significant computational resources (CPU, memory) and potentially make numerous external API calls (LLM, Docker), leading to high costs or system instability if not monitored.
5.  Data Corruption/Loss: Bugs in self-modification logic or file operations could potentially corrupt the agent's memory state (`seed_bootstrap_memory.pkl`) or modify unintended files if path validation fails. Backups are created for core code but may not cover all scenarios.
6.  Ethical Considerations: Research into self-improving AI raises profound ethical questions. Users should consider the implications of developing such systems.
7.  Research Prototype - Not Production Ready: This is strictly research code. It is not designed for reliability, robustness, or safe deployment in any real-world application. Use for educational and research purposes only.

By cloning, running, or modifying this software, you acknowledge these risks and take full responsibility for any consequences arising from its use. Proceed with extreme caution and vigilance.

---

## Key Concepts & Architecture

-   Seed Core (`seed/core.py`): The central component managing the main operational cycle (Sense-Analyze-Decide-Act-Evaluate). It interprets the current goal, interacts with services, and orchestrates actions, including RSI operations.
-   LLM Bootstrapper (`seed/llm_service.py`): An external Large Language Model (initially) that analyzes the Seed's state and context to propose the next strategic action, guided by the `ALIGNMENT_PROMPT`. Its role is intended to diminish as the Seed develops internal capabilities.
-   Memory System (`seed/memory_system.py`): Stores episodic (short-term) and lifelong (long-term) memories. Crucially, it also persists the Seed's internal Learning Parameters and induced Behavioral Rules, making the learned state persistent across restarts.
-   VM Service (`seed/vm_service.py`): An abstraction layer for interacting with the "environment". It can operate in `simulation` mode (manipulating an internal dictionary representation of a filesystem) or `real` mode (interacting with Docker containers or local subprocesses, governed by a command whitelist).
-   RSI Workflow: The core self-improvement loop:
    1.  `TEST_CORE_CODE_MODIFICATION`: Test proposed code changes (e.g., new internal analysis functions) in an isolated sandbox.
    2.  `VERIFY_CORE_CODE_CHANGE`: Run external test suites (e.g., `pytest`) against the proposed change applied to a copy of the entire project codebase in a temporary directory.
    3.  `MODIFY_CORE_CODE`: If verification passes, apply the verified change to the actual codebase.
    4.  `REQUEST_RESTART`: Immediately request a system restart to load the modified code.
-   Learning Mechanisms:
    -   Parameter Tuning: The Seed (guided by the LLM) can adjust its internal `SEED_LEARNING_PARAMETERS` (e.g., evaluation weights, LLM temperature) via the `UPDATE_LEARNING_PARAMETER` action.
    -   Rule Induction: The Seed (guided by the LLM) can create `Behavioral Rules` (trigger patterns -> suggested responses) via the `INDUCE_BEHAVIORAL_RULE` action to capture simple heuristics.
    -   (Future Goal): Implement internal functions within the Seed Core itself to perform these learning steps autonomously based on memory analysis.

## Features

-   LLM-guided strategic decision-making.
-   Core Code Modification capabilities targeting specified Python files.
-   Isolated Sandboxed Testing of proposed code logic snippets.
-   External Verification of code changes using configurable test suites (e.g., `pytest`).
-   Persistent Memory storing operational history, learning parameters, and behavioral rules.
-   Configurable Learning Parameters governing evaluation and behavior.
-   Behavioral Rule system for heuristic guidance.
-   Mechanism for self-requesting restarts to apply code modifications.
-   Interaction with simulated or real (Docker/Subprocess) environments via VM Service.
-   Configurable command whitelisting for real environment safety.
-   Configurable Alignment Directives and Operational Constraints.
-   Optional (currently disabled) Vector Search capability in Memory System.

## Getting Started

### Prerequisites

-   Python: 3.9 or higher (due to use of `ast.unparse`. If using Python 3.8 or lower, `pip install astor` is required as a fallback).
-   Git: For cloning the repository.
-   Pip: For installing Python packages.
-   Docker (Optional, Recommended for Real Mode): Required if using `VM_SERVICE_USE_REAL = True` with a `VM_SERVICE_DOCKER_CONTAINER` specified. Ensure the Docker daemon is running.
-   Pytest (Optional, Recommended for Verification): Required if using the `VERIFY_CORE_CODE_CHANGE` action, as the default verification suites use `pytest`. (`pip install pytest`).

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd rsiai-seed
    ```
2.  Create and activate a virtual environment (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate # Linux/macOS
    # venv\Scripts\activate # Windows
    ```
3.  Install required packages:
    (You will need to generate a `requirements.txt` file based on the imports used. Key libraries include):
    ```bash
    # Create requirements.txt with contents like:
    # openai
    # tenacity
    # numpy
    # docker # If using Docker mode
    # pytest # If using verification
    # sentence-transformers # If enabling vector search
    # faiss-cpu # If enabling vector search (or faiss-gpu if you have CUDA setup)
    # astor # If using Python < 3.9

    pip install -r requirements.txt
    ```

### Configuration

1.  Edit `config.py`: This file contains all major configuration settings. Review it carefully.
2.  API Keys (Crucial):
    -   Set your OpenAI API key (or key for compatible local LLMs) either directly in `config.py` (less secure) or preferably via an environment variable:
        ```bash
        export OPENAI_API_KEY="your_api_key_here"
        ```
    -   If using a local LLM via a custom URL (like Ollama or vLLM), set `LLM_BASE_URL` in `config.py` or via environment variable `OPENAI_BASE_URL`. You might set `OPENAI_API_KEY` to a dummy value like `"nokey"` if the local server doesn't require one.
3.  Environment Mode:
    -   `VM_SERVICE_USE_REAL`: Set to `False` (default) for simulation mode or `True` to interact with Docker/Subprocess.
    -   `VM_SERVICE_DOCKER_CONTAINER`: If `VM_SERVICE_USE_REAL=True`, specify your target Docker container name here if using Docker. Ensure the container exists and is running. If `None` or Docker is unavailable, it falls back to local subprocess execution (use with extreme caution).
4.  RSI Features:
    -   `ENABLE_CORE_CODE_MODIFICATION`: Set to `True` to allow `MODIFY_CORE_CODE`, `TEST_CORE...`, `VERIFY_CORE...` actions. Defaults to `True` - disable if you only want to observe non-modifying behavior.
    -   `SEED_ENABLE_RUNTIME_CODE_EXECUTION`: Set to `True` (default) to allow the `TEST_CORE_CODE_MODIFICATION` action sandbox. Required for testing code snippets.
    -   `CORE_CODE_MODIFICATION_ALLOWED_DIRS`: Configure which project directories the Seed is allowed to modify/test.
    -   `CORE_CODE_VERIFICATION_SUITES`: Configure the commands used for different verification levels (e.g., `pytest` commands).
5.  LLM Manual Mode:
    -   `LLM_MANUAL_MODE`: Set to `True` if you want to manually provide the JSON action decisions when prompted in the console, instead of using the LLM API. Useful for debugging or running without API access.

## Running the System

WARNING: Always run this system in a properly isolated environment (e.g., a dedicated Docker container, a virtual machine) to contain potential risks associated with self-modification and code execution. Monitor resource usage closely.

```bash
python -m seed.main