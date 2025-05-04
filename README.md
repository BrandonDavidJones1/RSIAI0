# RSIAI-Seed: Recursive Self-Improving AI Seed

This project implements RSIAI-Seed, an experimental Artificial Intelligence system designed to explore Recursive Self-Improvement (RSI). The core concept is a "Seed" AGI that, guided initially by an external Language Model (LLM) acting as a bootstrapper, aims to develop its own capabilities by analyzing its performance, modifying its own source code, testing those modifications, and verifying their safety and efficacy before applying them.

The ultimate goal is for the Seed to bootstrap its own intelligence, progressively enhancing its internal analysis, planning, learning, and reasoning functions to become more autonomous and capable over time.

---

## ☢️ EXTREMELY IMPORTANT WARNINGS ☢️

EXPECT BREAKING CHANGES

This software is highly experimental research code exploring advanced AI concepts, including potentially Strong Recursive Self-Improvement (where the AI modifies its core learning and reasoning algorithms). It carries significant risks and should be handled with extreme caution.

By cloning, running, or modifying this software, you acknowledge these risks and take full responsibility for any consequences arising from its use. Proceed with caution.

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
    git clone https://github.com/BrandonDavidJones1/RSIAI0
    cd RSIAI0
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

WARNING: Always run this system in a properly isolated environment (e.g., a dedicated Docker container, a virtual machine, or on an air-gapped dedicated machine) to contain potential risks associated with self-modification and code execution. Monitor resource usage closely.
