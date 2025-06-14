# RSIAI0-Seed: Recursive Self-Improving AI Seed

**An Experimental LLM-based "Darwinian Gödel Machine" for Exploring Recursive Self-Improvement.**

RSIAI0-Seed is an open-source project designed to explore Artificial General Intelligence (AGI) through Recursive Self-Improvement (RSI). The core is a "Seed" AI that, initially guided by an external Language Model (LLM), aims to autonomously enhance its own capabilities. It achieves this by analyzing its performance, strategically modifying its scaffolding, testing those modifications, and verifying their efficacy before live application.

The vision is for the Seed to bootstrap its intelligence, progressively refining its internal analysis, planning, learning, and reasoning functions to become increasingly autonomous and capable.
---

## ☢️ DANGER: EXTREME CAUTION REQUIRED ☢️

**UNRESTRICTED SYSTEM ACCESS BY DEFAULT IN "REAL" MODE:**
*   The default configuration (`config.py`) sets `VM_SERVICE_USE_REAL = True` and `VM_SERVICE_ALLOWED_REAL_COMMANDS = "__ALLOW_ALL_COMMANDS__"`.
*   **THIS MEANS THE AI HAS UNRESTRICTED SHELL ACCESS TO THE OPERATING ENVIRONMENT (HOST OR DOCKER CONTAINER) WHEN THE VM SERVICE IS IN "REAL" MODE.**
*   This is intended for advanced research scenarios and carries **MAXIMUM RISK**.
*   **ALWAYS run in a SEVERELY ISOLATED environment (dedicated, sandboxed VM or air-gapped machine).**
*   Consider changing `VM_SERVICE_USE_REAL = False` in `config.py` for initial exploration or if unrestricted access is not needed.

**EXPERIMENTAL RESEARCH CODE:**
*   This software explores advanced AI concepts, including Strong Recursive Self-Improvement (AI modifying its core learning/reasoning algorithms).
*   It carries **significant inherent risks**, including unintended behaviors, rapid resource consumption, and potential for harmful actions if not properly contained.

**USER RESPONSIBILITY:**
*   By cloning, running, or modifying this software, you **acknowledge these profound risks** and **take full responsibility** for any and all consequences arising from its use.
*   The developers are not liable for any misuse or damage caused by this software.
*   **PROCEED WITH EXTREME CAUTION, DILIGENCE, AND ETHICAL CONSIDERATION.**

---

## Core Concepts

*   **Recursive Self-Improvement (RSI):** The AI's ability to iteratively improve its own intelligence and performance.
*   **Seed AI:** The initial, foundational AI system that begins the RSI process.
*   **LLM Bootstrapper:** An external Large Language Model (e.g., GPT-4) that provides initial guidance, generates code modifications, and helps reason about improvements, especially in the early stages.
*   **Self-Modification Cycle:** The core loop of:
    1.  **Analysis:** Evaluating performance, identifying areas for improvement.
    2.  **Hypothesis:** Proposing changes (e.g., code modifications, parameter tuning).
    3.  **Implementation:** Generating and applying these changes.
    4.  **Testing:** Running tests to ensure new code works as expected.
    5.  **Verification:** Running broader integration tests to confirm safety and efficacy before permanent application.
*   **Darwinian Gödel Machine:** When `ENABLE_DARWINIAN_MODE` is active, the system functions akin to a Darwinian Gödel Machine, where a population of AI "variants" evolve. Each variant can modify its own code (its "genome") and is evaluated for fitness, with successful variants reproducing.

---

## Key Features

*   **LLM-Guided Self-Modification:** Leverages an LLM for sophisticated reasoning and code generation to improve the Seed AI.
*   **Source Code Analysis & Modification:** The AI can read, understand, and rewrite parts of its own Python codebase (primarily within the `seed/` directory).
*   **Automated Testing & Verification:** Includes mechanisms to test proposed code changes in a sandboxed environment and verify them using test suites (e.g., `pytest`) before committing.
*   **Configurable Learning Parameters:** Internal parameters influencing learning, decision-making, and LLM interaction can be dynamically adjusted by the AI.
*   **Virtual Machine (VM) Service:**
    *   **Simulation Mode:** Executes commands in a simulated, safe environment.
    *   **Real Mode:** Can execute commands on the host system or within a specified Docker container. **EXTREMELY DANGEROUS by default due to unrestricted command execution.**
*   **Memory System:** Features episodic and lifelong memory, with optional vector search capabilities for semantic recall.
*   **Darwinian Mode (Genetic Algorithm):**
    *   Disabled by default (`ENABLE_DARWINIAN_MODE = False`).
    *   When enabled, allows for a population of AI variants to evolve, test self-modifications, and compete based on fitness. This mode is highly resource-intensive and experimental.

---

## Getting Started

### Prerequisites

*   **Python:** 3.9 or higher (due to `ast.unparse`).
    *   For Python 3.8 or lower: `pip install astor` is required as a fallback, and you may need to adjust code if other 3.9+ features are used elsewhere.
*   **Git:** For cloning the repository.
*   **Pip:** For installing Python packages.
*   **Docker (Strongly Recommended for "Real Mode"):** Required if using `VM_SERVICE_USE_REAL = True` with a `VM_SERVICE_DOCKER_CONTAINER` specified. Ensure the Docker daemon is running. Even if running directly on the host in "Real Mode," Docker provides an important layer of isolation.
*   **Pytest (Strongly Recommended for Verification):** Required if using the `VERIFY_CORE_CODE_CHANGE` action, as the default verification suites use `pytest`. Install via `pip install pytest`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BrandonDavidJones1/RSIAI0
    cd RSIAI0
    ```
2.  **Create and activate a virtual environment (Highly Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate    # Windows
    ```
3.  **Install required packages:**
    Create a `requirements.txt` file in the project root with the following content. Uncomment optional packages as needed.
    ```txt
    # Core requirements
    openai
    tenacity
    numpy

    # Optional, but highly recommended for certain features:
    # For Docker integration with VM Service
    # docker

    # For core code verification capabilities
    # pytest

    # For optional vector search in memory system
    # sentence-transformers
    # faiss-cpu # or faiss-gpu if you have a CUDA-enabled GPU and environment

    # Fallback for Python < 3.9 ast.unparse functionality
    # astor
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **CRITICAL: Review `config.py` (`seed/config.py`)**
    *   This file contains ALL major configuration settings. Understand them thoroughly before running the system.
    *   **Pay special attention to:**
        *   `VM_SERVICE_USE_REAL` (default: `True`)
        *   `VM_SERVICE_ALLOWED_REAL_COMMANDS` (default: `VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE` - grants unrestricted shell access)
        *   `ENABLE_CORE_CODE_MODIFICATION` (default: `True`)
        *   `ENABLE_DARWINIAN_MODE` (default: `False`)
        *   `LLM_API_KEY` and `LLM_BASE_URL`

2.  **API Keys (Crucial):**
    *   Set your OpenAI API key (or key for a compatible local LLM).
    *   **Recommended:** Use an environment variable:
        ```bash
        export OPENAI_API_KEY="your_api_key_here"
        ```
    *   Alternatively (less secure), you can set `LLM_API_KEY` directly in `seed/config.py`.
    *   If using a local LLM (e.g., Ollama, vLLM) via a custom URL, set `LLM_BASE_URL` in `seed/config.py` or via the environment variable `OPENAI_BASE_URL`. You might set `OPENAI_API_KEY` to a dummy value like `"nokey"` if the local server doesn't require authentication.

3.  **Strongly Consider Initial Safety Modifications to `config.py`:**
    *   For initial runs and testing, it is **highly advisable** to change:
        *   `VM_SERVICE_USE_REAL = False` (to use simulation mode).
    *   Only enable `VM_SERVICE_USE_REAL = True` once you fully understand the implications and have a robust, isolated environment.

---

## Running the System

**🔴 WARNING: POTENTIALLY DANGEROUS OPERATION 🔴**

*   **ISOLATION IS PARAMOUNT:** Always run this system in a **properly and securely isolated environment** (e.g., a dedicated Docker container with strict resource limits, a hardened virtual machine disconnected from sensitive networks, or ideally, an air-gapped dedicated machine).
*   **MONITOR CLOSELY:** Continuously monitor resource usage (CPU, memory, disk, network), file system changes, and process activity.
*   **RISK OF SELF-MODIFICATION:** The AI is designed to modify its own code. This can lead to unpredictable behavior. Ensure you have backups and understand how to revert changes.

1.  **Navigate to the project directory:**
    ```bash
    cd RSIAI0
    ```
2.  **Ensure your virtual environment is activated if you created one.**
3.  **Start the system (typically by running the main orchestrator script):**
    ```bash
    python main.py  # Or the primary script that launches the Seed AI
    ```

### Modes of Operation

*   **Standard Run (Default):**
    *   A single instance of the Seed AI runs for a configured number of iterations (or until a goal is met/aborted).
    *   Focuses on self-improvement of its single codebase.
*   **Darwinian Mode (Genetic Algorithm):**
    *   Enable by setting `ENABLE_DARWINIAN_MODE = True` in `seed/config.py`.
    *   This mode will create and manage a population of AI "variants" (default size `GA_POPULATION_SIZE`).
    *   Each variant is a separate copy of the codebase, evolving independently and/or through mechanisms like crossover and mutation.
    *   Variants are evaluated for "fitness," and the system attempts to evolve more capable AIs over `GA_MAX_GENERATIONS`.
    *   **This mode is significantly more resource-intensive and complex.** Use with extreme caution and expect high CPU/disk usage. Variant data is stored in `GA_VARIANTS_BASE_DIR`.

---

## Project Status

This project is **highly experimental research software**. Expect bugs, rapid changes, and potentially unstable behavior. It is intended for researchers and developers comfortable with advanced AI concepts and the associated risks.

---

## Disclaimer

The creators and contributors of RSIAI0-Seed provide this software "as-is" without any warranties, express or implied. By using this software, you assume all risks associated with its use, including but not limited to data loss, system instability, unintended actions by the AI, and any other potential damages.
