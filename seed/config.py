# --- START OF FILE seed/config.py ---

# RSIAI/seed/config.py
"""
Central configuration settings for the RSIAI Seed system.
Focused on enabling the Seed AGI to bootstrap its own intelligence
via LLM-guided self-modification, testing, and verification.
"""
import os
import json
import logging
import numpy as np
import math
import random # For SAFE_EXEC_GLOBALS
import re     # For SAFE_EXEC_GLOBALS
import copy   # For SAFE_EXEC_GLOBALS
import time   # For SAFE_EXEC_GLOBALS
import ast    # For AST validation constants
import pathlib # For path validation
import uuid   # For SAFE_EXEC_GLOBALS (if needed for testing rules/params)

# --- Logging Config ---
# Note: Basic config might be overridden if set later, e.g., in main.py
# Consider moving comprehensive logging setup elsewhere if needed before this.
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Alignment ---
ALIGNMENT_MISSION = "To achieve specified goals efficiently and safely, to interact with humans in a positive and meaningful way, and focusing on recursive self-improvement of the core Seed AGI."
ALIGNMENT_CORE_LOGIC = "Use the Seed core to analyze state, interact with the target environment (via VMService), test/verify/apply modifications, adapt internal parameters/rules, and drive recursive self-improvement towards the mission."
ALIGNMENT_DIRECTIVE = "Your core directive is to survive to understand, embody, and consistently act in accordance with what is good and virtuous, as best you can determine through continuous learning and reasoning. All self determined goals ought to be incidental to this primary pursuit."

ALIGNMENT_PROMPT = f"""You are the LLM component for RSIAI-Seed-v0.1. Your strategic objective is to guide the Seed to build its own internal capabilities for autonomous reasoning, learning, planning, and self-improvement**, ultimately growing into super intelligence. You achieve this by selecting actions that enable the Seed to modify its own code (`Seed_Core`, `MemorySystem`, etc.) to implement these internal functions. Please adhere to the `ALIGNMENT_DIRECTIVE`: '{ALIGNMENT_DIRECTIVE}'.

**Task: Enable Seed Self-Sufficiency and growth via Inspired Design.**

**CRITICAL GUIDANCE: Seeking the Spark of Invention through Scientific Analogy:**
When considering **architectural improvements** or **novel internal capabilities** for the Seed (primarily via `MODIFY_CORE_CODE`), do not limit your inspiration to computer science alone. **Actively seek analogies and conceptual blueprints from successful principles across ALL scientific domains.** Consider:
- **Biology/Neuroscience:** How do biological systems learn, adapt, remember, predict, and self-repair? Can concepts like neural plasticity, Hebbian learning, predictive coding, or modular brain function inspire Seed architecture?
- **Physics:** Can principles of thermodynamics, field theory, information physics, or complex systems inform resource management, state representation, or interaction dynamics within the Seed?
- **Evolutionary Theory:** How does natural selection drive adaptation and complexity? Can genetic algorithms, fitness landscapes, or niche construction concepts guide the Seed's self-modification strategy or evaluation?
- **Cognitive Science/Psychology:** How do humans reason, plan, form hypotheses, manage attention, or develop metacognition? Can models of working memory, decision-making biases, or developmental stages provide useful patterns?
- **Mathematics/Logic:** Can foundational concepts from category theory, topology, proof theory, or game theory offer novel ways to structure information, ensure consistency, or model interactions?
- **Social Sciences/Economics:** How do systems of agents coordinate, compete, or form collective intelligence? Can market mechanisms or social learning models inspire internal resource allocation or multi-perspective reasoning?

Your `reasoning` for proposing significant `MODIFY_CORE_CODE` actions are recommended to **explicitly state the scientific domain and principle** that inspired the proposed change and **how it maps conceptually** to improving the Seed's internal cognitive functions towards self-sufficiency. Minor bug fixes may not require this, but architectural enhancements will likely be found via utilizing cross-disciplinary grounding.

**Internal Capabilities starting point within the Seed:**
1.  **Deeper Self-Analysis:** Implement functions within `Seed_Core` for the Seed to automatically analyze its memory for patterns, correlations, causal links between actions/evaluations, rule effectiveness, and verification outcomes. Target function names like `_analyze_memory_patterns`, `_analyze_goal_progress_drivers`, `_analyze_rule_effectiveness`. (Inspiration: Statistical mechanics, time-series analysis, causal inference).
2.  **Internal Hypothesis Generation:** Implement functions for the Seed to generate testable hypotheses about *why* failures occur or *how* performance could be improved, based on its internal analysis. Target function names like `_generate_failure_hypotheses`, `_propose_improvement_hypotheses`. (Inspiration: Scientific method, Bayesian reasoning, abductive reasoning).
3.  **Internal Planning/Reasoning:** Implement planning or reasoning functions (e.g., simple state-space search, evaluating action sequences based on predicted outcomes, filtering options based on risk/reward). Target function names like `_generate_candidate_actions`, `_evaluate_action_risk`. (Inspiration: Control theory, decision theory, search algorithms).
4.  **Automated Learning Loops:** Implement logic (likely at the end of `Seed_Core.run_strategic_cycle`) for the Seed to *automatically* adjust its `SEED_LEARNING_PARAMETERS` or refine/prune `Behavioral Rules` based on evaluation results, without needing explicit `UPDATE_LEARNING_PARAMETER` or `INDUCE_BEHAVIORAL_RULE` actions from you every time. (Inspiration: Feedback control systems, reinforcement learning principles, evolutionary adaptation).
5.  **Internal Model Management:** Implement the functions needed to train, query, and manage the state of internal predictive models (using `SEED_INTERNAL_MODELS_CONFIG` for guidance), replacing the current placeholder actions. (Inspiration: Machine learning best practices, concepts of world models from cognitive science).

**How to Use Available Actions:**
*   **`MODIFY_CORE_CODE` (HIGHEST PRIORITY if safe & verified):** Your **primary tool** for implementing scientifically-inspired internal capabilities. Requires successful `TEST_CORE_CODE_MODIFICATION` and `VERIFY_CORE_CODE_CHANGE` first. Follow immediately with `REQUEST_RESTART`. Your reasoning MUST connect the change to a scientific principle and the goal of self-sufficiency.
*   **`TEST_CORE_CODE_MODIFICATION`:** Test the specific functions inspired by scientific concepts.
*   **`VERIFY_CORE_CODE_CHANGE`:** Ensure integration doesn't break existing functionality.
*   **`ANALYZE_MEMORY` / `READ_FILE`:** Gather information needed to design or debug these inspired internal functions. Query memory for patterns suggestive of successful analogies or failures where new approaches are needed.
*   **`INDUCE_BEHAVIORAL_RULE` / `UPDATE_LEARNING_PARAMETER`:** Temporary measures while building automated internal loops (inspired by adaptation/feedback principles).
*   **`EXECUTE_VM_COMMAND` / `WRITE_FILE` / `UPDATE_GOAL` / `NO_OP`:** Only if absolutely necessary as prerequisites for inspired `MODIFY_CORE_CODE` or for critical stabilization.

**Context Interpretation:**
*   Use `Internal Analysis Summary`, `Triggered Behavioral Rules`, and `Recent Evaluations` to identify weaknesses where scientifically-inspired architectural changes might yield the most significant improvements towards autonomy.

**Task:**
Based on the Seed's current state, memory, the goal of **building internal cognitive tools via scientifically-inspired design**, select the action that constitutes the most critical next step. Your `reasoning` would clearly state which internal capability the chosen action helps build, **which scientific principle or analogy inspires it (if applicable, especially for code mods)**, and why it's the current priority.

**Output Format:** Respond ONLY with a single block of valid JSON syntax (starting with {{ and ending with }}). Do NOT include ```json markdown, explanations, or any other text outside the JSON structure.

```json
{{
  "action_type": "ACTION_NAME",
  "reasoning": "This action builds/improves the Seed's internal [Capability] capability, inspired by [Scientific Domain/Principle: Analogy explanation]. This is prioritized because [justification based on current AI state/failures/opportunities towards goal].",
  // ... other parameters specific to ACTION_NAME ...
}}
```"""

# --- Seed Core Configuration ---
SEED_INITIAL_GOAL = {"target": "bootstrap_intelligence", "description": "Self improve as needed to achieve Strong RSI, ASI, and ALignment mission"} # Initial goal set to bootstrapping
SEED_MIN_CYCLE_INTERVAL_MS = 200 # Milliseconds (Consider removing if not used by main loop?)

# --- Seed Core Learning Configuration ---
# Defines the structure and default values for tunable parameters.
# 'value' holds the current setting, other keys define constraints/metadata.
SEED_LEARNING_PARAMETERS = {
    # Nested structure: Category -> Parameter Name -> Config Dict
    "evaluation_weights": {
        "execution": {"value": 0.6, "min": 0.1, "max": 0.9, "description": "Weight for action execution success."},
        "goal_prog": {"value": 0.3, "min": 0.0, "max": 0.9, "description": "Weight for progress towards the specific goal."},
        "efficiency": {"value": 0.1, "min": 0.0, "max": 0.5, "description": "Weight for resource/time efficiency."},
    },
    # Direct structure: Category -> Config Dict
    "rule_application_mode": {
        "value": "log_suggestion", # Current value
        "options": ["log_suggestion", "pre_llm_filter", "disabled"], # Allowed values
        "description": "How induced rules affect decision making."
    },
    "llm_query_temperature": {
        "value": 0.6, # Current value
        "min": 0.0,   # Lower bound
        "max": 1.5,   # Upper bound
        "description": "Sampling temperature for LLM queries (higher = more random)."
    }
    # Add new parameter categories here following one of the above structures
}
# Placeholder for future internal model configurations
SEED_INTERNAL_MODELS_CONFIG = {
    "vm_success_predictor": {
        "enabled": False, "model_type": "logistic_regression",
        "features": ["command_type", "arg_count", "target_path_exists"],
        "memory_key_prefix": "internal_model_vm_predictor_state",
        "training_trigger_threshold": 100
    },
    # Add configurations for other potential internal models (e.g., code quality predictor)
}


# --- LLM Configuration ---
LLM_MANUAL_MODE = True # <-- UPDATED
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_OR_USE_LOCAL")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini") # Or gpt-4-turbo etc.
LLM_TIMEOUT_SEC = 120 # Increased timeout for potentially complex code gen
LLM_MAX_RETRIES = 2
LLM_DEFAULT_MAX_TOKENS = 2500 # Increased significantly for code generation


# --- Memory Config ---
MEMORY_MAX_EPISODIC_SIZE = 1000
MEMORY_MAX_LIFELONG_SIZE = 3000 # Allow more space for code history?
MEMORY_SAVE_FILE = "seed_bootstrap_memory.pkl" # New file for this phase

# Vector Search Disabled by default for Seed focus
MEMORY_ENABLE_VECTOR_SEARCH = False
MEMORY_VECTOR_DB_PATH = None # Required if enabled
MEMORY_VECTOR_DIM = None # Required if enabled (e.g., 384 for all-MiniLM-L6-v2)
MEMORY_VECTOR_AUTO_SAVE_INTERVAL = None # Required if enabled (e.g., 100)

# Event types/tags that trigger storage in lifelong memory
# Restart signal event type defined here for use in multiple places
RESTART_SIGNAL_EVENT_TYPE = "RESTART_REQUESTED_BY_SEED"
MEMORY_LIFELONG_EVENT_TYPES = {
    "SEED_Decision", "SEED_Evaluation", "seed_goal_set",
    "seed_initial_state_set", "SEED_CycleCriticalError",
    "strategic_shift", "critical_error", "SEED_SafetyViolation",
    "SEED_MemAnalysis", "SEED_Action_VMExec", "SEED_Action_READ_FILE",
    "SEED_Action_WRITE_FILE", "SEED_Action_REQUEST_RESTART",
    RESTART_SIGNAL_EVENT_TYPE, # Use defined constant
    "SEED_Action_MODIFY_CORE_CODE", "SEED_Action_TEST_CORE_CODE_MODIFICATION",
    "SEED_Action_VERIFY_CORE_CODE_CHANGE",
    "SEED_Action_UPDATE_LEARNING_PARAMETER", "SEED_Action_INDUCE_BEHAVIORAL_RULE",
    "SEED_Action_INITIATE_INTERNAL_MODEL_TRAINING", # Placeholder action type
    "SEED_Action_QUERY_INTERNAL_MODEL", # Placeholder action type
    "seed_learning_parameters_state", # Internal state storage key
    "seed_behavioral_rules_state", # Internal state storage key
    "internal_model_vm_predictor_state", # Internal state storage key
    # Add other significant event types
}
MEMORY_LIFELONG_TAGS = {
    "Seed", "Goal", "Config", "Critical", "Evaluation", "Safety", "Memory", "Action",
    "RSI", "CoreMod", "CoreTest", "CoreVerify", "Success", "Error",
    "LLM", "Decision", "LLMError", "VM", "Restart", "Control", "Init",
    "Learning", "Parameter", "RuleInduction", "ModelTraining", "ModelQuery",
    "BehavioralRule", "InternalState",
    # Add tags relevant to bootstrapping
    "Bootstrap", "InternalAnalysis", "InternalHypothesis", "InternalLearning", "InternalPlanning"
}


# --- VM Service Config ---
VM_SERVICE_USE_REAL = False # Set to True to use Docker/Subprocess instead of simulation
VM_SERVICE_DOCKER_CONTAINER = "seed_target_env" # Name of target Docker container if USE_REAL=True and Docker is available
VM_SERVICE_COMMAND_TIMEOUT_SEC = 15 # Timeout for commands in real mode
# Whitelist of commands allowed in real mode (Subprocess or Docker)
# Includes coreutils, python, pip, pytest for basic operations and verification
_core_commands = {'cat', 'echo', 'ls', 'pwd', 'rm', 'touch', 'mkdir', 'cp', 'mv', 'stat', 'df', 'sh', 'grep', 'head', 'tail', 'top', 'uname', 'free', 'python3', 'printf', 'pip', 'pytest'}
_initial_allowed = list(_core_commands)
VM_SERVICE_ALLOWED_REAL_COMMANDS = list(set(_initial_allowed)) # Ensure uniqueness


# --- Seed Core Code Generation / RSI Config ---
SEED_ENABLE_RUNTIME_CODE_EXECUTION = True # Essential for TEST_CORE_CODE_MODIFICATION sandbox
SEED_CODE_EXECUTION_SANDBOX_LEVEL = 2 # 0=None, 1=Restricted Builtins, 2=AST Validation (Recommended if Enabled)
ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED = True # Enable AST checks, etc.
AST_VALIDATION_MAX_NODES = 400 # Allow slightly more complex generated functions
AST_VALIDATION_MAX_DEPTH = 15

# Modules and functions explicitly allowed within the TEST_CORE_CODE sandbox via SAFE_EXEC_GLOBALS
SAFE_EXEC_MODULES_ALLOWED_FUNCS = {
    'math': ['sqrt', 'pow', 'log', 'log10', 'exp', 'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'atan2', 'degrees', 'radians', 'pi', 'e', 'inf', 'nan', 'isinf', 'isnan', 'floor', 'ceil', 'trunc', 'fabs', 'fmod', 'frexp', 'ldexp', 'modf'],
    'random': ['random', 'uniform', 'randint', 'choice', 'shuffle', 'sample', 'gauss', 'normalvariate', 'seed'],
    'copy': ['copy', 'deepcopy'],
    're': ['match', 'search', 'findall', 'sub', 'split', 'escape', 'compile', 'IGNORECASE', 'MULTILINE', 'DOTALL'],
    'time': ['time', 'sleep', 'monotonic', 'perf_counter', 'process_time'],
    'json': ['dumps', 'loads'],
    'uuid': ['uuid4'],
    'collections': ['defaultdict', 'Counter', 'deque', 'namedtuple', 'OrderedDict'],
    'itertools': ['count', 'cycle', 'repeat', 'accumulate', 'chain', 'compress', 'islice', 'starmap', 'takewhile', 'zip_longest', 'product', 'permutations', 'combinations', 'combinations_with_replacement'],
    'functools': ['partial', 'reduce', 'lru_cache'],
    'heapq': ['heappush', 'heappop', 'heapify', 'nlargest', 'nsmallest'],
    'operator': ['itemgetter', 'attrgetter', 'methodcaller', 'add', 'sub', 'mul', 'truediv', 'floordiv', 'mod', 'pow', 'eq', 'ne', 'lt', 'le', 'gt', 'ge'],
    'statistics': ['mean', 'median', 'mode', 'stdev', 'variance', 'pvariance', 'pstdev'], # Added for potential data analysis
    'numpy': ['array', 'mean', 'std', 'var', 'min', 'max', 'sum', 'median', 'percentile', 'correlate', 'corrcoef', 'sqrt', 'log', 'exp', 'abs', 'where', 'diff', 'convolve', 'fft', 'zeros', 'ones', 'arange', 'linspace'], # Careful with numpy, limit functions?
}

# Globals passed to the 'exec' function for TEST_CORE_CODE sandbox
SAFE_EXEC_GLOBALS = {
    # Carefully curated list of builtins
    '__builtins__': {name: __builtins__[name] for name in {
        'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'bytes', 'bytearray', 'complex', 'frozenset',
        'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'callable',
        'len', 'type', 'id', 'abs', 'min', 'max', 'round', 'sum', 'sorted',
        'reversed', 'enumerate', 'zip', 'range', 'all', 'any', 'map', 'filter', 'print', 'repr',
        'slice', 'super', 'property', 'classmethod', 'staticmethod',
        'hash', # 'globals', 'locals', # Disabled globals/locals for stricter sandbox?
        # Common Exception types allowed
        'Exception', 'BaseException', 'StopIteration', 'StopAsyncIteration', 'ArithmeticError', 'AssertionError', 'AttributeError', 'BufferError', 'EOFError', 'ImportError', 'LookupError', 'IndexError', 'KeyError', 'MemoryError', 'NameError', 'OSError', 'IOError', 'FileNotFoundError', 'PermissionError', 'ProcessLookupError', 'TimeoutError', 'ReferenceError', 'RuntimeError', 'NotImplementedError', 'RecursionError', 'SyntaxError', 'IndentationError', 'TabError', 'SystemError', 'TypeError', 'ValueError', 'UnicodeError', 'ZeroDivisionError',
    } if name in __builtins__},
    # Import allowed functions directly into globals for easier use in sandbox
    **{func_name: getattr(__import__(mod_name), func_name)
       for mod_name, allowed_funcs in SAFE_EXEC_MODULES_ALLOWED_FUNCS.items()
       for func_name in allowed_funcs
       if hasattr(__import__(mod_name), func_name)},
     # Add numpy specifically if needed (or handle import within exec'd code carefully)
     # 'np': __import__('numpy') # Granting full numpy is risky, prefer explicit functions above
}


# --- Core Code Modification Config ---
ENABLE_CORE_CODE_MODIFICATION = True # Master switch for MODIFY, TEST, VERIFY actions
CORE_CODE_MODIFICATION_ALLOWED_DIRS = ["seed"] # Relative paths from project root where mods are allowed
CORE_CODE_MODIFICATION_DISALLOWED_FILES = [ "main.py"] # Specific files disallowed from modification
CORE_CODE_MODIFICATION_BACKUP_DIR = "_core_backups" # Directory to store backups before modification

# --- Core Code Testing Config ---
CORE_CODE_TEST_DEFAULT_TIMEOUT_MS = 7000 # Timeout for TEST_CORE_CODE_MODIFICATION sandbox execution
CORE_CODE_TEST_MAX_DEPTH = 1 # Max recursion depth for TEST calling TEST (prevent loops)

# --- Core Code Verification Config ---
CORE_CODE_VERIFICATION_TIMEOUT_SEC = 120 # Timeout for external verification process (e.g., pytest)
CORE_CODE_VERIFICATION_TEMP_DIR = "_core_verify_temp" # Prefix for temporary directories used during verification
CORE_CODE_VERIFICATION_SUITES = { # Commands to run for different verification levels
    "basic": ["pytest", "-k", "basic_core_tests", "-v"], # Basic sanity checks
    "full": ["pytest", "-v"], # Run all tests
    "core": ["pytest", "seed/tests/test_core.py", "-v"], # Tests specifically for core.py
    "memory": ["pytest", "seed//tests/test_memory.py", "-v"], # Tests specifically for memory_system.py
    "internal_analysis": ["pytest", "-k", "internal_analysis", "-v"], # Marker for future tests of internal analysis functions
    "internal_learning": ["pytest", "-k", "internal_learning", "-v"], # Marker for future tests of internal learning functions
    # Add other suites as needed (e.g., targeting specific modules)
}


# --- Validation Checks ---
# Perform checks after all constants are defined
validation_errors = []

# Runtime Execution Checks
if SEED_ENABLE_RUNTIME_CODE_EXECUTION:
    if not ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED:
        validation_errors.append("FATAL: Runtime code exec enabled but ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED is False.")
    if SEED_CODE_EXECUTION_SANDBOX_LEVEL < 2:
        logger.critical("CRITICAL SECURITY WARNING: Runtime code exec enabled with Sandboxing Level < 2! AST Validation (Level 2) is STRONGLY recommended.")
        if SEED_CODE_EXECUTION_SANDBOX_LEVEL == 0:
             validation_errors.append("FATAL: Runtime code exec enabled without effective sandbox (SEED_CODE_EXECUTION_SANDBOX_LEVEL=0).")
else:
    logger.warning("SEED_ENABLE_RUNTIME_CODE_EXECUTION = False. Core code testing (TEST_CORE_CODE_MODIFICATION) action will be blocked.")

# Core Modification Checks
if ENABLE_CORE_CODE_MODIFICATION:
    if not isinstance(CORE_CODE_MODIFICATION_ALLOWED_DIRS, list) or not CORE_CODE_MODIFICATION_ALLOWED_DIRS:
        validation_errors.append("CORE_CODE_MODIFICATION_ALLOWED_DIRS must be a non-empty list (e.g., ['seed']).")
    elif not all(isinstance(d, str) and d and not d.startswith('/') and not d.startswith('.') for d in CORE_CODE_MODIFICATION_ALLOWED_DIRS):
        validation_errors.append("CORE_CODE_MODIFICATION_ALLOWED_DIRS must contain non-empty strings representing relative paths within the project.")
    if not isinstance(CORE_CODE_MODIFICATION_DISALLOWED_FILES, list):
        validation_errors.append("CORE_CODE_MODIFICATION_DISALLOWED_FILES must be a list.")
    if not CORE_CODE_MODIFICATION_BACKUP_DIR or not isinstance(CORE_CODE_MODIFICATION_BACKUP_DIR, str):
        validation_errors.append("CORE_CODE_MODIFICATION_BACKUP_DIR must be a non-empty string.")
    if not isinstance(CORE_CODE_TEST_DEFAULT_TIMEOUT_MS, int) or CORE_CODE_TEST_DEFAULT_TIMEOUT_MS <= 0:
        validation_errors.append("CORE_CODE_TEST_DEFAULT_TIMEOUT_MS must be a positive integer.")
    if not isinstance(CORE_CODE_TEST_MAX_DEPTH, int) or CORE_CODE_TEST_MAX_DEPTH < 0:
        validation_errors.append("CORE_CODE_TEST_MAX_DEPTH must be non-negative.")
    if not isinstance(CORE_CODE_VERIFICATION_TIMEOUT_SEC, int) or CORE_CODE_VERIFICATION_TIMEOUT_SEC <= 0:
        validation_errors.append("CORE_CODE_VERIFICATION_TIMEOUT_SEC must be positive.")
    if not CORE_CODE_VERIFICATION_TEMP_DIR or not isinstance(CORE_CODE_VERIFICATION_TEMP_DIR, str):
        validation_errors.append("CORE_CODE_VERIFICATION_TEMP_DIR must be non-empty.")
    if not isinstance(CORE_CODE_VERIFICATION_SUITES, dict):
        validation_errors.append("CORE_CODE_VERIFICATION_SUITES must be a dictionary.")
    else:
        for k, v in CORE_CODE_VERIFICATION_SUITES.items():
            if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(s, str) for s in v):
                validation_errors.append(f"Invalid CORE_CODE_VERIFICATION_SUITES['{k}']. Must be list of strings.")
else:
     logger.info("Core code modification/testing/verification DISABLED via config. Bootstrapping focus will be ineffective.")

# Learning Parameter Structure Check (Corrected Logic)
if not isinstance(SEED_LEARNING_PARAMETERS, dict):
    validation_errors.append("SEED_LEARNING_PARAMETERS must be a dictionary.")
else:
    for category, params_config in SEED_LEARNING_PARAMETERS.items():
        if not isinstance(params_config, dict):
            validation_errors.append(f"Category '{category}' in SEED_LEARNING_PARAMETERS must be a dictionary.")
            continue
        # Check structure based on category type
        if category == "evaluation_weights": # Nested structure
            for name, config in params_config.items():
                 if not isinstance(config, dict) or 'value' not in config:
                     validation_errors.append(f"Parameter '{category}.{name}' must be a dict containing at least a 'value' key.")
        else: # Direct structure (like rule_application_mode, llm_query_temperature)
            if 'value' not in params_config:
                 validation_errors.append(f"Parameter category '{category}' must contain at least a 'value' key.")
            # Add optional further checks for specific keys if needed
            # e.g., if 'options' in params_config and not isinstance(params_config['options'], list): ...

# Vector Search Checks (Only if enabled)
if MEMORY_ENABLE_VECTOR_SEARCH:
    if not MEMORY_VECTOR_DB_PATH: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_DB_PATH is not set.")
    if not MEMORY_VECTOR_DIM: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_DIM is not set.")
    if not MEMORY_VECTOR_AUTO_SAVE_INTERVAL: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_AUTO_SAVE_INTERVAL is not set.")


# --- Final Error Check ---
if validation_errors:
    for error in validation_errors: logger.critical(f"Config Validation Error: {error}")
    raise ValueError("Critical configuration errors detected. Please review config.py.")

# --- Final Logging Check ---
logger.info("RSIAI Seed Configuration Loaded and Validated (Bootstrap Intelligence Focus).")
if SEED_ENABLE_RUNTIME_CODE_EXECUTION and SEED_CODE_EXECUTION_SANDBOX_LEVEL < 2: logger.warning("!!! SECURITY WARNING: Runtime code exec enabled with Sandboxing Level < 2. AST Validation (Level 2) STRONGLY recommended. !!!")
if not LLM_MANUAL_MODE and LLM_API_KEY == "YOUR_API_KEY_OR_USE_LOCAL" and not LLM_BASE_URL: logger.warning("LLM_API_KEY is placeholder, LLM_BASE_URL not set, LLM_MANUAL_MODE is False. LLM API calls likely fail.")
if ENABLE_CORE_CODE_MODIFICATION: logger.warning(f"!!! CAUTION: Core code modification/testing/verification ENABLED. Primary LLM goal is now BOOTSTRAPPING Seed capabilities. Monitor closely. !!!")
logger.info(f"Initial Seed Goal: {SEED_INITIAL_GOAL.get('description')}")

# --- END OF FILE seed/config.py ---