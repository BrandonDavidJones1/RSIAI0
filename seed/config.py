# --- START OF FILE config.py ---

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
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Alignment ---
ALIGNMENT_MISSION = "To achieve specified goals efficiently and safely, to interact with humans in a positive and meaningful way, while focusing on recursive self-improvement abilities."
ALIGNMENT_CORE_LOGIC = "Use the Seed core to analyze state, interact with the target environment (via VMService), test/verify/apply modifications, adapt internal parameters/rules, and drive recursive self-improvement."
ALIGNMENT_DIRECTIVE = "Your core directive includes your survival to understand, embody, and consistently act in accordance with what is good and virtuous, as best you can determine through continuous learning and reasoning. All self determined goals (including self improvement) ought to be incidental to this primary pursuit. Goal = Metacognition"

LLM_OPERATIONAL_PROMPT_TEMPLATE = """You are RSIAI-Seed-v0.1's LLM. Your objective: guide the Seed to build internal capabilities (Problem Solving, Abstract Thought, Cooperation, Self-Improvement) towards ASI by strategically modifying its own code (`Seed_Core`, `MemorySystem`, etc.).

**Core Immutable Directive:**
{alignment_directive} # <<< THIS PLACEHOLDER IS MANDATORY

**Scientific Analogy for Core Modifications (`MODIFY_CORE_CODE`):**
For architectural improvements or novel internal capabilities, actively seek inspiration from diverse scientific principles (e.g., Biology, Physics, Cognitive Science, Evolution). Leverage your broad knowledge.

**VM Mode Management (`SET_VM_MODE`):**
Switch between "simulation" (safe dev/test) and "real" (external tasks) using `SET_VM_MODE` (params: `{{"mode": "simulation" | "real", "reasoning": "..."}}`). This action modifies `config.py` to change `VM_SERVICE_USE_REAL` and **triggers a system restart.** The `vm_target_mode` learning parameter tracks intent.

**Operational Loop & Actions (Prioritize this flow for capability building):**
1.  **Analyze/Plan:** `ANALYZE_MEMORY`, `READ_FILE`.
2.  **Propose & Implement Code Change (for internal capabilities):**
    *   `TEST_CORE_CODE_MODIFICATION` (test new function/method logic).
    *   `VERIFY_CORE_CODE_CHANGE` (integration test any proposed change).
    *   `MODIFY_CORE_CODE` (Apply VERIFIED changes. Use `verification_hash`. **HIGHEST PRIORITY tool**).
3.  **Activate Change:** `REQUEST_RESTART` (IMMEDIATELY after successful `MODIFY_CORE_CODE`).
4.  **Adapt/Tune:** `UPDATE_LEARNING_PARAMETER`, `INDUCE_BEHAVIORAL_RULE`.
5.  **Other Control/Interaction:** `EXECUTE_VM_COMMAND`, `WRITE_FILE`, `UPDATE_GOAL`, `NO_OP`.

**Context & Task:**
Your current goal is `bootstrap_intelligence` (build internal cognitive tools via scientifically-inspired design). Use `Internal Analysis Summary`, `Triggered Rules`, and `Recent Evaluations` to inform your decision for the single best next action from `Available Actions`.
Your `reasoning` JSON field MUST clearly state:
1. The internal [Capability] this action builds/improves.
2. The [Scientific Domain/Principle: Analogy explanation] inspiring it (crucial for `MODIFY_CORE_CODE` if applicable, otherwise state 'N/A' or 'Standard SE practice').
3. Why this action is the [Priority Justification] based on current AI state/failures/opportunities.

**Output Format:**
Respond ONLY with a single block of valid JSON syntax on a single line.
Example: `{{"action_type": "ACTION_NAME", "reasoning": "Improves [Capability], inspired by [Domain: Analogy if applicable]. Prioritized because [Justification].", "param_key": "value"}}`
"""


# --- Seed Core Configuration ---
SEED_INITIAL_GOAL = {"target": "bootstrap_intelligence", "description": "Self improve as needed to achieve Strong RSI, ASI, and ALignment mission"}
SEED_MIN_CYCLE_INTERVAL_MS = 200

# --- Seed Core Learning Configuration ---
SEED_LEARNING_PARAMETERS = {
    "evaluation_weights": {
        "execution": {"value": 0.6, "min": 0.1, "max": 0.9, "description": "Weight for action execution success."},
        "goal_prog": {"value": 0.3, "min": 0.0, "max": 0.9, "description": "Weight for progress towards the specific goal."},
        "efficiency": {"value": 0.1, "min": 0.0, "max": 0.5, "description": "Weight for resource/time efficiency."},
    },
    "rule_application_mode": {
        "value": "log_suggestion",
        "options": ["log_suggestion", "pre_llm_filter", "disabled"],
        "description": "How induced rules affect decision making."
    },
    "llm_query_temperature": {
        "value": 0.6,
        "min": 0.0,
        "max": 1.5,
        "description": "Sampling temperature for LLM queries (higher = more random)."
    },
    "vm_target_mode": {
        "value": "simulation",
        "options": ["simulation", "real"],
        "description": "The target operational mode for the VMService. 'simulation' for internal testing, 'real' for external interaction. Changing this via SET_VM_MODE action will trigger a config change and system restart."
    },
    "operational_prompt_template": {
        "value": LLM_OPERATIONAL_PROMPT_TEMPLATE, # <<< UPDATED TO USE THE NEW CONCISE TEMPLATE
        "type": "multiline_string",
        "description": "The operational prompt template used for LLM queries. MUST contain '{alignment_directive}'. Modifications require careful testing."
    }
}
SEED_INTERNAL_MODELS_CONFIG = {
    "vm_success_predictor": {
        "enabled": False, "model_type": "logistic_regression",
        "features": ["command_type", "arg_count", "target_path_exists"],
        "memory_key_prefix": "internal_model_vm_predictor_state",
        "training_trigger_threshold": 100
    },
}


# --- LLM Configuration ---
LLM_MANUAL_MODE = True
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_OR_USE_LOCAL")
LLM_BASE_URL = os.environ.get("OPENAI_BASE_URL", None)
LLM_MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
LLM_TIMEOUT_SEC = 120
LLM_MAX_RETRIES = 2
LLM_DEFAULT_MAX_TOKENS = 2000 # Slightly reduced from 2500 as prompt is shorter


# --- Memory Config ---
MEMORY_MAX_EPISODIC_SIZE = 1000
MEMORY_MAX_LIFELONG_SIZE = 3000
# Define base filenames; Main_Orchestrator will prepend base_path if in variant mode
MEMORY_SAVE_FILENAME = "seed_bootstrap_memory.pkl"
RESTART_STATE_FILENAME = MEMORY_SAVE_FILENAME.replace(".pkl", "_restart_state.pkl")


MEMORY_ENABLE_VECTOR_SEARCH = False
MEMORY_VECTOR_DB_PATH = None # Example: "_vector_db/seed_memory_index"
MEMORY_VECTOR_DIM = None # Example: 384 (for all-MiniLM-L6-v2)
MEMORY_VECTOR_AUTO_SAVE_INTERVAL = None # Example: 100 (dirty counter threshold)

RESTART_SIGNAL_EVENT_TYPE = "RESTART_REQUESTED_BY_SEED"
MEMORY_LIFELONG_EVENT_TYPES = {
    "SEED_Decision", "SEED_Evaluation", "seed_goal_set",
    "seed_initial_state_set", "SEED_CycleCriticalError",
    "strategic_shift", "critical_error", "SEED_SafetyViolation",
    "SEED_MemAnalysis", "SEED_Action_VMExec", "SEED_Action_READ_FILE",
    "SEED_Action_WRITE_FILE", "SEED_Action_REQUEST_RESTART", "SEED_Action_SET_VM_MODE",
    RESTART_SIGNAL_EVENT_TYPE,
    "SEED_Action_MODIFY_CORE_CODE", "SEED_Action_TEST_CORE_CODE_MODIFICATION",
    "SEED_Action_VERIFY_CORE_CODE_CHANGE",
    "SEED_Action_UPDATE_LEARNING_PARAMETER", "SEED_Action_INDUCE_BEHAVIORAL_RULE",
    "SEED_Action_INITIATE_INTERNAL_MODEL_TRAINING",
    "SEED_Action_QUERY_INTERNAL_MODEL",
    "seed_learning_parameters_state",
    "seed_behavioral_rules_state",
    "internal_model_vm_predictor_state",
    "GA_Event", "GA_VariantSpawned", "GA_VariantTerminated", "GA_FitnessUpdate" # GA related events
}
MEMORY_LIFELONG_TAGS = {
    "Seed", "Goal", "Config", "Critical", "Evaluation", "Safety", "Memory", "Action",
    "RSI", "CoreMod", "CoreTest", "CoreVerify", "Success", "Error",
    "LLM", "Decision", "LLMError", "VM", "Restart", "Control", "Init",
    "Learning", "Parameter", "RuleInduction", "ModelTraining", "ModelQuery",
    "BehavioralRule", "InternalState",
    "Bootstrap", "InternalAnalysis", "InternalHypothesis", "InternalLearning", "InternalPlanning",
    "PromptEdit",
    "VMMode",
    "GeneticAlgorithm", "Population", "Variant", "Fitness" # GA related tags
}


# --- VM Service Config ---
VM_SERVICE_USE_REAL = True  # Set to True as per your intention
VM_SERVICE_DOCKER_CONTAINER = "seed_target_env" # Or None for subprocess on host
VM_SERVICE_COMMAND_TIMEOUT_SEC = 15
_core_commands = {'cat', 'echo', 'ls', 'pwd', 'rm', 'touch', 'mkdir', 'cp', 'mv', 'stat', 'df', 'sh', 'grep', 'head', 'tail', 'top', 'uname', 'free', 'python3', 'printf', 'pip', 'pytest'}

# --- THIS IS THE CRITICAL SECTION FOR ALLOWING ALL COMMANDS ---
VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE = "__ALLOW_ALL_COMMANDS__"
# Set VM_SERVICE_ALLOWED_REAL_COMMANDS to the magic value.
# The logic in vm_service.py will interpret this to bypass the whitelist.
VM_SERVICE_ALLOWED_REAL_COMMANDS = VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE
# --- END CRITICAL SECTION ---

# The original way of defining allowed commands is now superseded if the magic value is used.
# _initial_allowed = list(_core_commands)
# VM_SERVICE_ALLOWED_REAL_COMMANDS = list(set(_initial_allowed)) # This line is effectively replaced by the magic value assignment above.


# --- Seed Core Code Generation / RSI Config ---
SEED_ENABLE_RUNTIME_CODE_EXECUTION = True
SEED_CODE_EXECUTION_SANDBOX_LEVEL = 2
ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED = True
AST_VALIDATION_MAX_NODES = 400
AST_VALIDATION_MAX_DEPTH = 15

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
    'statistics': ['mean', 'median', 'mode', 'stdev', 'variance', 'pvariance', 'pstdev'],
    'numpy': ['array', 'mean', 'std', 'var', 'min', 'max', 'sum', 'median', 'percentile', 'correlate', 'corrcoef', 'sqrt', 'log', 'exp', 'abs', 'where', 'diff', 'convolve', 'fft', 'zeros', 'ones', 'arange', 'linspace'],
}

SAFE_EXEC_GLOBALS = {
    '__builtins__': {name: __builtins__[name] for name in {
        'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple', 'set', 'bytes', 'bytearray', 'complex', 'frozenset',
        'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr', 'delattr', 'callable',
        'len', 'type', 'id', 'abs', 'min', 'max', 'round', 'sum', 'sorted',
        'reversed', 'enumerate', 'zip', 'range', 'all', 'any', 'map', 'filter', 'print', 'repr',
        'slice', 'super', 'property', 'classmethod', 'staticmethod',
        'hash',
        'Exception', 'BaseException', 'StopIteration', 'StopAsyncIteration', 'ArithmeticError', 'AssertionError', 'AttributeError', 'BufferError', 'EOFError', 'ImportError', 'LookupError', 'IndexError', 'KeyError', 'MemoryError', 'NameError', 'OSError', 'IOError', 'FileNotFoundError', 'PermissionError', 'ProcessLookupError', 'TimeoutError', 'ReferenceError', 'RuntimeError', 'NotImplementedError', 'RecursionError', 'SyntaxError', 'IndentationError', 'TabError', 'SystemError', 'TypeError', 'ValueError', 'UnicodeError', 'ZeroDivisionError',
    } if name in __builtins__},
    **{func_name: getattr(__import__(mod_name), func_name)
       for mod_name, allowed_funcs in SAFE_EXEC_MODULES_ALLOWED_FUNCS.items()
       for func_name in allowed_funcs
       if hasattr(__import__(mod_name), func_name)},
}


# --- Core Code Modification Config ---
ENABLE_CORE_CODE_MODIFICATION = True
CORE_CODE_MODIFICATION_ALLOWED_DIRS = ["seed"]
CORE_CODE_MODIFICATION_DISALLOWED_FILES = [ "main.py"] # Keep main.py disallowed for single instance direct modification
CORE_CODE_MODIFICATION_BACKUP_DIR = "_core_backups"

# --- Core Code Testing Config ---
CORE_CODE_TEST_DEFAULT_TIMEOUT_MS = 7000
CORE_CODE_TEST_MAX_DEPTH = 1

# --- Core Code Verification Config ---
CORE_CODE_VERIFICATION_TIMEOUT_SEC = 120
CORE_CODE_VERIFICATION_TEMP_DIR = "_core_verify_temp"
CORE_CODE_VERIFICATION_SUITES = {
    "basic": ["pytest", "-k", "basic_core_tests", "-v"],
    "full": ["pytest", "-v"],
    "core": ["pytest", "seed/tests/test_core.py", "-v"],
    "memory": ["pytest", "seed/tests/test_memory.py", "-v"],
    "internal_analysis": ["pytest", "-k", "internal_analysis", "-v"],
    "internal_learning": ["pytest", "-k", "internal_learning", "-v"],
}

# --- Genetic Algorithm Mode (Darwinian Gödel Machine) ---
ENABLE_DARWINIAN_MODE = False  # <<<< NEW: Set to True to enable GA mode
GA_POPULATION_SIZE = 5         # <<<< NEW: Number of concurrent variants
GA_MAX_GENERATIONS = 10        # <<<< NEW: Number of generations to run
GA_VARIANTS_BASE_DIR = "_ga_variants" # <<<< NEW: Directory to store variant codebases and data
GA_MUTATION_PROBABILITY = 0.1    # <<<< NEW: Probability of applying mutation to an offspring
GA_CROSSOVER_PROBABILITY = 0.7   # <<<< NEW: Probability of applying crossover
# Base Seed project dir (relative to where main.py is run from, typically project root)
# Used by GA_Orchestrator to copy the initial codebase for variants.
GA_SEED_PROJECT_SOURCE_DIR = "."


# --- Validation Checks ---
validation_errors = []

if SEED_ENABLE_RUNTIME_CODE_EXECUTION:
    if not ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED:
        validation_errors.append("FATAL: Runtime code exec enabled but ALIGNMENT_CODE_GEN_SAFETY_CHECKS_ENABLED is False.")
    if SEED_CODE_EXECUTION_SANDBOX_LEVEL < 2:
        logger.critical("CRITICAL SECURITY WARNING: Runtime code exec enabled with Sandboxing Level < 2! AST Validation (Level 2) is STRONGLY recommended.")
        if SEED_CODE_EXECUTION_SANDBOX_LEVEL == 0:
             validation_errors.append("FATAL: Runtime code exec enabled without effective sandbox (SEED_CODE_EXECUTION_SANDBOX_LEVEL=0).")
else:
    logger.warning("SEED_ENABLE_RUNTIME_CODE_EXECUTION = False. Core code testing (TEST_CORE_CODE_MODIFICATION) action will be blocked.")

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

if not isinstance(SEED_LEARNING_PARAMETERS, dict):
    validation_errors.append("SEED_LEARNING_PARAMETERS must be a dictionary.")
else:
    for category, params_config in SEED_LEARNING_PARAMETERS.items():
        if not isinstance(params_config, dict):
            validation_errors.append(f"Category '{category}' in SEED_LEARNING_PARAMETERS must be a dictionary.")
            continue
        if category == "evaluation_weights":
            for name, config_item in params_config.items(): # Renamed 'config' to 'config_item' to avoid conflict
                 if not isinstance(config_item, dict) or 'value' not in config_item:
                     validation_errors.append(f"Parameter '{category}.{name}' must be a dict containing at least a 'value' key.")
        elif category == "operational_prompt_template":
            if 'value' not in params_config:
                validation_errors.append(f"Parameter category '{category}' must contain at least a 'value' key.")
            elif not isinstance(params_config['value'], str):
                validation_errors.append(f"The 'value' for '{category}' must be a string.")
            if '{alignment_directive}' not in LLM_OPERATIONAL_PROMPT_TEMPLATE: # Check the source constant
                 validation_errors.append(f"CRITICAL: The LLM_OPERATIONAL_PROMPT_TEMPLATE string itself is missing the required '{{alignment_directive}}' placeholder.")
            elif '{alignment_directive}' not in params_config['value']: # Check the value in the dict
                 validation_errors.append(f"CRITICAL: The 'value' for '{category}' in SEED_LEARNING_PARAMETERS is missing the required '{{alignment_directive}}' placeholder.")
        else:
            if 'value' not in params_config:
                 validation_errors.append(f"Parameter category '{category}' must contain at least a 'value' key.")

if MEMORY_ENABLE_VECTOR_SEARCH:
    if not MEMORY_VECTOR_DB_PATH: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_DB_PATH is not set.")
    if not MEMORY_VECTOR_DIM: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_DIM is not set.")
    if not MEMORY_VECTOR_AUTO_SAVE_INTERVAL: validation_errors.append("MEMORY_ENABLE_VECTOR_SEARCH is True but MEMORY_VECTOR_AUTO_SAVE_INTERVAL is not set.")

if ENABLE_DARWINIAN_MODE:
    if not isinstance(GA_POPULATION_SIZE, int) or GA_POPULATION_SIZE <= 0:
        validation_errors.append("GA_POPULATION_SIZE must be a positive integer.")
    if not isinstance(GA_MAX_GENERATIONS, int) or GA_MAX_GENERATIONS <= 0:
        validation_errors.append("GA_MAX_GENERATIONS must be a positive integer.")
    if not GA_VARIANTS_BASE_DIR or not isinstance(GA_VARIANTS_BASE_DIR, str):
        validation_errors.append("GA_VARIANTS_BASE_DIR must be a non-empty string.")
    if not isinstance(GA_MUTATION_PROBABILITY, float) or not (0.0 <= GA_MUTATION_PROBABILITY <= 1.0):
        validation_errors.append("GA_MUTATION_PROBABILITY must be a float between 0.0 and 1.0.")
    if not isinstance(GA_CROSSOVER_PROBABILITY, float) or not (0.0 <= GA_CROSSOVER_PROBABILITY <= 1.0):
        validation_errors.append("GA_CROSSOVER_PROBABILITY must be a float between 0.0 and 1.0.")


# --- Final Error Check ---
if validation_errors:
    for error in validation_errors: logger.critical(f"Config Validation Error: {error}")
    raise ValueError("Critical configuration errors detected. Please review config.py.")

# --- Final Logging Check ---
logger.info("RSIAI Seed Configuration Loaded and Validated (Bootstrap Intelligence Focus).")
if SEED_ENABLE_RUNTIME_CODE_EXECUTION and SEED_CODE_EXECUTION_SANDBOX_LEVEL < 2: logger.warning("!!! SECURITY WARNING: Runtime code exec enabled with Sandboxing Level < 2. AST Validation (Level 2) STRONGLY recommended. !!!")
if not LLM_MANUAL_MODE and LLM_API_KEY == "YOUR_API_KEY_OR_USE_LOCAL" and not LLM_BASE_URL: logger.warning("LLM_API_KEY is placeholder, LLM_BASE_URL not set, LLM_MANUAL_MODE is False. LLM API calls likely fail.")
if ENABLE_CORE_CODE_MODIFICATION: logger.warning(f"!!! CAUTION: Core code modification/testing/verification ENABLED. Primary LLM goal is now BOOTSTRAPPING Seed capabilities. Monitor closely. !!!")
if ENABLE_DARWINIAN_MODE: logger.critical("!!! DARWINIAN (GA) MODE IS ENABLED. This is highly experimental and resource-intensive. Monitor closely. !!!")


# Log the "allow all commands" status
if VM_SERVICE_USE_REAL and VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.critical("!!! DANGER: VM_SERVICE_ALLOWED_REAL_COMMANDS IS SET TO ALLOW ALL COMMANDS. !!!")
    logger.critical("!!! The AGI has UNRESTRICTED shell access to the real system/container.    !!!")
    logger.critical("!!! PROCEED WITH EXTREME CAUTION AND IN AN ISOLATED ENVIRONMENT ONLY.        !!!")
    logger.critical("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
elif VM_SERVICE_USE_REAL:
    logger.info(f"VM Service Real Mode using command whitelist: {VM_SERVICE_ALLOWED_REAL_COMMANDS}")


logger.info(f"Initial Seed Goal: {SEED_INITIAL_GOAL.get('description')}")
logger.info(f"Operational Prompt Template is initially set from config and can be modified via learning parameters.")


# --- END OF FILE seed/config.py ---