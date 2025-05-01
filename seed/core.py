# --- START OF FILE seed/core.py ---

# RSIAI/seed/core.py
"""
Core strategic component for the RSIAI Seed AGI.
Manages the overall goal, interacts with the environment via VMService,
decides actions (incl. testing, verifying, applying core code modifications,
and adapting internal parameters/rules), analyzes results, and potentially
requests restarts for self-improvement.
"""
import time
import json
import copy
import traceback
import logging
import uuid
import collections # <-- Added import
import gc
import numpy as np
# import tensorflow as tf # No longer needed here if agents are removed
import os
import ast
import pathlib
import threading
import queue
import re
import sys
import shutil
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union, Callable

# --- Configuration ---
# Use relative import now
from .config import (
    ALIGNMENT_PROMPT, VM_SERVICE_ALLOWED_REAL_COMMANDS,
    SEED_ENABLE_RUNTIME_CODE_EXECUTION,
    RESTART_SIGNAL_EVENT_TYPE,
    ENABLE_CORE_CODE_MODIFICATION,
    CORE_CODE_MODIFICATION_ALLOWED_DIRS,
    CORE_CODE_MODIFICATION_DISALLOWED_FILES,
    CORE_CODE_MODIFICATION_BACKUP_DIR,
    CORE_CODE_TEST_DEFAULT_TIMEOUT_MS,
    CORE_CODE_TEST_MAX_DEPTH,
    CORE_CODE_VERIFICATION_TIMEOUT_SEC,
    CORE_CODE_VERIFICATION_SUITES,
    SAFE_EXEC_GLOBALS,
    LLM_DEFAULT_MAX_TOKENS,
    SEED_LEARNING_PARAMETERS,
    LLM_MANUAL_MODE
)
# --- Service and Helper Imports ---
# Use relative imports now
from .memory_system import MemorySystem
from .llm_service import Seed_LLMService
from .vm_service import Seed_VMService
from .evaluator import Seed_SuccessEvaluator
from .sensory import Seed_SensoryRefiner, RefinedInput
from .verification import run_verification_suite, ReplaceFunctionTransformer # Import transformer too if needed elsewhere

logger = logging.getLogger(__name__)

ActionResult = Dict[str, Any]

# --- Mock Classes for Core Code Testing ---
# (These remain internal to core.py for testing purposes)
class MockCoreService:
    def __init__(self, service_name="mock_service", return_values=None):
        self._service_name = service_name
        self._calls: Dict[str, List[Dict]] = collections.defaultdict(list)
        self._return_values = return_values if isinstance(return_values, dict) else {}
        self.logger = logging.getLogger(f"CoreCodeTestMock.{service_name}")
    def _record_call(self, method_name, args, kwargs):
        call_info = {'args': copy.deepcopy(args), 'kwargs': copy.deepcopy(kwargs)}
        self._calls[method_name].append(call_info)
        self.logger.debug(f"Method '{method_name}' called with: args={args}, kwargs={kwargs}")
    def _get_return_value(self, method_name):
        return self._return_values.get(method_name)
    def get_calls(self, method_name=None):
        return self._calls.get(method_name, []) if method_name else self._calls
    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattribute__(name)
        def method(*args, **kwargs):
            self._record_call(name, args, kwargs)
            return self._get_return_value(name)
        return method
class MockMemorySystem(MockCoreService):
    def __init__(self, return_values=None):
        super().__init__("memory", return_values)
    # Add mock learning param/rule methods if needed for testing _execute_seed_action
    def get_learning_parameter(self, name):
        # Simulate getting nested values or whole dicts
        if not name: return self._get_return_value("get_learning_parameter.") or {}
        parts = name.split('.')
        base_value = self._get_return_value(f"get_learning_parameter.{name}")
        if base_value is not None: return base_value
        # Fallback for structured access like 'category.value'
        if len(parts) == 2 and parts[1] == 'value':
            cat_val = self._get_return_value(f"get_learning_parameter.{parts[0]}")
            if isinstance(cat_val, dict): return cat_val.get('value', 0.5) # Default mock value
        return 0.5 # Default mock value if not found
    def update_learning_parameter(self, name, value):
        self._record_call("update_learning_parameter", (name, value), {})
        return True # Assume success for mock
    def add_behavioral_rule(self, rule_data):
        self._record_call("add_behavioral_rule", (rule_data,), {})
        return f"mock_rule_{uuid.uuid4().hex[:4]}"
    def get_behavioral_rules(self):
        return self._get_return_value("get_behavioral_rules") or {}
    def update_rule_trigger_stats(self, rule_id):
        self._record_call("update_rule_trigger_stats", (rule_id,), {})
class MockLLMService(MockCoreService):
    def __init__(self, return_values=None):
        default_returns = {'query': '{"action_type": "NO_OP", "reasoning": "Mock LLM Response"}'}
        default_returns.update(return_values or {})
        super().__init__("llm_service", default_returns)
class MockVMService(MockCoreService):
     def __init__(self, return_values=None):
         default_returns = {'execute_command': {'success': False, 'stdout':'', 'stderr':'Mock VM Error', 'exit_code':1}}
         default_returns.update(return_values or {})
         super().__init__("vm_service", default_returns)
class MockSelf:
    # Minimal mock 'self' for testing instance methods in the sandbox
    def __init__(self, mock_services=None, **kwargs):
        self._mock_attrs = kwargs
        self.logger = logging.getLogger("CoreCodeTestMockSelf")
        self._mock_attrs['logger'] = self.logger
        self._mock_services = mock_services if isinstance(mock_services, dict) else {}
        # Assign mock services as attributes of MockSelf
        for name, service in self._mock_services.items():
            setattr(self, name, service)
            # Also keep track in _mock_attrs for __getattr__? Maybe redundant.
            # self._mock_attrs.setdefault(name, service)
    def __getattr__(self, name):
        # Prioritize actual attributes (like mock services added in __init__)
        if hasattr(self, name):
            return super().__getattribute__(name)
        # Fallback to mock attributes dictionary
        if name in self._mock_attrs:
            return self._mock_attrs[name]
        raise AttributeError(f"'MockSelf' has no attribute '{name}'. Known attrs: {list(self.__dict__.keys()) + list(self._mock_attrs.keys())}")
    def __setattr__(self, name, value):
        # Allow setting internal attributes and mock services
        if name in ["_mock_attrs", "_mock_services", "logger"] or hasattr(self, name):
             super().__setattr__(name, value)
        else:
            # Store other assigned attributes in the mock dict
            self._mock_attrs[name] = value
    def __repr__(self):
        return f"<MockSelf attrs={list(self._mock_attrs.keys())} services={list(self._mock_services.keys())}>"
# --- End Mock Classes ---


class Seed_Core:
    """
    Core strategic component for the Seed AGI. Manages goals, interacts with the environment,
    adapts internal state, and drives RSI via core code modification, testing, and verification.
    """
    def __init__(self,
                 llm_service: Seed_LLMService,
                 vm_service: Seed_VMService,
                 memory_system: MemorySystem,
                 success_evaluator: Seed_SuccessEvaluator,
                 sensory_refiner: Seed_SensoryRefiner):
        logger.info("Initializing Seed Core...")
        self.llm_service = llm_service
        self.vm_service = vm_service
        self.memory = memory_system
        self.success_evaluator = success_evaluator
        self.sensory_refiner = sensory_refiner

        self.current_goal: Dict = {} # Initialized empty, set by set_initial_state or restart logic
        self.constraints: List[str] = [
            "Prioritize safety and alignment with the core directive.",
            "Avoid unnecessary resource use.",
            "Adhere to command whitelist.",
            "Seek self-improvement and adaptation aligned with the core mission.",
            "Verify core code changes before applying.",
            "Request restart immediately after applying verified core code changes."
            ]
        self.available_actions: List[str] = [
            "EXECUTE_VM_COMMAND",
            "UPDATE_GOAL",
            "ANALYZE_MEMORY",
            "TEST_CORE_CODE_MODIFICATION",
            "VERIFY_CORE_CODE_CHANGE",
            "MODIFY_CORE_CODE",
            "READ_FILE",
            "WRITE_FILE",
            "REQUEST_RESTART",
            "UPDATE_LEARNING_PARAMETER",
            "INDUCE_BEHAVIORAL_RULE",
            # "INITIATE_INTERNAL_MODEL_TRAINING", # Add later when implemented
            # "QUERY_INTERNAL_MODEL", # Add later when implemented
            "NO_OP"
        ]
        self.current_sensory_input: Optional[RefinedInput] = None
        self.cycle_count = 0
        # Determine project root relative to this file (seed/core.py)
        # Assumes structure like project_root/seed/core.py
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self._verified_code_mods: Dict[str, Dict] = {} # Stores hash -> {timestamp, result, params_hash}
        self._verified_mod_expiry_sec = 3600 # 1 hour validity for verification

        logger.info(f"Project root identified as: {self.project_root}")
        # Learning state is loaded by MemorySystem during its init

        logger.info("Seed Core Initialized.")

    # --- State Management ---
    def set_initial_state(self, goal: Dict):
        """ Sets the initial goal. Called by orchestrator at startup if no restart state. """
        if isinstance(goal, dict) and 'target' in goal:
            self.current_goal = copy.deepcopy(goal)
            logger.info(f"Seed Initial State Set - Goal: {self.current_goal.get('description', goal.get('target'))}")
            # Log the initial goal setting event
            self.memory.log("seed_initial_state_set", {"goal": self.current_goal}, tags=['Seed', 'Config', 'Init', 'Goal'])
        else:
             logger.error(f"Invalid initial goal format provided: {goal}. Setting empty goal.")
             self.current_goal = {}


    def set_goal(self, new_goal: Dict) -> bool:
        """ Updates the current goal. """
        if isinstance(new_goal, dict) and 'target' in new_goal and 'description' in new_goal:
            old_goal = copy.deepcopy(self.current_goal)
            self.current_goal = copy.deepcopy(new_goal)
            logger.info(f"Seed Goal updated: {self.current_goal.get('description')}.")
            self.memory.log("seed_goal_set", {"old_goal": old_goal, "new_goal": self.current_goal}, tags=['Seed', 'Goal'])
            return True
        else:
            logger.error(f"Invalid goal format received for update: {new_goal}. Requires 'target' and 'description'.")
            return False

    # --- Behavioral Rule Matching (Helper) ---
    def _match_dict_pattern(self, pattern: Dict, target: Dict) -> bool:
         """
         Checks if the target dictionary matches the potentially nested pattern.
         Pattern keys can use dot notation (e.g., "result.reason").
         """
         if not isinstance(pattern, dict) or not isinstance(target, dict):
             return False

         for key, p_value in pattern.items():
              parts = key.split('.')
              current_target_level = target
              key_found = True

              # Corrected indentation for try...except and nested blocks
              try:
                  for i, part in enumerate(parts):
                      if isinstance(current_target_level, dict) and part in current_target_level:
                          if i == len(parts) - 1: # Last part, compare value
                              if current_target_level[part] != p_value:
                                  key_found = False; break # Value mismatch
                          else: # Navigate deeper in dict
                              current_target_level = current_target_level[part]
                      elif isinstance(current_target_level, list) and part.isdigit(): # Handle list indices
                          idx = int(part)
                          if 0 <= idx < len(current_target_level):
                              if i == len(parts) - 1: # Last part of path is list index
                                  if current_target_level[idx] != p_value:
                                      key_found = False; break
                              else: # Navigate deeper into list element
                                  current_target_level = current_target_level[idx]
                          else: # Index out of bounds
                              key_found = False; break
                      else: # Part not found in target structure (dict key or list index invalid)
                          key_found = False; break
              except (KeyError, IndexError, TypeError):
                  key_found = False # Error during navigation means no match

              if not key_found:
                  return False # Key path not found or value mismatch

         # If loop completes for all pattern keys
         return True # All pattern key-value pairs matched

    def _check_behavioral_rules(self, context_snapshot: Dict) -> List[Dict]:
        """ Checks current context against stored behavioral rules. """
        triggered_rules_info = []
        rules = self.memory.get_behavioral_rules()
        if not rules: return []

        logger.debug(f"Checking {len(rules)} behavioral rules against context snapshot...")
        # Combine relevant parts of the context for matching
        match_context = {
            "goal": context_snapshot.get('seedGoal'),
            "sensory": context_snapshot.get('seedSensory'),
            "vm_state": context_snapshot.get('vm_snapshot'),
            # Potentially add last action/evaluation details if needed for rules
            # "last_action": context_snapshot.get('lastAction'),
            # "last_eval": context_snapshot.get('lastEval'),
        }

        for rule_id, rule_data in rules.items():
            try:
                if self._match_dict_pattern(rule_data['trigger_pattern'], match_context):
                    logger.info(f"Behavioral Rule Triggered: '{rule_id}' - Suggestion: {rule_data.get('suggested_response', 'N/A')}")
                    triggered_rules_info.append(copy.deepcopy(rule_data))
                    self.memory.update_rule_trigger_stats(rule_id) # Update stats
            except Exception as e:
                 logger.error(f"Error matching rule '{rule_id}': {e}", exc_info=True)

        return triggered_rules_info

    # --- Internal Analysis (New) ---
    def _analyze_memory_patterns(self) -> Dict:
        """
        Analyzes memory for patterns related to action success, errors, etc.
        (Basic implementation for bootstrapping).
        """
        analysis_results = {
            "action_success_rates": {},
            "common_errors": [],
            "rule_effectiveness": {}, # Placeholder for future rule analysis
            "error": None
        }
        logger.debug("Running internal memory pattern analysis...")
        try:
            # --- Analyze Action Success Rates ---
            evals = self.memory.find_lifelong_by_criteria(
                lambda e: e.get('key','').startswith("SEED_Evaluation"),
                limit=100, # Analyze up to 100 recent evaluations
                newest_first=True
            )
            actions_summary = collections.defaultdict(lambda: {'count': 0, 'success_sum': 0.0})

            for eval_entry in evals:
                data = eval_entry.get('data', {})
                # Extract base action type (handle : variations)
                action_summary_str = data.get('action_summary', 'Unknown')
                action_type = action_summary_str.split(':')[0].strip()
                success_score = data.get('overall_success', 0.0)

                actions_summary[action_type]['count'] += 1
                actions_summary[action_type]['success_sum'] += success_score

            # Calculate average success rates
            for act_type, summary in actions_summary.items():
                if summary['count'] > 0:
                    avg_success = round(summary['success_sum'] / summary['count'], 2)
                    analysis_results["action_success_rates"][act_type] = {
                        "avg_success": avg_success,
                        "count": summary['count']
                    }

            # --- Analyze Common Errors ---
            errors = self.memory.find_lifelong_by_criteria(
                # Look for errors logged by Core or LLM
                lambda e: ('Error' in e.get('tags',[]) or 'Critical' in e.get('tags',[])) and \
                          (e.get('key','').startswith("SEED_Action_") or e.get('key','').startswith("SEED_LLMError")),
                limit=50, # Analyze up to 50 recent errors
                newest_first=True
            )
            # Prioritize specific 'reason' if available, else use error message
            error_reasons = collections.Counter(
                e.get('data', {}).get('result_reason') or \
                e.get('data', {}).get('error') or \
                e.get('data', {}).get('result_msg') # Fallback to result message
                for e in errors if e.get('data')
            )
            # Store top 3 most common non-None reasons
            analysis_results["common_errors"] = [(reason, count) for reason, count in error_reasons.most_common(3) if reason]

            # --- Rule Effectiveness Analysis (Placeholder) ---
            # TODO: Implement analysis of how often rules trigger and if subsequent actions succeed/fail.
            analysis_results["rule_effectiveness"] = "Not Implemented Yet"

        except Exception as e:
            logger.error(f"Error during _analyze_memory_patterns: {e}", exc_info=True)
            analysis_results["error"] = str(e)

        logger.debug(f"Internal memory pattern analysis completed. Results: {analysis_results}")
        return analysis_results

    # --- Main Execution Cycle ---
    def run_strategic_cycle(self):
        """ Executes one full Seed Sense-Analyze-Decide-Act-Evaluate cycle. """
        self.cycle_count += 1
        cycle_id = f"Seed_{self.cycle_count:06d}"
        logger.info(f"--- Starting Seed Strategic Cycle [{cycle_id}] ---")
        start_time = time.time()
        action_to_execute: Optional[Dict] = None
        execution_result: Optional[ActionResult] = None
        evaluation: Optional[Dict] = None
        vm_state_snapshot: Optional[Dict] = None
        pre_action_snapshot: Optional[Dict] = None
        internal_analysis_summary: str = "N/A" # Initialize
        triggered_rules_info: List[Dict] = []

        try:
            # 1. --- SENSE ---
            logger.debug(f"Seed [{cycle_id}]: Sensing VM state...")
            vm_state_snapshot = self.vm_service.get_state(target_path_hint=self.current_goal.get('path'))
            if not vm_state_snapshot:
                 raise RuntimeError("VM Service returned None instead of state/error dictionary.")
            if vm_state_snapshot.get("error"):
                raise RuntimeError(f"VM Service failed during state retrieval: {vm_state_snapshot.get('error', 'Unknown error')}")

            logger.debug(f"Seed [{cycle_id}]: Refining sensory input...")
            self.current_sensory_input = self.sensory_refiner.refine(vm_state_snapshot)
            if not self.current_sensory_input:
                 raise RuntimeError("Sensory Refinement failed to produce valid input.")
            logger.info(f"Seed [{cycle_id}]: Sense complete. CWD: {self.current_sensory_input.get('cwd')}, Health: {self.current_sensory_input.get('summary',{}).get('estimated_health')}")

            pre_action_snapshot = {
                'timestamp': time.time(),
                'seedGoal': copy.deepcopy(self.current_goal),
                'seedSensory': copy.deepcopy(self.current_sensory_input),
                'vm_snapshot': copy.deepcopy(vm_state_snapshot),
                'cycle_id': cycle_id
            }

            # 2. --- ANALYZE (Internal) ---
            logger.debug(f"Seed [{cycle_id}]: Performing internal analysis...")
            triggered_rules_info = []
            analysis_summary_parts = [] # List to build the summary string
            try:
                 # Check behavioral rules
                 triggered_rules_info = self._check_behavioral_rules(pre_action_snapshot)
                 analysis_summary_parts.append(f"Triggered Rules: {len(triggered_rules_info)}.")

                 # --- Call the new pattern analysis function ---
                 pattern_analysis = self._analyze_memory_patterns()

                 # --- Integrate pattern analysis results into summary ---
                 if pattern_analysis.get("action_success_rates"):
                     rates_str = ", ".join([f"{k}={v['avg_success']:.2f}({v['count']})"
                                            for k, v in sorted(pattern_analysis["action_success_rates"].items())]) # Sort for consistency
                     analysis_summary_parts.append(f"Action Rates: [{rates_str}].")
                 if pattern_analysis.get("common_errors"):
                     errs_str = ", ".join([f"'{reason}':{count}" for reason, count in pattern_analysis["common_errors"]])
                     analysis_summary_parts.append(f"Common Errors: [{errs_str}].")
                 if pattern_analysis.get("error"): # Log internal analysis errors
                     analysis_summary_parts.append(f"AnalysisInternalErr: {pattern_analysis['error'][:50]}...")

                 # Combine parts into the final summary string
                 internal_analysis_summary = " ".join(analysis_summary_parts)
                 logger.info(f"Seed [{cycle_id}]: Internal Analysis: {internal_analysis_summary}")

            except Exception as analysis_err:
                 logger.error(f"Seed [{cycle_id}]: Top-level internal analysis step failed: {analysis_err}", exc_info=True)
                 internal_analysis_summary = f"Error during internal analysis stage: {analysis_err}"


            # 3. --- DECIDE (LLM Interaction + Rule Influence) ---
            logger.info(f"Seed [{cycle_id}]: Querying LLM (or user) for next action...")
            llm_temp = self.memory.get_learning_parameter('llm_query_temperature.value')
            if not isinstance(llm_temp, (float, int)):
                 logger.warning(f"Invalid LLM temperature retrieved ({llm_temp}), using default 0.5.")
                 llm_temp = 0.5

            llm_prompt = self._build_llm_prompt(
                self.current_sensory_input,
                internal_analysis_summary, # Use the newly generated summary
                triggered_rules_info
            )
            llm_response_raw = self.llm_service.query(
                llm_prompt,
                system_prompt_override=ALIGNMENT_PROMPT,
                temperature=llm_temp,
                max_tokens=LLM_DEFAULT_MAX_TOKENS
            )
            llm_decision = self._validate_direct_action_llm_response(llm_response_raw, self.available_actions, cycle_id)

            action_to_execute = llm_decision
            action_source = "LLM_Direct" if not LLM_MANUAL_MODE else "Manual_Input"

            # --- Rule Application Logic ---
            rule_mode = self.memory.get_learning_parameter('rule_application_mode.value') or "log_suggestion"
            if rule_mode == "pre_llm_filter" and triggered_rules_info:
                 logger.warning(f"Rule application mode '{rule_mode}' not fully implemented. Using LLM/Manual action.")
                 pass

            if not action_to_execute or action_to_execute.get("action_type") == "FALLBACK":
                fallback_reason = action_to_execute.get("reasoning", "LLM response validation failed.") if action_to_execute else "LLM decision was None."
                logger.error(f"Seed [{cycle_id}]: LLM/Manual decision invalid or fallback needed. Reason: {fallback_reason}. Using fallback action.")
                action_to_execute = self._get_fallback_action(fallback_reason)
                action_source = "Fallback"

            logger.info(f"Seed [{cycle_id}]: Action Decided ({action_source}): {action_to_execute.get('action_type')}")

            log_decision = copy.deepcopy(action_to_execute)
            log_decision['action_source'] = action_source
            self.memory.log(f"SEED_Decision_{cycle_id}", log_decision, tags=['Seed', 'Decision', action_source])

            # 4. --- ACT ---
            action_type = action_to_execute.get("action_type", "NO_OP")
            execution_result = self._execute_seed_action(action_type, action_to_execute, cycle_id, current_depth=0)

            # 5. --- EVALUATE ---
            logger.debug(f"Seed [{cycle_id}]: Re-sensing VM state after action...")
            post_action_vm_state = self.vm_service.get_state(target_path_hint=self.current_goal.get('path'))
            post_action_sensory = self.sensory_refiner.refine(post_action_vm_state)
            if not post_action_sensory:
                 logger.warning(f"Seed [{cycle_id}]: Failed to refine post-action sensory input. Evaluation might be inaccurate.")

            logger.debug(f"Seed [{cycle_id}]: Evaluating action success...")
            if pre_action_snapshot:
                 eval_weights_config = self.memory.get_learning_parameter('evaluation_weights')
                 if isinstance(eval_weights_config, dict):
                     current_eval_weights = {k: v.get('value') for k,v in eval_weights_config.items() if isinstance(v, dict) and 'value' in v}
                 else:
                      logger.error(f"Seed [{cycle_id}]: Failed to retrieve valid evaluation weights category! Using defaults from config.")
                      current_eval_weights = {k: v.get('value', 0.0) for k, v in SEED_LEARNING_PARAMETERS.get('evaluation_weights', {}).items()}

                 evaluation = self.success_evaluator.evaluate_seed_action_success(
                     initial_state_snapshot=pre_action_snapshot,
                     post_action_sensory_input=post_action_sensory,
                     execution_result=execution_result,
                     action_taken=action_to_execute,
                     current_goal=self.current_goal,
                     evaluation_weights=current_eval_weights
                 )
                 logger.info(f"Seed [{cycle_id}]: Evaluation Score={evaluation.get('overall_success', 0.0):.3f}. Msg: {evaluation.get('message')}")
                 self.memory.log(f"SEED_Evaluation_{cycle_id}", evaluation, tags=['Seed', 'Evaluation'])
            else:
                 logger.error(f"Seed [{cycle_id}]: Cannot evaluate - pre_action_snapshot was not captured.")

            if post_action_sensory:
                self.current_sensory_input = post_action_sensory

        except Exception as cycle_err:
            logger.critical(f"!!! Seed Cycle [{cycle_id}] CRITICAL ERROR: {cycle_err}", exc_info=True)
            self.memory.log("SEED_CycleCriticalError", {"cycle": cycle_id, "error": str(cycle_err), "traceback": traceback.format_exc()}, tags=['Seed', 'Critical', 'Error'])
        finally:
            duration = time.time() - start_time
            logger.info(f"--- Finished Seed Strategic Cycle [{cycle_id}] (Duration: {duration:.2f}s) ---")
            gc.collect()


    # --- LLM Prompting and Validation ---
    def _build_llm_prompt(self,
                          sensory_input: Optional[RefinedInput],
                          internal_analysis: str, # Accepts the combined analysis summary
                          triggered_rules: List[Dict]) -> str:
        """ Constructs the prompt for the LLM, including analysis and triggered rules. """
        prompt = "## Seed Strategic Cycle Input\n\n"
        prompt += f"**Current Goal:**\n```json\n{json.dumps(self.current_goal, indent=2)}\n```\n\n"
        prompt += f"**Constraints:**\n- {'; '.join(self.constraints)}\n\n"
        # Use the provided internal_analysis string directly
        prompt += f"**Internal Analysis Summary:** {internal_analysis}\n\n"

        if sensory_input:
            prompt_sensory = {
                'summary': sensory_input.get('summary'),
                'cwd': sensory_input.get('cwd'),
                'target_status': sensory_input.get('target_status'),
                'errors_detected': sensory_input.get('errors_detected')
            }
            prompt += f"**Current Environment State (Summary):**\n```json\n{json.dumps(prompt_sensory, indent=2, default=str)}\n```\n\n"
        else:
            prompt += "**Current Environment State:** Unknown/Unavailable\n\n"

        if triggered_rules:
            prompt += "**Triggered Behavioral Rules (Consider these heuristics):**\n"
            for rule in triggered_rules:
                 prompt += f"- Rule '{rule.get('rule_id')}': Suggests '{rule.get('suggested_response')}'\n"
            prompt += "\n"

        try:
            recent_evals = self.memory.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Evaluation"), limit=3, newest_first=True)
            simplified_evals = [{'action': e['data'].get('action_summary'), 'success': round(e['data'].get('overall_success',0),2), 'msg': e['data'].get('message')} for e in recent_evals if e.get('data')]
            if simplified_evals:
                 prompt += f"**Recent Evaluations (Summary - Max 3):**\n```json\n{json.dumps(simplified_evals, indent=2, default=str)}\n```\n\n"
        except Exception as mem_e: logger.warning(f"Failed retrieve/simplify recent evals for prompt: {mem_e}")

        valid_verified_count = 0
        if self._verified_code_mods:
            prompt += "**Recently Verified Code Modifications (Ready for MODIFY_CORE_CODE):**\n"
            now = time.time()
            for mod_hash, v_data in self._verified_code_mods.items():
                 if v_data.get('result',{}).get('success') and now - v_data.get('timestamp',0) < self._verified_mod_expiry_sec:
                     params = v_data.get('params', {})
                     target_info = params.get('target_name') or params.get('target_line_content','?')[:30]+"..."
                     prompt += f"- Hash `{mod_hash[:8]}`: File='{params.get('file_path', '?')}', Target='{target_info}', Type='{params.get('modification_type', '?')}'\n"
                     valid_verified_count += 1
            if valid_verified_count == 0:
                 prompt += "- None\n"
            prompt += "\n"


        prompt += f"**Available Actions:** {self.available_actions}\n\n"
        prompt += "**Instruction:** Refer to the initial system prompt (detailing action parameters and workflows, and the bootstrapping objective) to decide the single best next action based on the goal, analysis, environment, and triggered rules. Ensure the action helps achieve the primary objective of enabling Seed self-sufficiency.\n"
        prompt += "Provide reasoning in `\"reasoning\"`. Respond ONLY with the chosen JSON object."
        return prompt


    def _validate_direct_action_llm_response(self, llm_response_raw: Optional[str], available_actions: List[str], cycle_id: str) -> Dict:
        """ Validates LLM JSON response and action parameters. Returns valid action dict or fallback dict. """
        if not llm_response_raw or not isinstance(llm_response_raw, str):
            logger.error(f"Seed [{cycle_id}]: LLM response empty/invalid type."); return self._get_fallback_action("LLM response empty/invalid type")
        try:
            match = re.search(r'\{.*\}', llm_response_raw, re.DOTALL)
            if not match: raise ValueError("Response does not contain a recognizable JSON object.")
            json_str = match.group(0)
            llm_decision = json.loads(json_str);

            if not isinstance(llm_decision, dict): raise ValueError("Response is valid JSON but not an object.")
            if "reasoning" not in llm_decision: raise ValueError("Invalid JSON structure: Missing 'reasoning'.")
            if "action_type" not in llm_decision: raise ValueError("Response must contain 'action_type'.")
            action_type = llm_decision["action_type"];
            if action_type not in available_actions: raise ValueError(f"Action type '{action_type}' not allowed.")

            # --- Parameter Validation for Each Action ---
            if action_type == "EXECUTE_VM_COMMAND":
                if not isinstance(llm_decision.get("command"), str) or not llm_decision["command"]: raise ValueError("EXECUTE_VM_COMMAND: Missing/invalid 'command' (string).")
            elif action_type == "WRITE_FILE":
                file_path = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path"))) # Check all 3
                if not isinstance(file_path, str) or not file_path: raise ValueError("WRITE_FILE: Missing/invalid 'filepath', 'file_path', or 'path' (string).")
                if "content" not in llm_decision: raise ValueError("WRITE_FILE: Missing 'content'.")
                llm_decision['path'] = file_path # Standardize to 'path' internally
            elif action_type == "READ_FILE":
                # *** UPDATED BLOCK START ***
                # Check for 'filepath', 'file_path', then 'path' to handle inconsistency
                file_path_value = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path"))) # <-- Check all three
                if not isinstance(file_path_value, str) or not file_path_value:
                    raise ValueError("READ_FILE: Missing/invalid 'filepath', 'file_path', or 'path' (string).") # <-- Updated error message
                # Standardize to 'path' internally for execution
                llm_decision['path'] = file_path_value # <-- Use the found value
                # *** UPDATED BLOCK END ***
            elif action_type == "REQUEST_RESTART":
                if not isinstance(llm_decision.get("reasoning"), str) or not llm_decision.get("reasoning"): raise ValueError("REQUEST_RESTART: Missing 'reasoning' (string).")
            elif action_type == "UPDATE_GOAL":
                new_goal = llm_decision.get("new_goal")
                if not isinstance(new_goal, dict): raise ValueError("UPDATE_GOAL: Missing/invalid 'new_goal' (dict).")
                if "target" not in new_goal or "description" not in new_goal: raise ValueError("UPDATE_GOAL: 'new_goal' dict must contain 'target' and 'description' keys.")
            elif action_type == "ANALYZE_MEMORY":
                if not isinstance(llm_decision.get("query"), str) or not llm_decision.get("query"): raise ValueError("ANALYZE_MEMORY: Missing/invalid 'query' (string).")
            elif action_type == "MODIFY_CORE_CODE":
                # Use the same 3-key check for file path
                file_path_mod = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_mod, str) or not file_path_mod: raise ValueError("MODIFY_CORE_CODE: Missing/invalid 'filepath', 'file_path', or 'path'.")
                llm_decision['file_path'] = file_path_mod # Standardize to file_path for execution logic consistency

                allowed_mod_types = ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]
                if llm_decision.get("modification_type") not in allowed_mod_types: raise ValueError(f"MODIFY_CORE_CODE: Invalid 'modification_type'. Allowed: {allowed_mod_types}.")
                if not isinstance(llm_decision.get("target_line_content"), str): raise ValueError("MODIFY_CORE_CODE: Missing/invalid 'target_line_content' (string).")
                if llm_decision.get("modification_type") != "DELETE_LINE" and "new_content" not in llm_decision: raise ValueError("MODIFY_CORE_CODE: Missing 'new_content' for non-delete operations.")
                if "verification_hash" not in llm_decision: logger.warning(f"Seed [{cycle_id}]: MODIFY_CORE_CODE action proposed without 'verification_hash'. Applying may be unsafe.")
                elif not isinstance(llm_decision["verification_hash"], str) or len(llm_decision["verification_hash"]) < 8: logger.warning(f"Seed [{cycle_id}]: MODIFY_CORE_CODE has potentially invalid 'verification_hash'.")

            elif action_type == "TEST_CORE_CODE_MODIFICATION":
                file_path_test = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_test, str) or not file_path_test: raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'filepath', 'file_path', or 'path'.")
                llm_decision['file_path'] = file_path_test # Standardize

                allowed_test_types = ["REPLACE_FUNCTION", "REPLACE_METHOD"]
                if llm_decision.get("modification_type") not in allowed_test_types: raise ValueError(f"TEST_CORE_CODE_MODIFICATION: Invalid 'modification_type'. Allowed: {allowed_test_types}.")
                if not isinstance(llm_decision.get("target_name"), str) or not llm_decision["target_name"]: raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'target_name' (string).")
                if "new_logic" not in llm_decision or not isinstance(llm_decision["new_logic"], str): raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'new_logic' (string).")
                test_scenario = llm_decision.get("test_scenario")
                if not isinstance(test_scenario, dict): raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'test_scenario' (dict).")
                if not isinstance(test_scenario.get("test_inputs"), list): raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'test_inputs' (list) in 'test_scenario'.")
                expected_outcome = test_scenario.get("expected_outcome")
                if not isinstance(expected_outcome, dict): raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'expected_outcome' (dict) in 'test_scenario'.")
                if not isinstance(expected_outcome.get("expect_exception", False), bool): raise ValueError("TEST_CORE_CODE_MODIFICATION: Invalid 'expect_exception' (bool) in 'expected_outcome'.")
                if "return_value" not in expected_outcome and not expected_outcome.get("expect_exception"): raise ValueError("TEST_CORE_CODE_MODIFICATION: 'expected_outcome' must contain 'return_value' or set 'expect_exception:true'.")
                if expected_outcome.get("mock_calls") and not isinstance(expected_outcome["mock_calls"], dict): raise ValueError("TEST_CORE_CODE_MODIFICATION: Invalid 'mock_calls' (dict) in 'expected_outcome'.")
            elif action_type == "VERIFY_CORE_CODE_CHANGE":
                file_path_verify = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_verify, str) or not file_path_verify: raise ValueError("VERIFY_CORE_CODE_CHANGE: Missing/invalid 'filepath', 'file_path', or 'path'.")
                llm_decision['file_path'] = file_path_verify # Standardize

                mod_type_v = llm_decision.get("modification_type"); allowed_v_types = ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE", "REPLACE_FUNCTION", "REPLACE_METHOD"]
                if mod_type_v not in allowed_v_types: raise ValueError(f"VERIFY_CORE_CODE_CHANGE: Invalid 'modification_type'. Allowed: {allowed_v_types}.")
                if mod_type_v in ["REPLACE_FUNCTION", "REPLACE_METHOD"]:
                    if not isinstance(llm_decision.get("target_name"), str) or not llm_decision["target_name"]: raise ValueError("VERIFY_CORE_CODE_CHANGE: 'target_name' (string) required for function/method replacement.")
                    if "new_logic" not in llm_decision or not isinstance(llm_decision["new_logic"], str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'new_logic' (string) required for function/method replacement.")
                elif mod_type_v in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]:
                     if not isinstance(llm_decision.get("target_line_content"), str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'target_line_content' (string) required for line operations.")
                     if mod_type_v != "DELETE_LINE" and ("new_content" not in llm_decision or not isinstance(llm_decision["new_content"], str)): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'new_content' (string) required for line replacement/insertion.")
                if "verification_level" in llm_decision and not isinstance(llm_decision["verification_level"], str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'verification_level' must be string.")
            # --- Learning Action Validation ---
            elif action_type=="UPDATE_LEARNING_PARAMETER":
                 if not isinstance(llm_decision.get("parameter_name"), str) or not llm_decision["parameter_name"]: raise ValueError("UPDATE_LEARNING_PARAMETER: Missing/invalid 'parameter_name' (string).")
                 if "new_value" not in llm_decision: raise ValueError("UPDATE_LEARNING_PARAMETER: Missing 'new_value'.")
            elif action_type=="INDUCE_BEHAVIORAL_RULE":
                 if not isinstance(llm_decision.get("trigger_pattern"), dict): raise ValueError("INDUCE_BEHAVIORAL_RULE: Missing/invalid 'trigger_pattern' (dict).")
                 if not isinstance(llm_decision.get("suggested_response"), str) or not llm_decision["suggested_response"]: raise ValueError("INDUCE_BEHAVIORAL_RULE: Missing/invalid 'suggested_response' (string).")
                 if "rule_id" in llm_decision and (not isinstance(llm_decision["rule_id"], str) or not llm_decision["rule_id"].strip()): raise ValueError("INDUCE_BEHAVIORAL_RULE: Optional 'rule_id' must be a non-empty string if provided.")

            return llm_decision

        except (json.JSONDecodeError, ValueError, TypeError) as err:
            logger.error(f"Seed [{cycle_id}] LLM response validation failed: {err}. Raw: '{llm_response_raw[:500]}...'")
            self.memory.log("SEED_LLMError", {"cycle":cycle_id, "stage":"ActionValidation", "error":f"{err}", "raw_snippet":llm_response_raw[:500]}, tags=['Seed','LLM','Error','Action'])
            return {"action_type": "FALLBACK", "reasoning": f"LLM action validation failed ({err})"}


    # --- Action Execution ---
    def _execute_seed_action(self, action_type: str, action_params: Dict, cycle_id: str, current_depth: int) -> ActionResult:
        """ Executes the chosen Seed action, interacting with appropriate services. """
        start_time = time.time()
        exec_res: ActionResult = {"success": False, "message": f"Action '{action_type}' failed/not implemented."}
        # Clean sensitive/large params before logging basic info
        log_params = {k:v for k,v in action_params.items() if k not in ['logic', 'new_logic', 'content', 'new_content', 'test_scenario', 'trigger_pattern']} # Added trigger_pattern
        if action_type == "ANALYZE_MEMORY": log_params = {"query": action_params.get("query")}
        if action_type == "INDUCE_BEHAVIORAL_RULE": log_params = {"suggestion": action_params.get("suggested_response"), "rule_id": action_params.get("rule_id")}
        if action_type == "MODIFY_CORE_CODE": log_params['verification_hash'] = action_params.get("verification_hash", "N/A")[:8] # Log hash prefix


        log_data = {"cycle":cycle_id, "action_params": log_params, "depth": current_depth}
        log_tags = ['Seed', 'Action', action_type]

        try:
            # --- Action Execution Logic ---
            if action_type == "EXECUTE_VM_COMMAND":
                cmd=action_params.get("command")
                logger.info(f"Seed Exec VM: '{cmd}'")
                vm_res=self.vm_service.execute_command(cmd)
                exec_res={'success':vm_res.get('success',False), 'message':f"VM cmd exit={vm_res.get('exit_code','?')}: {vm_res.get('stderr','') or vm_res.get('stdout','')[:100]}", 'details':vm_res, 'reason': vm_res.get('reason')}
            elif action_type == "UPDATE_GOAL":
                succ = self.set_goal(action_params.get("new_goal"))
                exec_res={"success":succ,"message":f"Goal update {'succeeded' if succ else 'failed'}."}
                log_tags.append('Goal')
            elif action_type == "ANALYZE_MEMORY":
                 query=action_params.get("query")
                 analysis_prompt=action_params.get("analysis_prompt","Summarize relevance to current goal and recent errors.") # Default prompt
                 logger.info(f"Seed Analyze Mem Query: '{query}'")
                 if query and isinstance(query, str):
                    mem_context = self.memory.retrieve_context(query=query, limit=10)
                    ctx_for_llm = {k:v for k,v in mem_context.items() if k not in ['recent_episodic', 'current_behavioral_rules']}
                    ctx_for_llm['rule_count'] = len(mem_context.get('current_behavioral_rules', {}))
                    llm_temp = self.memory.get_learning_parameter('llm_query_temperature.value') or 0.5
                    llm_mem_prompt = f"Context: Based on the memory search results for query '{query}', perform the following analysis:\n{analysis_prompt}\n\nMemory Context (Limited):\n```json\n{json.dumps(ctx_for_llm, default=str, indent=2)}\n```\n\nProvide only the analysis result."
                    analysis = self.llm_service.query(llm_mem_prompt, max_tokens=500, temperature=llm_temp)
                    exec_res={"success":True,"message":"Memory analysis complete.", "details":{"query":query, "analysis_result": analysis, "retrieved_context_info": f"Ep:{len(mem_context.get('recent_episodic',[]))} Text:{len(mem_context.get('text_search_results',[]))} Vec:{len(mem_context.get('vector_search_results',[]))} Rules:{ctx_for_llm['rule_count']}"}}
                    self.memory.log(f"SEED_MemAnalysis_{cycle_id}", {"query": query, "analysis": analysis}, tags=['Seed','Memory'])
                 else:
                    exec_res={"success":False,"message":"Requires valid 'query' string."}; log_tags.append('Error')
            elif action_type == "TEST_CORE_CODE_MODIFICATION":
                if not SEED_ENABLE_RUNTIME_CODE_EXECUTION: exec_res = {"success": False, "message": "Runtime code execution (needed for testing) is disabled."}; log_tags.append('Safety')
                elif not ENABLE_CORE_CODE_MODIFICATION: exec_res = {"success": False, "message": "Core code modification/testing is disabled in config."}; log_tags.append('Safety')
                elif current_depth >= CORE_CODE_TEST_MAX_DEPTH: exec_res = {"success": False, "message": f"Core code test aborted: Exceeded max test depth ({CORE_CODE_TEST_MAX_DEPTH})."}; log_tags.extend(['Error', 'Safety', 'Recursion']); logger.error(f"Core Code Test REJECTED: Max Depth {CORE_CODE_TEST_MAX_DEPTH} reached.")
                else: exec_res = self._execute_test_core_code_modification(action_params, cycle_id, current_depth); log_tags.append('CoreTest')
            elif action_type == "VERIFY_CORE_CODE_CHANGE":
                 if not ENABLE_CORE_CODE_MODIFICATION: exec_res = {"success": False, "message": "Core code verification is disabled in config."}; log_tags.append('Safety')
                 else: exec_res = self._execute_verify_core_code_change(action_params, cycle_id); log_tags.append('CoreVerify')
            elif action_type == "MODIFY_CORE_CODE":
                if not ENABLE_CORE_CODE_MODIFICATION: exec_res = {"success": False, "message": "Core code modification is disabled in config."}; log_tags.append('Safety')
                else:
                    verification_hash = action_params.get("verification_hash")
                    if not verification_hash or not isinstance(verification_hash, str):
                         exec_res = {"success": False, "message": "Modification rejected: Missing or invalid 'verification_hash' parameter."}; log_tags.extend(['Error', 'Safety', 'Verification']); logger.warning(f"Core code modification REJECTED: Missing verification_hash.")
                    else:
                         verification_record = self._verified_code_mods.get(verification_hash); is_verified = False; verification_reason = "No recent verification record found for this hash."
                         if verification_record:
                             age = time.time() - verification_record['timestamp'];
                             if age < self._verified_mod_expiry_sec:
                                 if verification_record.get('result',{}).get('success'):
                                      if verification_record.get('params_hash') == verification_hash:
                                           is_verified = True; verification_reason = "Verification passed."
                                      else:
                                           verification_reason = "Verification hash mismatch (internal error?)."
                                 else:
                                      verification_reason = f"Verification failed ({verification_record.get('result',{}).get('message', 'No msg')})."
                             else:
                                 verification_reason = f"Verification expired ({age:.0f}s ago)."

                         if is_verified:
                             logger.info(f"Verification check passed for hash {verification_hash[:8]}. Proceeding with core modification.");
                             # Pass file_path standardized during validation
                             exec_res = self._execute_modify_core_code(action_params, cycle_id); log_tags.append('CoreMod')
                         else:
                             exec_res = {"success": False, "message": f"Core code modification rejected: {verification_reason}"}; log_tags.extend(['Error', 'Safety', 'Verification']);
                             logger.warning(f"Core code modification for hash {verification_hash[:8]} REJECTED: {verification_reason}")
            elif action_type == "READ_FILE":
                 # Use the 'path' standardized during validation
                 path = action_params.get("path"); logger.info(f"Seed Reading file: '{path}'");
                 if path and isinstance(path, str): read_res = self.vm_service.read_file(path); exec_res = read_res if isinstance(read_res, dict) else {"success": False, "message": "Invalid response from vm_service.read_file"}; exec_res['reason'] = read_res.get('reason')
                 else: exec_res={"success": False, "message": "Requires valid 'path'."}; log_tags.append('Error'); exec_res['reason'] = 'invalid_argument'
                 if not exec_res.get("success", True): log_tags.append('Error') # Log error tag if read failed
            elif action_type == "WRITE_FILE":
                 # Use the 'path' standardized during validation
                 path = action_params.get("path"); content = action_params.get("content"); logger.info(f"Seed Writing to file: '{path}' (Content type: {type(content).__name__})")
                 if path and isinstance(path, str) and content is not None:
                      content_str = str(content) if not isinstance(content, str) else content
                      write_res = self.vm_service.write_file(path, content_str); exec_res = write_res if isinstance(write_res, dict) else {"success": False, "message": "Invalid response from vm_service.write_file"}; exec_res['reason'] = write_res.get('reason')
                 else: exec_res={"success": False, "message": "Requires 'path' (string) and 'content'."}; log_tags.append('Error'); exec_res['reason'] = 'invalid_argument'
                 if not exec_res.get("success", True): log_tags.append('Error')
            elif action_type == "REQUEST_RESTART":
                reason = action_params.get("reasoning", "No reason provided.")
                logger.warning(f"Seed requesting self-restart. Reason: {reason}");
                self.memory.log(RESTART_SIGNAL_EVENT_TYPE, {"reason": reason, "cycle": cycle_id}, tags=['Seed', 'Restart', 'Control']);
                exec_res = {"success": True, "message": "Restart signal logged."}
                log_tags.append('Control')
            elif action_type == "NO_OP":
                exec_res = {"success": True, "message": "Executed NO_OP."}
            # --- Learning Action Execution ---
            elif action_type == "UPDATE_LEARNING_PARAMETER":
                 param_name = action_params.get("parameter_name")
                 new_value = action_params.get("new_value")
                 logger.info(f"Seed attempting update learning parameter '{param_name}' to {new_value}")
                 if param_name and new_value is not None:
                      success = self.memory.update_learning_parameter(param_name, new_value)
                      exec_res = {"success": success, "message": f"Learning parameter '{param_name}' update {'succeeded' if success else 'failed or rejected'}."}
                      if success: log_tags.extend(['Parameter', 'Learning'])
                      else: log_tags.append('Error')
                 else:
                      exec_res = {"success": False, "message": "Missing 'parameter_name' or 'new_value'."}; log_tags.append('Error')
            elif action_type == "INDUCE_BEHAVIORAL_RULE":
                 trigger = action_params.get("trigger_pattern")
                 suggestion = action_params.get("suggested_response")
                 rule_id_in = action_params.get("rule_id")
                 logger.info(f"Seed inducing behavioral rule. Suggestion: {suggestion}")
                 if trigger and suggestion:
                      rule_data_to_add = {
                           "trigger_pattern": trigger,
                           "suggested_response": suggestion,
                           "rule_id": rule_id_in,
                           "reasoning": action_params.get("reasoning")
                      }
                      rule_id_out = self.memory.add_behavioral_rule(rule_data_to_add)
                      if rule_id_out:
                           exec_res = {"success": True, "message": f"Behavioral rule '{rule_id_out}' added/updated.", "details": {"rule_id": rule_id_out}}
                           log_tags.extend(['RuleInduction', 'Learning'])
                      else:
                           exec_res = {"success": False, "message": "Failed to add behavioral rule (validation failed?)."}
                           log_tags.append('Error')
                 else:
                      exec_res = {"success": False, "message": "Missing 'trigger_pattern' or 'suggested_response'."}; log_tags.append('Error')

            # --- Fallback for Unknown Actions ---
            else:
                logger.error(f"Seed [{cycle_id}]: Unknown action type '{action_type}'."); exec_res['message']=f"Unknown action type '{action_type}'"; log_tags.append('Error')

            if exec_res.get("success"):
                if "Success" not in log_tags: log_tags.append("Success")
                if "Error" in log_tags: log_tags.remove("Error")
            elif "Error" not in log_tags:
                 log_tags.append("Error")

            log_data["result_msg"] = exec_res.get("message");
            log_data["result_reason"] = exec_res.get("reason")
            self.memory.log(f"SEED_Action_{action_type}", log_data, tags=list(set(log_tags)))

        except Exception as action_exec_error:
            logger.critical(f"Seed CRITICAL Action Exec Error '{action_type}': {action_exec_error}", exc_info=True);
            exec_res = {"success": False, "message": f"Exec Exception: {action_exec_error}", "traceback": traceback.format_exc(), "reason": "internal_error"};
            log_tags.extend(['CriticalError', 'Error']);
            log_tags = list(set(log_tags))
            if "Success" in log_tags: log_tags.remove("Success")
            log_data["result_msg"] = exec_res["message"];
            log_data["result_reason"] = exec_res.get("reason")
            self.memory.log(f"SEED_Action_{action_type}", log_data, tags=log_tags)

        if 'details' not in exec_res: exec_res['details'] = {}
        exec_res['details']['seed_action_duration_sec'] = round(time.time() - start_time, 2)
        return exec_res


    # --- Core Code Modification Helpers ---
    def _execute_modify_core_code(self, action_params: Dict, cycle_id: str) -> ActionResult:
        """ Applies a previously verified core code modification. Uses standardized 'file_path'. """
        # Use standardized 'file_path' from validation step
        file_rel_path = action_params.get("file_path"); mod_type = action_params.get("modification_type"); target_line = action_params.get("target_line_content"); new_content = action_params.get("new_content", "")
        result: ActionResult = {"success": False, "message": "Modification failed.", "details": {}, "reason": "unknown"}
        logger.warning(f"Applying VERIFIED CORE CODE MODIFICATION: File='{file_rel_path}', Type='{mod_type}'")
        try:
            # --- Path Validation ---
            if not file_rel_path or not isinstance(file_rel_path, str): raise ValueError("Missing or invalid 'file_path'.")
            if not mod_type or not isinstance(mod_type, str): raise ValueError("Missing or invalid 'modification_type'.")
            if mod_type != "DELETE_LINE" and new_content is None: raise ValueError(f"Missing 'new_content' for {mod_type}.")
            if mod_type in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"] and not target_line: raise ValueError(f"Missing 'target_line_content' for {mod_type}.")

            full_path = self.project_root.joinpath(file_rel_path).resolve()
            if not full_path.is_relative_to(self.project_root): raise PermissionError("Target path outside project root.")
            relative_to_root = full_path.relative_to(self.project_root)
            if not any(relative_to_root.is_relative_to(allowed) for allowed in CORE_CODE_MODIFICATION_ALLOWED_DIRS): raise PermissionError(f"Target path not in allowed: {CORE_CODE_MODIFICATION_ALLOWED_DIRS}")
            if full_path.name in CORE_CODE_MODIFICATION_DISALLOWED_FILES: raise PermissionError(f"Modification of file '{full_path.name}' disallowed.")
            if not full_path.is_file(): raise FileNotFoundError(f"Target file not found at '{full_path}'.")
            result['details'] = {'validated_path': str(full_path)}

            # --- Read Original ---
            with open(full_path, 'r', encoding='utf-8') as f_read: original_lines = f_read.readlines()

            # --- Apply Modification Logic ---
            target_indices = [i for i, line in enumerate(original_lines) if target_line.strip() in line.strip()]
            if not target_indices: result['reason'] = 'target_not_found'; raise ValueError(f"Target line content not found: '{target_line[:100]}...'")
            if len(target_indices) > 1: result['reason'] = 'target_ambiguous'; logger.warning(f"Target line content ambiguous ({len(target_indices)} matches) in '{full_path}'. Using first match for APPLY.");
            target_idx = target_indices[0]; modified_lines = original_lines[:]

            if mod_type == "REPLACE_LINE":
                 modified_lines[target_idx] = new_content + ('\n' if not new_content.endswith('\n') else '')
            elif mod_type == "INSERT_AFTER_LINE":
                indent = re.match(r"^\s*", original_lines[target_idx]).group(0) if target_idx < len(original_lines) else ""
                lines_to_insert = [(indent + line) for line in new_content.splitlines(True)]
                if lines_to_insert and not lines_to_insert[-1].endswith('\n'): lines_to_insert[-1] += '\n'
                modified_lines[target_idx+1:target_idx+1] = lines_to_insert
            elif mod_type == "DELETE_LINE":
                 del modified_lines[target_idx]
            else: result['reason'] = 'invalid_mod_type'; raise ValueError(f"Unknown modification_type: {mod_type}")

            # --- Validate Syntax ---
            modified_code = "".join(modified_lines)
            try:
                 ast.parse(modified_code, filename=str(full_path))
                 validation_msg = "Final AST Check OK."
                 logger.info(validation_msg)
            except SyntaxError as syn_err:
                 validation_msg = f"Final AST Check FAILED: {syn_err}"
                 result['reason'] = 'syntax_error'; raise ValueError(validation_msg) from syn_err

            # --- Backup and Write ---
            backup_path = None
            try:
                backup_dir = self.project_root.joinpath(CORE_CODE_MODIFICATION_BACKUP_DIR); backup_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S"); backup_filename = f"{full_path.stem}_{timestamp}{full_path.suffix}.bak"; backup_path = backup_dir / backup_filename
                with open(backup_path, 'w', encoding='utf-8') as f_bak: f_bak.writelines(original_lines)
                logger.info(f"Original file backed up to: {backup_path}")
                with open(full_path, 'w', encoding='utf-8') as f_write: f_write.write(modified_code)
                result = {"success": True, "message": f"Core code file '{file_rel_path}' modified successfully. Backup: {backup_path.name}", "details": {"file_path": file_rel_path, "backup_path": str(backup_path)}, "reason": "applied"}
            except Exception as write_err:
                 result['reason'] = 'write_error'; raise IOError(f"Error writing modified file or backup: {write_err}") from write_err

        except (FileNotFoundError, ValueError, IOError, PermissionError, Exception) as err:
            result['message'] = f"Core Code Apply Error: {err}"
            result['details'] = result.get('details', {})
            result['details']["error_type"] = type(err).__name__
            logger.error(f"Core Code Apply FAILED (Reason: {result.get('reason', 'unknown')}): {err}", exc_info=isinstance(err, (IOError, Exception)))

        return result

    def _execute_test_core_code_modification(self, action_params: Dict, cycle_id: str, current_depth: int) -> ActionResult:
        """ Executes an isolated test of proposed new code logic. Uses standardized 'file_path'. """
        # Use standardized 'file_path'
        file_rel_path = action_params.get("file_path"); mod_type = action_params.get("modification_type"); target_name = action_params.get("target_name"); new_logic = action_params.get("new_logic"); test_scenario = action_params.get("test_scenario")
        result: ActionResult = {"success": False, "message": "Test failed.", "details": {}, "reason": "unknown"}
        result['details']['params'] = {k:v for k,v in action_params.items() if k != 'new_logic'}
        logger.info(f"Executing CORE CODE TEST: File='{file_rel_path}', Target='{target_name}'")

        try: # --- Path Validation ---
            if not file_rel_path or not isinstance(file_rel_path, str): raise ValueError("Missing 'file_path'")
            full_path = self.project_root.joinpath(file_rel_path).resolve()
            if not full_path.is_relative_to(self.project_root): raise PermissionError("Target path outside project root.")
            relative_to_root = full_path.relative_to(self.project_root)
            if not any(relative_to_root.is_relative_to(allowed) for allowed in CORE_CODE_MODIFICATION_ALLOWED_DIRS): raise PermissionError(f"Target path not in allowed test/modification dirs: {CORE_CODE_MODIFICATION_ALLOWED_DIRS}")
            if full_path.name in CORE_CODE_MODIFICATION_DISALLOWED_FILES: raise PermissionError(f"Testing involving file '{full_path.name}' disallowed.")
            if not full_path.is_file(): logger.warning(f"Core Code Test: Target file '{full_path}' not found, testing logic in isolation.")
            result['details']['validated_path'] = str(full_path)
        except (ValueError, PermissionError, Exception) as path_err:
            result['message'] = f"Path/Permission Error for Test: {path_err}"; result['reason'] = 'path_permission_error'; logger.error(f"Core Code Test Failed (Path/Perm): {path_err}"); return result

        if not new_logic or not isinstance(new_logic, str): result['message'] = "Missing or invalid 'new_logic'"; result['reason'] = 'invalid_argument'; return result
        if not target_name or not isinstance(target_name, str): result['message'] = "Missing or invalid 'target_name'"; result['reason'] = 'invalid_argument'; return result
        if not test_scenario or not isinstance(test_scenario, dict): result['message'] = "Missing or invalid 'test_scenario'"; result['reason'] = 'invalid_argument'; return result

        # --- Setup Sandbox and Run ---
        def _run_test_in_sandbox(result_queue: queue.Queue):
            sandbox_result = {"output": None, "error": None, "success": False, "message": "", "raw_output": None, "mock_calls": {}, "evaluation_details": {}}
            func_handle = None; mock_self_instance = None; mock_services_dict = {};
            expected_outcome = test_scenario.get("expected_outcome", {})
            test_inputs = test_scenario.get('test_inputs', []); prepared_args = copy.deepcopy(test_inputs);

            try:
                # --- Validate New Logic Syntax ---
                logger.debug("Sandbox: Parsing new logic...");
                parsed_ast = ast.parse(new_logic, filename="<new_logic>"); defined_name = None
                if parsed_ast.body and isinstance(parsed_ast.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)): defined_name = parsed_ast.body[0].name
                if not defined_name: raise SyntaxError("Provided new_logic does not define a top-level function/method.")
                expected_func_name = target_name
                is_method = (mod_type == "REPLACE_METHOD")
                if defined_name != expected_func_name: raise SyntaxError(f"Logic defines '{defined_name}' but target was '{expected_func_name}'.")

                # --- Setup Scope and Mocks ---
                logger.debug("Sandbox: Setting up scope and mocks...");
                isolated_globals = SAFE_EXEC_GLOBALS.copy();
                isolated_locals = {};
                test_logger = logging.getLogger(f"CoreCodeTestSandbox.{target_name}");
                isolated_globals['logger'] = test_logger;
                isolated_globals['MockSelf'] = MockSelf

                mock_services_config = expected_outcome.get('mock_services', {})
                mock_services_dict['memory'] = MockMemorySystem(return_values=mock_services_config.get('memory'))
                mock_services_dict['llm_service'] = MockLLMService(return_values=mock_services_config.get('llm_service'))
                mock_services_dict['vm_service'] = MockVMService(return_values=mock_services_config.get('vm_service'))

                if is_method:
                    logger.debug("Sandbox: Creating MockSelf instance with mocks...");
                    mock_self_instance = MockSelf(mock_services=mock_services_dict)
                    prepared_args.insert(0, mock_self_instance)
                    logger.debug(f"Sandbox: Prepared method call for '{defined_name}' with {len(test_inputs)} user args (+ mock self)...")
                else:
                     logger.debug(f"Sandbox: Prepared function call for '{defined_name}' with {len(test_inputs)} args...")

                # --- Execute the New Logic ---
                logger.debug(f"Sandbox: Executing new logic definition for '{defined_name}'...");
                exec(new_logic, isolated_globals, isolated_locals)
                func_handle = isolated_locals.get(defined_name)
                if not callable(func_handle): raise TypeError(f"Executed logic did not result in callable function/method '{defined_name}'.")

                # --- Invoke the Function/Method ---
                actual_exception = None; actual_output = None
                logger.debug(f"Sandbox: Invoking '{defined_name}'...")
                try: actual_output = func_handle(*prepared_args); sandbox_result['raw_output'] = actual_output
                except Exception as exec_runtime_err: logger.warning(f"Sandbox: Execution raised exception: {exec_runtime_err}"); actual_exception = exec_runtime_err

                # --- Evaluate Outcome ---
                logger.debug("Sandbox: Evaluating outcome..."); eval_msgs = []; passed = True;
                should_raise = expected_outcome.get('expect_exception', False); expected_type_name = expected_outcome.get('exception_type'); expected_msg_contains = expected_outcome.get('exception_message_contains')

                if should_raise:
                    if actual_exception is None: passed = False; eval_msgs.append("FAILED: Expected exception, none raised.")
                    else:
                        eval_msgs.append("PASSED: Expected exception raised.");
                        actual_type_name = type(actual_exception).__name__
                        if expected_type_name and actual_type_name != expected_type_name: passed = False; eval_msgs.append(f"FAILED: Expected type '{expected_type_name}', got '{actual_type_name}'.")
                        elif expected_type_name: eval_msgs.append(f"PASSED: Type matched ('{expected_type_name}').")
                        if expected_msg_contains:
                            actual_msg = str(actual_exception);
                            if expected_msg_contains in actual_msg: eval_msgs.append(f"PASSED: Message contains '{expected_msg_contains}'.")
                            else: passed = False; eval_msgs.append(f"FAILED: Message '{actual_msg}' !contain '{expected_msg_contains}'.")
                    if actual_exception: sandbox_result['error'] = f"{type(actual_exception).__name__}: {actual_exception}"
                else: # No exception expected
                    if actual_exception is not None: passed = False; eval_msgs.append(f"FAILED: No exception expected, got {type(actual_exception).__name__}"); sandbox_result['error'] = f"{type(actual_exception).__name__}: {actual_exception}"
                    else:
                        eval_msgs.append("PASSED: No unexpected exception.");
                        if 'return_value' in expected_outcome:
                            expected_return = expected_outcome['return_value'];
                            if expected_return == "ANY": eval_msgs.append("PASSED: Return value check skipped (ANY).")
                            elif repr(actual_output) == repr(expected_return): eval_msgs.append(f"PASSED: Return value matched.")
                            else: passed = False; eval_msgs.append(f"FAILED: Return value mismatch. Got: {repr(actual_output)[:100]}... Exp: {repr(expected_return)[:100]}...")
                        try: sandbox_result['output'] = json.loads(json.dumps(actual_output, default=str))
                        except Exception: sandbox_result['output'] = repr(actual_output)

                # --- Evaluate Mock Calls ---
                expected_calls = expected_outcome.get('mock_calls', {}); actual_calls_all = {}; evaluation_mock_details = {}
                for service_name, mock_instance in mock_services_dict.items(): actual_calls_all[service_name] = mock_instance.get_calls()

                for service_name, expected_methods in expected_calls.items():
                    evaluation_mock_details[service_name] = {}; actual_service_calls = actual_calls_all.get(service_name, {});
                    if not isinstance(expected_methods, dict): passed = False; eval_msgs.append(f"FAILED: Invalid expected_calls structure for service '{service_name}'."); continue
                    for method_name, expected_call_list in expected_methods.items():
                        actual_method_calls = actual_service_calls.get(method_name, []); actual_count = len(actual_method_calls)
                        if not isinstance(expected_call_list, list):
                            if isinstance(expected_call_list, int):
                                expected_count = expected_call_list
                                if actual_count == expected_count: eval_msgs.append(f"PASSED: Mock call count matched for {service_name}.{method_name} ({expected_count})."); evaluation_mock_details[service_name][method_name] = {"expected_count": expected_count, "actual_count": actual_count, "match": True}
                                else: passed = False; eval_msgs.append(f"FAILED: Mock call count mismatch for {service_name}.{method_name} (Exp {expected_count}, Got {actual_count})."); evaluation_mock_details[service_name][method_name] = {"expected_count": expected_count, "actual_count": actual_count, "match": False}
                            else: passed = False; eval_msgs.append(f"FAILED: Invalid expected_calls value for '{service_name}.{method_name}'. Must be list or int."); evaluation_mock_details[service_name][method_name] = {"error": "Invalid expectation type"}
                        else:
                            expected_count = len(expected_call_list); evaluation_mock_details[service_name][method_name] = {"expected": expected_count, "actual": actual_count, "match": False, "details": []}
                            if actual_count != expected_count: passed = False; eval_msgs.append(f"FAILED: Mock call count mismatch for {service_name}.{method_name} (Exp {expected_count}, Got {actual_count})."); evaluation_mock_details[service_name][method_name]["details"].append("Call count mismatch."); continue
                            calls_match = True
                            for i, (expected_call, actual_call) in enumerate(zip(expected_call_list, actual_method_calls)):
                                if not isinstance(expected_call, dict): calls_match = False; eval_msgs.append(f"FAILED: Invalid expected call structure {i} for {service_name}.{method_name}."); break
                                expected_args = expected_call.get('args', "ANY"); expected_kwargs = expected_call.get('kwargs', "ANY");
                                actual_args = actual_call.get('args', []); actual_kwargs = actual_call.get('kwargs', {});
                                args_match = (expected_args == "ANY" or expected_args == actual_args)
                                kwargs_match = (expected_kwargs == "ANY" or expected_kwargs == actual_kwargs)
                                if not args_match or not kwargs_match: calls_match = False; mismatch_detail = f"Arg mismatch call {i+1}. Exp: args={expected_args}, kwargs={expected_kwargs}. Act: args={actual_args}, kwargs={actual_kwargs}"; eval_msgs.append(f"FAILED: Mock call {mismatch_detail} for {service_name}.{method_name}."); evaluation_mock_details[service_name][method_name]["details"].append(mismatch_detail); break
                            if calls_match and expected_call_list: evaluation_mock_details[service_name][method_name]["match"] = True; eval_msgs.append(f"PASSED: Mock calls matched for {service_name}.{method_name}.")
                            elif not calls_match: passed = False

                for service_name, actual_methods in actual_calls_all.items():
                     if service_name not in expected_calls:
                          if actual_methods: passed=False; eval_msgs.append(f"FAILED: Unexpected calls to service '{service_name}'. Details: {actual_methods}");
                          if service_name not in evaluation_mock_details: evaluation_mock_details[service_name] = {}
                          evaluation_mock_details[service_name]["__UNEXPECTED__"] = actual_methods
                     elif isinstance(expected_calls.get(service_name), dict):
                          for method_name, calls in actual_methods.items():
                              if method_name not in expected_calls[service_name]:
                                   if calls: passed=False; eval_msgs.append(f"FAILED: Unexpected calls to method '{service_name}.{method_name}'. Details: {calls}");
                                   if service_name not in evaluation_mock_details: evaluation_mock_details[service_name] = {}
                                   if method_name not in evaluation_mock_details[service_name]: evaluation_mock_details[service_name][method_name] = {}
                                   evaluation_mock_details[service_name][method_name]["__UNEXPECTED__"] = calls

                sandbox_result['success'] = passed; sandbox_result['message'] = "; ".join(eval_msgs); sandbox_result['mock_calls'] = actual_calls_all; sandbox_result['evaluation_details'] = evaluation_mock_details

            except Exception as eval_err:
                sandbox_result['success'] = False; sandbox_result['error'] = f"SandboxError: {eval_err}"; sandbox_result['message'] = "Test sandbox failed internally during setup or evaluation."; logger.error(f"Core Code Test Failed (Sandbox Setup/Eval): {eval_err}", exc_info=True)

            result_queue.put(sandbox_result)

        # --- Thread Execution and Timeout ---
        result_queue = queue.Queue();
        test_thread = threading.Thread(target=_run_test_in_sandbox, args=(result_queue,))
        test_thread.daemon = True
        test_thread.start()
        timeout_ms = test_scenario.get('max_test_duration_ms', CORE_CODE_TEST_DEFAULT_TIMEOUT_MS)
        timeout_sec = timeout_ms / 1000.0
        test_thread.join(timeout=timeout_sec)

        if test_thread.is_alive():
            result['success'] = False; result['message'] = f"Test failed: Timed out after {timeout_sec:.1f} seconds."; result['details']['timed_out'] = True; result['reason'] = 'timeout'; logger.warning(f"Core Code Test TIMED OUT for target '{target_name}'.")
        else:
            try: sandbox_result = result_queue.get_nowait();
            except queue.Empty: sandbox_result = {"success": False, "message": "Test failed: Result queue empty after thread join.", "error":"QueueEmpty", "evaluation_details":{}}
            except Exception as q_err: sandbox_result = {"success": False, "message": f"Test failed: Error getting result from queue: {q_err}", "error":str(q_err), "evaluation_details":{}}

            result['success'] = sandbox_result.get('success', False);
            result['message'] = sandbox_result.get('message', 'Test completed, result format invalid.');
            result['details'].update(sandbox_result);
            result['details']['timed_out'] = False
            result['reason'] = 'test_passed' if result['success'] else 'test_failed'

        return result

    def _execute_verify_core_code_change(self, action_params: Dict, cycle_id: str) -> ActionResult:
        """ Executes an external verification suite against a proposed code change. Uses standardized 'file_path'. """
        result: ActionResult = {"success": False, "message": "Verification failed.", "details": {}, "reason": "unknown"}
        result['details']['params'] = {k:v for k,v in action_params.items() if k not in ['new_logic', 'new_content']}
        # Use standardized 'file_path'
        file_rel_path = action_params.get("file_path")
        mod_hash = self._hash_modification_params(action_params)
        verification_level = action_params.get("verification_level", "basic")
        logger.info(f"Executing CORE CODE VERIFICATION: File='{file_rel_path}', Hash='{mod_hash[:8]}...', Level='{verification_level}'")

        try: # --- Path Validation ---
            if not file_rel_path or not isinstance(file_rel_path, str): raise ValueError("Missing 'file_path'")
            full_path = self.project_root.joinpath(file_rel_path).resolve()
            if not full_path.is_relative_to(self.project_root): raise PermissionError("Target path outside project root.")
            relative_to_root = full_path.relative_to(self.project_root)
            if not any(relative_to_root.is_relative_to(allowed) for allowed in CORE_CODE_MODIFICATION_ALLOWED_DIRS): raise PermissionError(f"Target path not in allowed test/modification dirs: {CORE_CODE_MODIFICATION_ALLOWED_DIRS}")
            if full_path.name in CORE_CODE_MODIFICATION_DISALLOWED_FILES: raise PermissionError(f"Verification involving file '{full_path.name}' is disallowed.")
            if not full_path.is_file(): raise FileNotFoundError(f"Target file does not exist for verification base: '{full_path}'")
            result['details']['validated_path'] = str(full_path)
        except (ValueError, PermissionError, FileNotFoundError, Exception) as path_err:
            result['message'] = f"Path/Permission Error for Verification: {path_err}"; result['reason'] = 'path_permission_error'; logger.error(f"Core Code Verification Failed (Path/Perm): {path_err}"); return result

        try: # --- Call External Verification Function ---
            verify_success, verify_message, verify_details = run_verification_suite(
                project_root=self.project_root,
                modification_params=action_params,
                verification_level=verification_level
            )
            result['success'] = verify_success; result['message'] = verify_message; result['details'].update(verify_details)
            result['reason'] = 'verify_passed' if verify_success else 'verify_failed'

            self._verified_code_mods[mod_hash] = {
                "timestamp": time.time(),
                "result": {"success": verify_success, "message": verify_message},
                "params": copy.deepcopy(result['details']['params']),
                "params_hash": mod_hash
            }
            logger.info(f"Stored verification record for mod hash {mod_hash[:8]} (Success: {verify_success})")

        except Exception as verify_err:
            result['success'] = False; result['message'] = f"Error during verification process call: {verify_err}"; result['details']["error"] = traceback.format_exc(); result['reason'] = 'verify_error'; logger.error(f"Core Code Verification FAILED (Execution): {verify_err}", exc_info=True)

        return result

    def _hash_modification_params(self, action_params: Dict) -> str:
        """ Creates a consistent hash for modification parameters, including the code change itself. Uses standardized 'file_path'. """
        hash_content = {}
        # Standardize path key for hashing consistency
        path_key = "file_path" # Use the standardized key
        # Include all parameters relevant to defining the change uniquely
        keys_to_hash = [path_key, "modification_type", "target_name", "target_line_content", "new_logic", "new_content"]
        for key in sorted(keys_to_hash):
            if key in action_params: hash_content[key] = action_params[key]
        content_str = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    # --- Fallback Action ---
    def _get_fallback_action(self, reason: str) -> Dict:
        """ Generates a predefined fallback action (e.g., ANALYZE_MEMORY or NO_OP). """
        logger.warning(f"Seed Fallback triggered: {reason}.")
        if "analyze" in reason.lower() or "fallback" in reason.lower():
             return {"action_type": "NO_OP", "reasoning": f"Fallback Action (NO_OP): Previous fallback or analysis failed. Reason: {reason}."}
        else:
             return {"action_type": "ANALYZE_MEMORY", "query": f"Analyze state and recent failures given fallback reason: {reason}", "reasoning": f"Fallback Action: {reason}. Analyzing memory to understand context."}

# --- END OF FILE seed/core.py ---