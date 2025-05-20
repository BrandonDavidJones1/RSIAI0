# RSIAI/seed/core.py
"""
Core strategic component for the RSIAI Seed AGI.
Manages the overall goal, interacts with the environment via VMService,
decides actions (incl. testing, verifying, applying core code modifications,
and adapting internal parameters/rules), analyzes results, and potentially
requests restarts for self-improvement.

Includes initial implementations of internal analysis, hypothesis generation,
and automated learning loops as a starting point for bootstrapping.
"""
import time
import json
import copy
import traceback
import logging
import uuid
import collections # Used for internal analysis
import gc
import numpy as np
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
    VM_SERVICE_ALLOWED_REAL_COMMANDS,
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
from .verification import run_verification_suite, ReplaceFunctionTransformer

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
    def get_learning_parameter(self, name):
        if not name: return self._get_return_value("get_learning_parameter.") or {}
        parts = name.split('.')
        base_value = self._get_return_value(f"get_learning_parameter.{name}")
        if base_value is not None: return base_value
        if len(parts) == 2 and parts[1] == 'value':
            cat_val = self._get_return_value(f"get_learning_parameter.{parts[0]}")
            if isinstance(cat_val, dict): return cat_val.get('value', 0.5)
        return 0.5
    def update_learning_parameter(self, name, value):
        self._record_call("update_learning_parameter", (name, value), {})
        return True
    def add_behavioral_rule(self, rule_data):
        self._record_call("add_behavioral_rule", (rule_data,), {})
        return f"mock_rule_{uuid.uuid4().hex[:4]}"
    def get_behavioral_rules(self):
        return self._get_return_value("get_behavioral_rules") or {}
    def update_rule_trigger_stats(self, rule_id):
        self._record_call("update_rule_trigger_stats", (rule_id,), {})
    def find_lifelong_by_criteria(self, filter_function, limit=None, newest_first=False):
        self._record_call("find_lifelong_by_criteria", (filter_function,), {"limit": limit, "newest_first": newest_first})
        if "SEED_Evaluation" in repr(filter_function):
            return self._get_return_value("find_lifelong_by_criteria.evals") or []
        elif "Error" in repr(filter_function):
             return self._get_return_value("find_lifelong_by_criteria.errors") or []
        else:
             return self._get_return_value("find_lifelong_by_criteria.default") or []


class MockLLMService(MockCoreService):
    def __init__(self, return_values=None):
        default_returns = {'query': '{"action_type": "NO_OP", "reasoning": "Mock LLM Response"}'}
        if return_values: default_returns.update(return_values)
        super().__init__("llm_service", default_returns)

class MockVMService(MockCoreService):
     def __init__(self, return_values=None):
         default_returns = {'execute_command': {'success': False, 'stdout':'', 'stderr':'Mock VM Error', 'exit_code':1}}
         if return_values: default_returns.update(return_values)
         super().__init__("vm_service", default_returns)

class MockSelf:
    def __init__(self, mock_services=None, **kwargs):
        self._mock_attrs = kwargs
        self.logger = MockCoreService(service_name="self_logger")
        self._mock_attrs['logger'] = self.logger 
        self._mock_services = mock_services if isinstance(mock_services, dict) else {}
        for name, service in self._mock_services.items():
            setattr(self, name, service)
    def __getattr__(self, name):
        if name == 'memory':
             if 'memory' in self._mock_services:
                 return self._mock_services['memory']
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in self._mock_attrs:
                return self._mock_attrs[name]
        raise AttributeError(f"'MockSelf' has no attribute '{name}'. Known attrs: {list(self.__dict__.keys()) + list(self._mock_attrs.keys())}")
    def __setattr__(self, name, value):
        if name in ["_mock_attrs", "_mock_services", "logger"] or name in self.__dict__:
             super().__setattr__(name, value)
        else:
            self._mock_attrs[name] = value
    def __repr__(self):
        return f"<MockSelf attrs={list(self._mock_attrs.keys())} services={list(self._mock_services.keys())}>"

# --- End Mock Classes ---


class Seed_Core:
    """
    Core strategic component for the Seed AGI. Manages goals, interacts with the environment,
    adapts internal state, and drives RSI via core code modification, testing, and verification.
    Includes initial implementations of internal analysis/hypothesis/learning functions.
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

        self.current_goal: Dict = {}
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
            "SET_VM_MODE",
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
            "NO_OP"
        ]
        self.current_sensory_input: Optional[RefinedInput] = None
        self.cycle_count = 0
        self.project_root = pathlib.Path(__file__).resolve().parent.parent
        self._verified_code_mods: Dict[str, Dict] = {}
        self._verified_mod_expiry_sec = 3600
        self._recent_eval_scores: collections.deque = collections.deque(maxlen=5)

        logger.info(f"Project root identified as: {self.project_root}")
        logger.info("Seed Core Initialized with internal analysis/hypothesis/learning stubs.")

    def set_initial_state(self, goal: Dict):
        if isinstance(goal, dict) and 'target' in goal:
            self.current_goal = copy.deepcopy(goal)
            logger.info(f"Seed Initial State Set - Goal: {self.current_goal.get('description', goal.get('target'))}")
            self.memory.log("seed_initial_state_set", {"goal": self.current_goal}, tags=['Seed', 'Config', 'Init', 'Goal'])
        else:
             logger.error(f"Invalid initial goal format provided: {goal}. Setting empty goal.")
             self.current_goal = {}

    def set_goal(self, new_goal: Dict) -> bool:
        if isinstance(new_goal, dict) and 'target' in new_goal and 'description' in new_goal:
            old_goal = copy.deepcopy(self.current_goal)
            self.current_goal = copy.deepcopy(new_goal)
            logger.info(f"Seed Goal updated: {self.current_goal.get('description')}.")
            self.memory.log("seed_goal_set", {"old_goal": old_goal, "new_goal": self.current_goal}, tags=['Seed', 'Goal'])
            return True
        else:
            logger.error(f"Invalid goal format received for update: {new_goal}. Requires 'target' and 'description'.")
            return False

    def _match_dict_pattern(self, pattern: Dict, target: Dict) -> bool:
         if not isinstance(pattern, dict) or not isinstance(target, dict):
             return False
         for key, p_value in pattern.items():
              parts = key.split('.')
              current_target_level = target
              key_found = True
              try:
                  for i, part in enumerate(parts):
                      if isinstance(current_target_level, dict) and part in current_target_level:
                          if i == len(parts) - 1:
                              if current_target_level[part] != p_value:
                                  key_found = False; break
                          else:
                              current_target_level = current_target_level[part]
                      elif isinstance(current_target_level, list) and part.isdigit():
                          idx = int(part)
                          if 0 <= idx < len(current_target_level):
                              if i == len(parts) - 1:
                                  if current_target_level[idx] != p_value:
                                      key_found = False; break
                              else:
                                  current_target_level = current_target_level[idx]
                          else:
                              key_found = False; break
                      else:
                          key_found = False; break
              except (KeyError, IndexError, TypeError):
                  key_found = False
              if not key_found:
                  return False
         return True

    def _check_behavioral_rules(self, context_snapshot: Dict) -> List[Dict]:
        triggered_rules_info = []
        rules = self.memory.get_behavioral_rules()
        if not rules: return []
        logger.debug(f"Checking {len(rules)} behavioral rules against context snapshot...")
        match_context = {
            "goal": context_snapshot.get('seedGoal'),
            "sensory": context_snapshot.get('seedSensory'),
            "vm_state": context_snapshot.get('vm_snapshot'),
        }
        for rule_id, rule_data in rules.items():
            try:
                if self._match_dict_pattern(rule_data['trigger_pattern'], match_context):
                    logger.info(f"Behavioral Rule Triggered: '{rule_id}' - Suggestion: {rule_data.get('suggested_response', 'N/A')}")
                    triggered_rules_info.append(copy.deepcopy(rule_data))
                    self.memory.update_rule_trigger_stats(rule_id)
            except Exception as e:
                 logger.error(f"Error matching rule '{rule_id}': {e}", exc_info=True)
        return triggered_rules_info

    def _analyze_memory_patterns(self, history_limit: int = 50) -> Dict:
        analysis_results = {
            "action_success_rates": {},
            "common_errors": [],
            "error": None
        }
        logger.debug(f"Running internal memory pattern analysis (limit={history_limit})...")
        if not hasattr(self.memory, 'find_lifelong_by_criteria'):
             analysis_results['error'] = "MemorySystem lacks 'find_lifelong_by_criteria' method."
             logger.error(analysis_results['error'])
             return analysis_results

        try:
            evals = self.memory.find_lifelong_by_criteria(
                lambda e: e.get('key','').startswith("SEED_Evaluation"),
                limit=history_limit,
                newest_first=True
            )
            actions_summary = collections.defaultdict(lambda: {'count': 0, 'success_sum': 0.0})

            for eval_entry in evals:
                data = eval_entry.get('data', {})
                action_summary_str = data.get('action_summary') 
                if not isinstance(action_summary_str, str):
                    logger.debug(f"Skipping eval entry due to missing or non-string 'action_summary': {eval_entry.get('key')}")
                    continue
                
                action_type_parts = action_summary_str.split(':', 1)
                action_type = action_type_parts[0].strip()

                if not action_type: 
                    logger.debug(f"Skipping eval entry due to empty action_type from 'action_summary': {action_summary_str}")
                    continue

                success_score_raw = data.get('overall_success')
                if success_score_raw is None:
                    logger.debug(f"Skipping eval entry {eval_entry.get('key')} for action {action_type} due to missing 'overall_success'.")
                    continue
                
                try:
                    success_score = float(success_score_raw)
                except (ValueError, TypeError):
                    logger.debug(f"Skipping eval entry {eval_entry.get('key')} for action {action_type} due to non-float 'overall_success': {success_score_raw}")
                    continue

                actions_summary[action_type]['count'] += 1
                actions_summary[action_type]['success_sum'] += success_score
            
            logger.debug(f"Intermediate actions_summary for rates: {dict(actions_summary)}") 

            for act_type, summary in actions_summary.items():
                if summary['count'] > 0:
                    avg_success = float(round(summary['success_sum'] / summary['count'], 3))
                    analysis_results["action_success_rates"][act_type] = {
                        "avg_success": avg_success, "count": summary['count']
                    }

            errors = self.memory.find_lifelong_by_criteria(
                lambda e: ('Error' in e.get('tags', []) or 'Critical' in e.get('tags', [])) and \
                          (e.get('key','').startswith("SEED_Action_") or e.get('key','').startswith("SEED_LLMError") or e.get('key','').startswith("SEED_CycleCriticalError")),
                limit=history_limit,
                newest_first=True
            )
            error_identifiers = collections.Counter()
            for e in errors:
                data = e.get('data', {})
                reason = data.get('result_reason') or data.get('reason')
                if reason: error_identifiers[reason] += 1; continue
                error_msg = data.get('error')
                if error_msg: error_identifiers[f"Error: {str(error_msg)[:50]}..."] += 1; continue
                result_msg = data.get('result_msg') or data.get('message')
                if result_msg: error_identifiers[f"Msg: {str(result_msg)[:50]}..."] += 1; continue
            analysis_results["common_errors"] = [(reason, count) for reason, count in error_identifiers.most_common(3) if reason]

        except AttributeError as ae:
             analysis_results['error'] = f"AttributeError during analysis (likely missing memory method): {ae}"
             logger.error(analysis_results['error'], exc_info=True)
        except Exception as e:
            analysis_results['error'] = f"Unexpected error during memory analysis: {e}"
            logger.error(analysis_results['error'], exc_info=True)

        logger.debug(f"Internal memory pattern analysis completed. Rates: {analysis_results.get('action_success_rates')}") 
        self.memory.log("SEED_InternalAnalysis", analysis_results, tags=["Seed", "Analysis", "InternalState"])
        return analysis_results

    def _generate_failure_hypotheses(self, last_action: Optional[Dict], last_eval: Optional[Dict]) -> List[str]:
        hypotheses = []
        if not last_action or not last_eval:
            return hypotheses
        if last_eval.get('overall_success', 1.0) < 0.5:
            action_type = last_action.get('action_type', 'Unknown')
            reason = last_eval.get('details', {}).get('reason') or \
                     last_eval.get('details', {}).get('details',{}).get('reason')
            logger.debug(f"Generating failure hypotheses for action '{action_type}' with reason '{reason}'")
            if not reason:
                 hypotheses.append(f"Hypothesis ({action_type}): Failure reason unclear from evaluation details. Need deeper investigation of execution logs or state change.")
            elif reason == 'file_not_found':
                 path = last_action.get('path', last_action.get('file_path', last_action.get('filepath', 'N/A')))
                 hypotheses.append(f"Hypothesis ({action_type}): Target path '{path}' might be incorrect, misspelled, or require different relative/absolute structure.")
                 hypotheses.append(f"Hypothesis ({action_type}): The file/directory might not have been created yet or was deleted.")
            elif reason == 'permission_denied':
                 hypotheses.append(f"Hypothesis ({action_type}): Action lacked necessary permissions on the target or its parent directory.")
                 hypotheses.append(f"Hypothesis ({action_type}): System context (user/group) needs adjustment or target permissions need modification.")
            elif reason == 'timeout':
                 hypotheses.append(f"Hypothesis ({action_type}): Action took too long. Maybe the command/operation is too complex or the environment is slow/unresponsive.")
            elif reason == 'invalid_argument' or reason == 'parse_error':
                 hypotheses.append(f"Hypothesis ({action_type}): Parameters provided to the action were invalid, missing, or incorrectly formatted ({last_action.get('command') or last_action.get('path') or ''}).")
            elif reason == 'safety_violation':
                 hypotheses.append(f"Hypothesis ({action_type}): Action involved a disallowed command or path modification.")
            elif reason == 'is_directory' or reason == 'is_not_directory':
                 hypotheses.append(f"Hypothesis ({action_type}): Action expected a file but found a directory, or vice versa.")
            else:
                 hypotheses.append(f"Hypothesis ({action_type}): Failure reason '{reason}'. Potential causes: unexpected state, resource limits, or internal action logic error.")
        if hypotheses:
            logger.info(f"Generated Failure Hypotheses: {hypotheses}")
            self.memory.log("SEED_FailureHypotheses", {"hypotheses": hypotheses, "failed_action": last_action, "evaluation": last_eval}, tags=["Seed", "Hypothesis", "Error"])
        return hypotheses

    def _propose_improvement_hypotheses(self, analysis: Dict) -> List[str]:
        hypotheses = []
        if not analysis or not isinstance(analysis, dict):
            return hypotheses
        rates = analysis.get("action_success_rates", {})
        errors = analysis.get("common_errors", [])
        for action_type, stats in rates.items():
            if stats.get('count', 0) >= 3 and stats.get('avg_success', 1.0) < 0.5:
                 hypotheses.append(f"Hypothesis (Improvement): Action '{action_type}' has low success ({stats['avg_success']:.2f} over {stats['count']} tries). Analyze specific failures or parameters.")
        if errors:
            top_error_reason = errors[0][0]
            hypotheses.append(f"Hypothesis (Improvement): The most common error reason is '{top_error_reason}'. Investigate root cause and potentially modify related action logic or preconditions.")
        if analysis.get("error"):
             hypotheses.append(f"Hypothesis (Improvement): Internal analysis function itself failed ('{analysis['error']}'). Needs debugging.")
        if hypotheses:
            logger.info(f"Generated Improvement Hypotheses: {hypotheses}")
            self.memory.log("SEED_ImprovementHypotheses", {"hypotheses": hypotheses, "triggering_analysis": analysis}, tags=["Seed", "Hypothesis", "Improvement"])
        return hypotheses

    def _perform_automated_learning(self, last_evaluation: Dict):
        if not last_evaluation or 'overall_success' not in last_evaluation:
            logger.debug("Skipping automated learning: Invalid evaluation data.")
            return
        try:
            current_score = float(last_evaluation['overall_success'])
            self._recent_eval_scores.append(current_score)
            if len(self._recent_eval_scores) == self._recent_eval_scores.maxlen:
                avg_score = sum(self._recent_eval_scores) / len(self._recent_eval_scores)
                temp_param_name = "llm_query_temperature.value"
                current_temp = self.memory.get_learning_parameter(temp_param_name)
                if current_temp is None:
                    logger.warning("Could not retrieve current LLM temperature for auto-adjustment.")
                    return
                new_temp = current_temp
                temp_change = 0.05
                if avg_score < 0.3:
                    new_temp += temp_change
                    logger.info(f"Automated Learning: Low avg success ({avg_score:.2f}), increasing LLM temp towards {new_temp:.2f}.")
                elif avg_score > 0.8:
                    new_temp -= temp_change
                    logger.info(f"Automated Learning: High avg success ({avg_score:.2f}), decreasing LLM temp towards {new_temp:.2f}.")
                else:
                    logger.debug(f"Automated Learning: Avg success ({avg_score:.2f}) in nominal range, no temp change.")
                if abs(new_temp - current_temp) > 0.001:
                    self.memory.update_learning_parameter(temp_param_name, new_temp)
        except Exception as e:
            logger.error(f"Error during automated learning step: {e}", exc_info=True)
            self.memory.log("SEED_AutoLearnError", {"error": str(e)}, tags=["Seed", "Learning", "Error"])

    def run_strategic_cycle(self):
        self.cycle_count += 1
        cycle_id = f"Seed_{self.cycle_count:06d}"
        logger.info(f"--- Starting Seed Strategic Cycle [{cycle_id}] ---")
        start_time = time.time()
        action_to_execute: Optional[Dict] = None
        execution_result: Optional[ActionResult] = None
        evaluation: Optional[Dict] = None
        vm_state_snapshot: Optional[Dict] = None
        pre_action_snapshot: Optional[Dict] = None
        internal_analysis: Dict = {}
        improvement_hypotheses: List[str] = []
        failure_hypotheses: List[str] = []
        triggered_rules_info: List[Dict] = []

        try:
            logger.debug(f"Seed [{cycle_id}]: Sensing VM state...")
            vm_state_snapshot = self.vm_service.get_state(target_path_hint=self.current_goal.get('path'))
            if not vm_state_snapshot or vm_state_snapshot.get("error"):
                 error_msg = vm_state_snapshot.get("error", "VM Service returned invalid state") if vm_state_snapshot else "VM Service returned None state"
                 raise RuntimeError(f"VM Service failed during state retrieval: {error_msg}")
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
            logger.info(f"Seed [{cycle_id}]: Performing internal analysis & hypothesis generation...")
            try:
                 internal_analysis = self._analyze_memory_patterns()
                 improvement_hypotheses = self._propose_improvement_hypotheses(internal_analysis)
                 triggered_rules_info = self._check_behavioral_rules(pre_action_snapshot)
            except Exception as analysis_err:
                 logger.error(f"Seed [{cycle_id}]: Internal analysis/hypothesis step failed: {analysis_err}", exc_info=True)
                 self.memory.log("SEED_InternalAnalysisError", {"error": str(analysis_err)}, tags=["Seed", "Analysis", "Error"])
            logger.info(f"Seed [{cycle_id}]: Querying LLM (or user) for next action...")
            llm_temp = self.memory.get_learning_parameter('llm_query_temperature.value')
            if not isinstance(llm_temp, (float, int)):
                 logger.warning(f"Invalid LLM temperature retrieved ({llm_temp}), using default 0.5.")
                 llm_temp = 0.5
            llm_prompt = self._build_llm_prompt(
                self.current_sensory_input,
                internal_analysis,
                improvement_hypotheses,
                triggered_rules_info
            )
            llm_response_raw = self.llm_service.query(
                llm_prompt,
                temperature=llm_temp,
                max_tokens=LLM_DEFAULT_MAX_TOKENS
            )
            llm_decision = self._validate_direct_action_llm_response(llm_response_raw, self.available_actions, cycle_id)
            action_to_execute = llm_decision
            action_source = "LLM_Direct" if not LLM_MANUAL_MODE else "Manual_Input"
            rule_mode = self.memory.get_learning_parameter('rule_application_mode.value') or "log_suggestion"
            if rule_mode == "pre_llm_filter" and triggered_rules_info:
                 logger.warning(f"Rule application mode '{rule_mode}' not fully implemented. Using LLM/Manual action.")
            if not action_to_execute or action_to_execute.get("action_type") == "FALLBACK":
                fallback_reason = action_to_execute.get("reasoning", "LLM response validation failed.") if action_to_execute else "LLM decision was None."
                logger.error(f"Seed [{cycle_id}]: LLM/Manual decision invalid or fallback needed. Reason: {fallback_reason}. Using fallback action.")
                action_to_execute = self._get_fallback_action(fallback_reason)
                action_source = "Fallback"
            logger.info(f"Seed [{cycle_id}]: Action Decided ({action_source}): {action_to_execute.get('action_type')}")
            log_decision = copy.deepcopy(action_to_execute)
            log_decision['action_source'] = action_source
            self.memory.log(f"SEED_Decision_{cycle_id}", log_decision, tags=['Seed', 'Decision', action_source])
            action_type = action_to_execute.get("action_type", "NO_OP")
            execution_result = self._execute_seed_action(action_type, action_to_execute, cycle_id, current_depth=0)
            logger.debug(f"Seed [{cycle_id}]: Re-sensing VM state after action...")
            post_action_vm_state = self.vm_service.get_state(target_path_hint=self.current_goal.get('path'))
            post_action_sensory = self.sensory_refiner.refine(post_action_vm_state)
            if not post_action_sensory:
                 logger.warning(f"Seed [{cycle_id}]: Failed to refine post-action sensory input. Evaluation might be inaccurate.")
            logger.debug(f"Seed [{cycle_id}]: Evaluating action success...")
            if pre_action_snapshot:
                 eval_weights_config = self.memory.get_learning_parameter('evaluation_weights')
                 current_eval_weights = {}
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
                 if not execution_result.get("success", True):
                     failure_hypotheses = self._generate_failure_hypotheses(action_to_execute, evaluation)
            else:
                 logger.error(f"Seed [{cycle_id}]: Cannot evaluate - pre_action_snapshot was not captured.")
                 evaluation = {"error": "Missing pre_action_snapshot for evaluation."}
            if post_action_sensory:
                self.current_sensory_input = post_action_sensory
            logger.debug(f"Seed [{cycle_id}]: Performing automated learning...")
            if evaluation and not evaluation.get("error"):
                 self._perform_automated_learning(evaluation)
            else:
                 logger.warning("Skipping automated learning due to evaluation error or missing evaluation.")
        except Exception as cycle_err:
            logger.critical(f"!!! Seed Cycle [{cycle_id}] CRITICAL ERROR: {cycle_err}", exc_info=True)
            self.memory.log("SEED_CycleCriticalError", {"cycle": cycle_id, "error": str(cycle_err), "traceback": traceback.format_exc()}, tags=['Seed', 'Critical', 'Error'])
        finally:
            duration = time.time() - start_time
            logger.info(f"--- Finished Seed Strategic Cycle [{cycle_id}] (Duration: {duration:.2f}s) ---")
            gc.collect()

    def _build_llm_prompt(self,
                          sensory_input: Optional[RefinedInput],
                          internal_analysis: Dict,
                          improvement_hypotheses: List[str],
                          triggered_rules: List[Dict]) -> str:
        prompt = "## Seed Strategic Cycle Input\n\n"
        prompt += f"**Current Goal:**\n```json\n{json.dumps(self.current_goal, indent=2)}\n```\n\n"
        prompt += f"**Constraints:**\n- {'; '.join(self.constraints)}\n\n"
        prompt += "**Internal Analysis Summary:**\n"
        if internal_analysis.get("error"):
            prompt += f"- Analysis Error: {internal_analysis['error']}\n"
        else:
            rates = internal_analysis.get("action_success_rates", {})
            errors = internal_analysis.get("common_errors", [])
            if rates: prompt += f"- Recent Action Success Rates (Avg/Count): {json.dumps(rates)}\n"
            else: prompt += "- No recent action success data.\n"
            if errors: prompt += f"- Most Common Recent Errors (Reason/Count): {json.dumps(errors)}\n"
            else: prompt += "- No common errors identified recently.\n"
        prompt += "\n"
        if improvement_hypotheses:
             prompt += "**Suggested Improvement Hypotheses (from internal analysis):**\n"
             for hyp in improvement_hypotheses: prompt += f"- {hyp}\n"
             prompt += "\n"
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
            recent_evals_mem = self.memory.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Evaluation"), limit=3, newest_first=True)
            simplified_evals = [{'action': e['data'].get('action_summary'), 'success': round(e['data'].get('overall_success',0),2), 'msg': e['data'].get('message')} for e in recent_evals_mem if e.get('data')]
            if simplified_evals:
                 prompt += f"**Recent Evaluations (Summary - Max 3):**\n```json\n{json.dumps(simplified_evals, indent=2, default=str)}\n```\n\n"
        except Exception as mem_e: logger.warning(f"Failed retrieve/simplify recent evals for prompt: {mem_e}")
        valid_verified_count = 0
        if self._verified_code_mods:
            prompt += "**Recently Verified Code Modifications (Ready for MODIFY_CORE_CODE):**\n"
            now = time.time()
            sorted_hashes = sorted(self._verified_code_mods.keys(), key=lambda h: self._verified_code_mods[h].get('timestamp', 0), reverse=True)
            shown_count = 0
            for mod_hash in sorted_hashes:
                 v_data = self._verified_code_mods[mod_hash]
                 if v_data.get('result',{}).get('success') and now - v_data.get('timestamp',0) < self._verified_mod_expiry_sec:
                     if shown_count >= 3:
                          prompt += "- (...more verified mods available...)\n"
                          break
                     params = v_data.get('params', {})
                     target_info = params.get('target_name') or params.get('target_line_content','?')[:30]+"..."
                     prompt += f"- Hash `{mod_hash[:8]}`: File='{params.get('file_path', '?')}', Target='{target_info}', Type='{params.get('modification_type', '?')}'\n"
                     valid_verified_count += 1
                     shown_count += 1
            if valid_verified_count == 0:
                 prompt += "- None\n"
            prompt += "\n"
        prompt += f"**Available Actions:** {self.available_actions}\n\n"
        prompt += "**Instruction:** Refer to the initial system prompt (detailing action parameters and workflows, and the bootstrapping objective). Based on the **goal**, **internal analysis**, **hypotheses**, **environment state**, and **recent evaluations**, decide the single best next action. Prioritize actions that implement/improve internal capabilities (analysis, hypothesis, planning, learning) or test/verify/apply such changes. Ensure the action helps achieve the primary objective of enabling Seed self-sufficiency.\n"
        prompt += "Provide reasoning in `\"reasoning\"`, referencing the analysis/hypotheses if relevant. Respond ONLY with the chosen JSON object on a single line."
        return prompt

    def _validate_direct_action_llm_response(self, llm_response_raw: Optional[str], available_actions: List[str], cycle_id: str) -> Dict:
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
            
            if action_type == "EXECUTE_VM_COMMAND":
                if not isinstance(llm_decision.get("command"), str) or not llm_decision["command"]: raise ValueError("EXECUTE_VM_COMMAND: Missing/invalid 'command' (string).")
            elif action_type == "WRITE_FILE":
                file_path = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path, str) or not file_path: raise ValueError("WRITE_FILE: Missing/invalid 'filepath', 'file_path', or 'path' (string).")
                if "content" not in llm_decision: raise ValueError("WRITE_FILE: Missing 'content'.")
                llm_decision['path'] = file_path 
            elif action_type == "READ_FILE":
                file_path_value = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_value, str) or not file_path_value:
                    raise ValueError("READ_FILE: Missing/invalid 'filepath', 'file_path', or 'path' (string).")
                llm_decision['path'] = file_path_value 
            elif action_type == "REQUEST_RESTART":
                if not isinstance(llm_decision.get("reasoning"), str) or not llm_decision.get("reasoning"): raise ValueError("REQUEST_RESTART: Missing 'reasoning' (string).")
            elif action_type == "UPDATE_GOAL":
                new_goal = llm_decision.get("new_goal")
                if not isinstance(new_goal, dict): raise ValueError("UPDATE_GOAL: Missing/invalid 'new_goal' (dict).")
                if "target" not in new_goal or "description" not in new_goal: raise ValueError("UPDATE_GOAL: 'new_goal' dict must contain 'target' and 'description' keys.")
            elif action_type == "ANALYZE_MEMORY":
                if not isinstance(llm_decision.get("query"), str) or not llm_decision.get("query"): raise ValueError("ANALYZE_MEMORY: Missing/invalid 'query' (string).")
            elif action_type == "MODIFY_CORE_CODE":
                file_path_mod = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_mod, str) or not file_path_mod: raise ValueError("MODIFY_CORE_CODE: Missing/invalid 'filepath', 'file_path', or 'path'.")
                llm_decision['file_path'] = file_path_mod 
                allowed_mod_types = ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]
                if llm_decision.get("modification_type") not in allowed_mod_types: raise ValueError(f"MODIFY_CORE_CODE: Invalid 'modification_type'. Allowed: {allowed_mod_types}.")
                if not isinstance(llm_decision.get("target_line_content"), str): raise ValueError("MODIFY_CORE_CODE: Missing/invalid 'target_line_content' (string).")
                if llm_decision.get("modification_type") != "DELETE_LINE" and "new_content" not in llm_decision: raise ValueError("MODIFY_CORE_CODE: Missing 'new_content' for non-delete operations.")
                if "verification_hash" not in llm_decision: logger.warning(f"Seed [{cycle_id}]: MODIFY_CORE_CODE action proposed without 'verification_hash'. Applying may be unsafe.")
                elif llm_decision.get("verification_hash") is not None and (not isinstance(llm_decision["verification_hash"], str) or len(llm_decision["verification_hash"]) < 8):
                      logger.warning(f"Seed [{cycle_id}]: MODIFY_CORE_CODE has potentially invalid 'verification_hash'.")
            elif action_type == "TEST_CORE_CODE_MODIFICATION":
                file_path_test = llm_decision.get("filepath", llm_decision.get("file_path", llm_decision.get("path")))
                if not isinstance(file_path_test, str) or not file_path_test: raise ValueError("TEST_CORE_CODE_MODIFICATION: Missing/invalid 'filepath', 'file_path', or 'path'.")
                llm_decision['file_path'] = file_path_test 
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
                llm_decision['file_path'] = file_path_verify 
                mod_type_v = llm_decision.get("modification_type"); allowed_v_types = ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE", "REPLACE_FUNCTION", "REPLACE_METHOD"]
                if mod_type_v not in allowed_v_types: raise ValueError(f"VERIFY_CORE_CODE_CHANGE: Invalid 'modification_type'. Allowed: {allowed_v_types}.")
                if mod_type_v in ["REPLACE_FUNCTION", "REPLACE_METHOD"]:
                    if not isinstance(llm_decision.get("target_name"), str) or not llm_decision["target_name"]: raise ValueError("VERIFY_CORE_CODE_CHANGE: 'target_name' (string) required for function/method replacement.")
                    if "new_logic" not in llm_decision or not isinstance(llm_decision["new_logic"], str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'new_logic' (string) required for function/method replacement.")
                elif mod_type_v in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]:
                     if not isinstance(llm_decision.get("target_line_content"), str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'target_line_content' (string) required for line operations.")
                     if mod_type_v != "DELETE_LINE" and ("new_content" not in llm_decision or not isinstance(llm_decision["new_content"], str)): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'new_content' (string) required for line replacement/insertion.")
                if "verification_level" in llm_decision and not isinstance(llm_decision["verification_level"], str): raise ValueError("VERIFY_CORE_CODE_CHANGE: 'verification_level' must be string.")
            elif action_type=="UPDATE_LEARNING_PARAMETER":
                 if not isinstance(llm_decision.get("parameter_name"), str) or not llm_decision["parameter_name"]: raise ValueError("UPDATE_LEARNING_PARAMETER: Missing/invalid 'parameter_name' (string).")
                 if "new_value" not in llm_decision: raise ValueError("UPDATE_LEARNING_PARAMETER: Missing 'new_value'.")
            elif action_type=="INDUCE_BEHAVIORAL_RULE":
                 if not isinstance(llm_decision.get("trigger_pattern"), dict): raise ValueError("INDUCE_BEHAVIORAL_RULE: Missing/invalid 'trigger_pattern' (dict).")
                 if not isinstance(llm_decision.get("suggested_response"), str) or not llm_decision["suggested_response"]: raise ValueError("INDUCE_BEHAVIORAL_RULE: Missing/invalid 'suggested_response' (string).")
                 if "rule_id" in llm_decision and (not isinstance(llm_decision["rule_id"], str) or not llm_decision["rule_id"].strip()): raise ValueError("INDUCE_BEHAVIORAL_RULE: Optional 'rule_id' must be a non-empty string if provided.")
            elif action_type == "SET_VM_MODE":
                mode_value = llm_decision.get("mode")
                if mode_value not in ["simulation", "real"]:
                    raise ValueError("SET_VM_MODE: Invalid 'mode' parameter. Must be 'simulation' or 'real'.")
                if not isinstance(llm_decision.get("reasoning"), str) or not llm_decision.get("reasoning"): 
                    raise ValueError("SET_VM_MODE: Missing 'reasoning' (string).")
            return llm_decision
        except (json.JSONDecodeError, ValueError, TypeError) as err:
            logger.error(f"Seed [{cycle_id}] LLM response validation failed: {err}. Raw: '{llm_response_raw[:500]}...'")
            self.memory.log("SEED_LLMError", {"cycle":cycle_id, "stage":"ActionValidation", "error":f"{err}", "raw_snippet":llm_response_raw[:500]}, tags=['Seed','LLM','Error','Action'])
            return {"action_type": "FALLBACK", "reasoning": f"LLM action validation failed ({err})"}

    def _execute_seed_action(self, action_type: str, action_params: Dict, cycle_id: str, current_depth: int) -> ActionResult:
        start_time = time.time()
        exec_res: ActionResult = {"success": False, "message": f"Action '{action_type}' failed/not implemented."}
        log_params = {k:v for k,v in action_params.items() if k not in ['logic', 'new_logic', 'content', 'new_content', 'test_scenario', 'trigger_pattern']}
        if action_type == "ANALYZE_MEMORY": log_params = {"query": action_params.get("query")}
        if action_type == "INDUCE_BEHAVIORAL_RULE": log_params = {"suggestion": action_params.get("suggested_response"), "rule_id": action_params.get("rule_id")}
        if action_type == "MODIFY_CORE_CODE":
            v_hash = action_params.get("verification_hash")
            log_params['verification_hash'] = (v_hash[:8] if isinstance(v_hash, str) else str(v_hash))
        if action_type == "SET_VM_MODE": 
            log_params = {"mode": action_params.get("mode"), "reasoning": action_params.get("reasoning")}

        log_data = {"cycle":cycle_id, "action_params": log_params, "depth": current_depth}
        log_tags = ['Seed', 'Action', action_type]
        
        try:
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
                 analysis_prompt=action_params.get("analysis_prompt","Summarize relevance to current goal and recent errors.")
                 logger.info(f"Seed Analyze Mem Query (LLM): '{query}'")
                 if query and isinstance(query, str):
                    mem_context = self.memory.retrieve_context(query=query, limit=10)
                    ctx_for_llm = {k:v for k,v in mem_context.items() if k not in ['recent_episodic', 'current_behavioral_rules']}
                    ctx_for_llm['rule_count'] = len(mem_context.get('current_behavioral_rules', {}))
                    llm_temp = self.memory.get_learning_parameter('llm_query_temperature.value') or 0.5
                    llm_mem_prompt = f"Context: Based on the memory search results for query '{query}', perform the following analysis:\n{analysis_prompt}\n\nMemory Context (Limited):\n```json\n{json.dumps(ctx_for_llm, default=str, indent=2)}\n```\n\nProvide only the analysis result in JSON."
                    analysis = self.llm_service.query(llm_mem_prompt, max_tokens=500, temperature=llm_temp)
                    exec_res={"success":True,"message":"LLM-guided memory analysis complete.", "details":{"query":query, "analysis_result": analysis}}
                    self.memory.log(f"SEED_MemAnalysis_{cycle_id}", {"query": query, "analysis": analysis, "guided_by": "LLM"}, tags=['Seed','Memory', 'LLM'])
                 else:
                    exec_res={"success":False,"message":"Requires valid 'query' string."}; log_tags.append('Error'); exec_res['reason'] = 'invalid_argument'
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
                    if verification_hash is None:
                         logger.warning(f"Attempting emergency core code modification with null verification_hash.")
                         exec_res = self._execute_modify_core_code(action_params, cycle_id); log_tags.append('CoreMod')
                    elif not isinstance(verification_hash, str):
                         exec_res = {"success": False, "message": "Modification rejected: Invalid 'verification_hash' parameter type."}; log_tags.extend(['Error', 'Safety', 'Verification']); logger.warning(f"Core code modification REJECTED: Invalid verification_hash type.")
                    else:
                         verification_record = self._verified_code_mods.get(verification_hash); is_verified = False; verification_reason = "No recent verification record found for this hash."
                         if verification_record:
                             age = time.time() - verification_record['timestamp'];
                             if age < self._verified_mod_expiry_sec:
                                 if verification_record.get('result',{}).get('success'):
                                      if verification_record.get('params_hash') == verification_hash:
                                           is_verified = True; verification_reason = "Verification passed."
                                      else: verification_reason = f"Verification hash mismatch. Record: {verification_record.get('params_hash')[:8]}..., Provided: {verification_hash[:8]}..."
                                 else: verification_reason = f"Verification previously failed ({verification_record.get('result',{}).get('message', 'No msg')})."
                             else: verification_reason = f"Verification expired ({age:.0f}s ago)."
                         if is_verified:
                             logger.info(f"Verification check passed for hash {verification_hash[:8]}. Proceeding with core modification.");
                             exec_res = self._execute_modify_core_code(action_params, cycle_id); log_tags.append('CoreMod')
                         else:
                             exec_res = {"success": False, "message": f"Core code modification rejected: {verification_reason}"}; log_tags.extend(['Error', 'Safety', 'Verification']);
                             logger.warning(f"Core code modification for hash {verification_hash[:8]} REJECTED: {verification_reason}")
            elif action_type == "READ_FILE":
                 path = action_params.get("path"); logger.info(f"Seed Reading file: '{path}'");
                 if path and isinstance(path, str): read_res = self.vm_service.read_file(path); exec_res = read_res if isinstance(read_res, dict) else {"success": False, "message": "Invalid response from vm_service.read_file"}; exec_res['reason'] = read_res.get('reason')
                 else: exec_res={"success": False, "message": "Requires valid 'path'."}; log_tags.append('Error'); exec_res['reason'] = 'invalid_argument'
                 if not exec_res.get("success"): log_tags.append('Error')
            elif action_type == "WRITE_FILE":
                 path = action_params.get("path"); content = action_params.get("content"); logger.info(f"Seed Writing to file: '{path}' (Content type: {type(content).__name__})")
                 if path and isinstance(path, str) and content is not None:
                      content_str = str(content) if not isinstance(content, str) else content
                      write_res = self.vm_service.write_file(path, content_str); exec_res = write_res if isinstance(write_res, dict) else {"success": False, "message": "Invalid response from vm_service.write_file"}; exec_res['reason'] = write_res.get('reason')
                 else: exec_res={"success": False, "message": "Requires 'path' (string) and 'content'."}; log_tags.append('Error'); exec_res['reason'] = 'invalid_argument'
                 if not exec_res.get("success"): log_tags.append('Error')
            elif action_type == "REQUEST_RESTART":
                reason = action_params.get("reasoning", "No reason provided.")
                logger.warning(f"Seed requesting self-restart. Reason: {reason}");
                self.memory.log(RESTART_SIGNAL_EVENT_TYPE, {"reason": reason, "cycle": cycle_id}, tags=['Seed', 'Restart', 'Control']);
                exec_res = {"success": True, "message": "Restart signal logged."}
                log_tags.append('Control')
            elif action_type == "NO_OP":
                exec_res = {"success": True, "message": "Executed NO_OP."}
            elif action_type == "UPDATE_LEARNING_PARAMETER":
                 param_name = action_params.get("parameter_name")
                 new_value = action_params.get("new_value")
                 logger.info(f"Seed attempting update learning parameter '{param_name}' to {new_value}")
                 if param_name and new_value is not None:
                      success_update = self.memory.update_learning_parameter(param_name, new_value)
                      exec_res = {"success": success_update, "message": f"Learning parameter '{param_name}' update {'succeeded' if success_update else 'failed or rejected'}."}
                      if success_update: log_tags.extend(['Parameter', 'Learning'])
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
            elif action_type == "SET_VM_MODE":
                log_tags.append('VMMode')
                requested_mode = action_params.get("mode")
                if requested_mode not in ["simulation", "real"]:
                    exec_res = {"success": False, "message": f"Invalid mode '{requested_mode}' for SET_VM_MODE. Must be 'simulation' or 'real'.", "reason": "invalid_argument"}
                else:
                    config_file_rel_path = "seed/config.py"
                    config_file_abs_path = self.project_root.joinpath(config_file_rel_path)
                    target_bool_value = "True" if requested_mode == "real" else "False"
                    target_line_regex = r"^\s*VM_SERVICE_USE_REAL\s*=\s*(True|False)"
                    replacement_line = f"VM_SERVICE_USE_REAL = {target_bool_value}"
                    
                    config_changed = False
                    config_content_lines = []
                    try:
                        if not config_file_abs_path.is_file():
                            raise FileNotFoundError(f"Config file {config_file_rel_path} not found.")
                        
                        with open(config_file_abs_path, 'r', encoding='utf-8') as f:
                            config_content_lines = f.readlines()

                        found_and_modified = False
                        for i, line in enumerate(config_content_lines):
                            if re.match(target_line_regex, line.strip()):
                                # Check if already the desired mode
                                current_value_match = re.search(r"=\s*(True|False)", line)
                                if current_value_match and current_value_match.group(1) == target_bool_value:
                                    logger.info(f"VM_SERVICE_USE_REAL already set to {target_bool_value}. No change needed.")
                                    exec_res = {"success": True, "message": f"VM_SERVICE_USE_REAL already set to '{requested_mode}'. No change made.", "reason": "already_set"}
                                    found_and_modified = True # Treat as success, no restart needed yet
                                    break 
                                
                                # Preserve indentation
                                indent = re.match(r"^\s*", line).group(0)
                                config_content_lines[i] = f"{indent}{replacement_line}\n"
                                logger.info(f"Config line for VM_SERVICE_USE_REAL prepared for update to: {replacement_line.strip()}")
                                found_and_modified = True
                                config_changed = True 
                                break
                        
                        if not found_and_modified and not exec_res.get("success"): # If loop finished and not already set
                             raise ValueError(f"Could not find VM_SERVICE_USE_REAL line in {config_file_rel_path}")

                        if config_changed:
                            # Perform the actual modification (simplified for direct internal call)
                            # Backup (optional but good practice, simplified here)
                            # backup_dir = self.project_root.joinpath(CORE_CODE_MODIFICATION_BACKUP_DIR)
                            # backup_dir.mkdir(parents=True, exist_ok=True)
                            # backup_path = backup_dir / f"config_{time.strftime('%Y%m%d_%H%M%S')}.py.bak"
                            # shutil.copy2(config_file_abs_path, backup_path)
                            # logger.info(f"Backed up {config_file_rel_path} to {backup_path.name}")

                            with open(config_file_abs_path, 'w', encoding='utf-8') as f:
                                f.writelines(config_content_lines)
                            logger.info(f"Successfully modified {config_file_rel_path} to set VM_SERVICE_USE_REAL = {target_bool_value}.")
                            
                            # Update learning parameter
                            self.memory.update_learning_parameter("vm_target_mode.value", requested_mode)
                            
                            # Trigger restart
                            logger.warning(f"SET_VM_MODE: Requesting system restart to apply config change to '{requested_mode}' mode.")
                            restart_reason = f"SET_VM_MODE action changed VM_SERVICE_USE_REAL to {target_bool_value}."
                            self.memory.log(RESTART_SIGNAL_EVENT_TYPE, {"reason": restart_reason, "cycle": cycle_id, "requested_mode": requested_mode}, tags=['Seed', 'Restart', 'Control', 'VMMode'])
                            exec_res = {"success": True, "message": f"VM mode set to '{requested_mode}'. Config updated. Restart requested.", "reason": "restart_triggered"}
                        elif not exec_res.get("success"): # If found_and_modified was false and not already set
                            exec_res = {"success": False, "message": f"Failed to find VM_SERVICE_USE_REAL line in {config_file_rel_path}.", "reason": "config_line_not_found"}

                    except Exception as e_cfg_mod:
                        logger.error(f"Error during SET_VM_MODE config modification: {e_cfg_mod}", exc_info=True)
                        exec_res = {"success": False, "message": f"Error modifying config for SET_VM_MODE: {e_cfg_mod}", "reason": "config_modification_error"}
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

    def _execute_modify_core_code(self, action_params: Dict, cycle_id: str) -> ActionResult:
        file_rel_path = action_params.get("file_path"); mod_type = action_params.get("modification_type"); target_line = action_params.get("target_line_content"); new_content = action_params.get("new_content", "")
        result: ActionResult = {"success": False, "message": "Modification failed.", "details": {}, "reason": "unknown"}
        log_prefix = "Applying VERIFIED CORE CODE MODIFICATION" if action_params.get("verification_hash") else "Applying UNVERIFIED (EMERGENCY) CORE CODE MODIFICATION"
        logger.warning(f"{log_prefix}: File='{file_rel_path}', Type='{mod_type}'")
        try:
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
            result['details']['validated_path'] = str(full_path)
            with open(full_path, 'r', encoding='utf-8') as f_read: original_lines = f_read.readlines()
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
            modified_code = "".join(modified_lines)
            try: ast.parse(modified_code, filename=str(full_path)); logger.info("Final AST Check OK.")
            except SyntaxError as syn_err: result['reason'] = 'syntax_error'; raise ValueError(f"Final AST Check FAILED: {syn_err}") from syn_err
            backup_path = None
            try:
                backup_dir = self.project_root.joinpath(CORE_CODE_MODIFICATION_BACKUP_DIR); backup_dir.mkdir(parents=True, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S"); backup_filename = f"{full_path.stem}_{timestamp}{full_path.suffix}.bak"; backup_path = backup_dir / backup_filename
                with open(backup_path, 'w', encoding='utf-8') as f_bak: f_bak.writelines(original_lines)
                logger.info(f"Original file backed up to: {backup_path}")
                with open(full_path, 'w', encoding='utf-8') as f_write: f_write.write(modified_code)
                result = {"success": True, "message": f"Core code file '{file_rel_path}' modified successfully. Backup: {backup_path.name}", "details": {"file_path": file_rel_path, "backup_path": str(backup_path)}, "reason": "applied"}
            except Exception as write_err: result['reason'] = 'write_error'; raise IOError(f"Error writing modified file or backup: {write_err}") from write_err
        except (FileNotFoundError, ValueError, IOError, PermissionError, Exception) as err:
            result['message'] = f"Core Code Apply Error: {err}"
            result['details'] = result.get('details', {}); result['details']["error_type"] = type(err).__name__
            logger.error(f"Core Code Apply FAILED (Reason: {result.get('reason', 'unknown')}): {err}", exc_info=isinstance(err, (IOError, Exception)))
        return result

    def _execute_test_core_code_modification(self, action_params: Dict, cycle_id: str, current_depth: int) -> ActionResult:
        file_rel_path = action_params.get("file_path"); mod_type = action_params.get("modification_type"); target_name = action_params.get("target_name"); new_logic = action_params.get("new_logic"); test_scenario = action_params.get("test_scenario")
        result: ActionResult = {"success": False, "message": "Test failed.", "details": {}, "reason": "unknown"}
        result['details']['params'] = {k:v for k,v in action_params.items() if k != 'new_logic'}
        logger.info(f"Executing CORE CODE TEST: File='{file_rel_path}', Target='{target_name}'")
        try:
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
        
        is_method = (mod_type == "REPLACE_METHOD")

        def _run_test_in_sandbox(result_queue: queue.Queue):
            sandbox_result = {"output": None, "error": None, "success": False, "message": "", "raw_output": None, "mock_calls": {}, "evaluation_details": {}}
            func_handle = None; mock_self_instance = None; mock_services_dict = {};
            expected_outcome = test_scenario.get("expected_outcome", {})
            test_inputs = test_scenario.get('test_inputs', []); prepared_args = copy.deepcopy(test_inputs);
            try:
                logger.debug("Sandbox: Parsing new logic...");
                parsed_ast = ast.parse(new_logic, filename="<new_logic>"); defined_name = None
                if parsed_ast.body and isinstance(parsed_ast.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)): defined_name = parsed_ast.body[0].name
                if not defined_name: raise SyntaxError("Provided new_logic does not define a top-level function/method.")
                expected_func_name = target_name
                if defined_name != expected_func_name: raise SyntaxError(f"Logic defines '{defined_name}' but target was '{expected_func_name}'.")
                logger.debug("Sandbox: Setting up scope and mocks...");
                isolated_globals = SAFE_EXEC_GLOBALS.copy(); isolated_locals = {};
                
                if 'MockCoreService' not in isolated_globals:
                    isolated_globals['MockCoreService'] = MockCoreService

                test_logger = logging.getLogger(f"CoreCodeTestSandbox.{target_name}");
                isolated_globals['logger'] = test_logger; 
                isolated_globals['MockSelf'] = MockSelf 

                mock_services_config = expected_outcome.get('mock_services', {})
                mock_services_dict['memory'] = MockMemorySystem(return_values=mock_services_config.get('memory'))
                mock_services_dict['llm_service'] = MockLLMService(return_values=mock_services_config.get('llm_service'))
                mock_services_dict['vm_service'] = MockVMService(return_values=mock_services_config.get('vm_service'))
                
                if is_method:
                    logger.debug("Sandbox: Creating MockSelf instance with mocks for method test...");
                    mock_self_instance = MockSelf(mock_services=mock_services_dict) 
                    prepared_args.insert(0, mock_self_instance)
                    logger.debug(f"Sandbox: Prepared method call for '{defined_name}' with {len(test_inputs)} user args (+ mock self)...")
                else:
                     logger.debug(f"Sandbox: Prepared function call for '{defined_name}' with {len(test_inputs)} args...")

                logger.debug(f"Sandbox: Executing new logic definition for '{defined_name}'...");
                exec(new_logic, isolated_globals, isolated_locals)
                func_handle = isolated_locals.get(defined_name)
                if not callable(func_handle): raise TypeError(f"Executed logic did not result in callable function/method '{defined_name}'.")
                
                actual_exception = None; actual_output = None
                logger.debug(f"Sandbox: Invoking '{defined_name}'...")
                try: actual_output = func_handle(*prepared_args); sandbox_result['raw_output'] = actual_output
                except Exception as exec_runtime_err: logger.warning(f"Sandbox: Execution raised exception: {exec_runtime_err}"); actual_exception = exec_runtime_err
                
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
                else: 
                    if actual_exception is not None: 
                        passed = False; 
                        eval_msgs.append(f"FAILED: No exception expected, got {type(actual_exception).__name__}")
                        sandbox_result['error'] = f"{type(actual_exception).__name__}: {actual_exception}" 
                    else: 
                        eval_msgs.append("PASSED: No unexpected exception.");
                        if 'return_value' in expected_outcome:
                            expected_return = expected_outcome['return_value'];
                            if expected_return == "ANY": eval_msgs.append("PASSED: Return value check skipped (ANY).")
                            elif repr(actual_output) == repr(expected_return): eval_msgs.append(f"PASSED: Return value matched.")
                            else: passed = False; eval_msgs.append(f"FAILED: Return value mismatch. Got: {repr(actual_output)[:100]}... Exp: {repr(expected_return)[:100]}...")
                        try: sandbox_result['output'] = json.loads(json.dumps(actual_output, default=str))
                        except Exception: sandbox_result['output'] = repr(actual_output)

                expected_calls = expected_outcome.get('mock_calls', {}); 
                actual_calls_all = {}; 
                if 'evaluation_details' not in sandbox_result: sandbox_result['evaluation_details'] = {} 

                for service_name, mock_instance_svc in mock_services_dict.items(): 
                    actual_calls_all[service_name] = mock_instance_svc.get_calls() 

                if is_method and mock_self_instance and hasattr(mock_self_instance, 'logger') and "logger" in expected_calls:
                    if hasattr(mock_self_instance.logger, 'get_calls') and callable(mock_self_instance.logger.get_calls):
                        if "logger" not in actual_calls_all: 
                            actual_calls_all["logger"] = mock_self_instance.logger.get_calls()
                            sandbox_result['evaluation_details']["logger_from_self"] = "Retrieved calls from self.logger (MockCoreService)"
                    elif "logger" in expected_calls: 
                        sandbox_result['evaluation_details']["logger_from_self_error"] = "self.logger exists but is not a MockCoreService or similar; cannot get_calls() for mock verification."
                
                for service_name, expected_methods in expected_calls.items():
                    if service_name not in sandbox_result['evaluation_details']: sandbox_result['evaluation_details'][service_name] = {}
                    actual_service_calls = actual_calls_all.get(service_name, {});
                    if not isinstance(expected_methods, dict): 
                        passed = False; eval_msgs.append(f"FAILED: Invalid expected_calls structure for service '{service_name}'."); 
                        sandbox_result['evaluation_details'][service_name]["__error__"] = "Invalid expected_calls structure";
                        continue
                    for method_name, expected_call_list_or_count in expected_methods.items(): 
                        actual_method_calls = actual_service_calls.get(method_name, []); actual_count = len(actual_method_calls)
                        if isinstance(expected_call_list_or_count, int): 
                            expected_count = expected_call_list_or_count
                            if actual_count == expected_count: 
                                eval_msgs.append(f"PASSED: Mock call count matched for {service_name}.{method_name} ({expected_count})."); 
                                sandbox_result['evaluation_details'][service_name][method_name] = {"expected_count": expected_count, "actual_count": actual_count, "match": True}
                            else: 
                                passed = False; eval_msgs.append(f"FAILED: Mock call count mismatch for {service_name}.{method_name} (Exp {expected_count}, Got {actual_count})."); 
                                sandbox_result['evaluation_details'][service_name][method_name] = {"expected_count": expected_count, "actual_count": actual_count, "match": False}
                        elif isinstance(expected_call_list_or_count, list): 
                            expected_call_list = expected_call_list_or_count
                            expected_count = len(expected_call_list); 
                            current_method_eval_details = {"expected_count": expected_count, "actual_count": actual_count, "match": False, "details": []}
                            if actual_count != expected_count: 
                                passed = False; eval_msgs.append(f"FAILED: Mock call count mismatch for {service_name}.{method_name} (Exp {expected_count}, Got {actual_count})."); 
                                current_method_eval_details["details"].append("Call count mismatch.")
                            else: 
                                calls_match_detail = True
                                for i, (expected_call_detail, actual_call_detail) in enumerate(zip(expected_call_list, actual_method_calls)):
                                    if not isinstance(expected_call_detail, dict): 
                                        calls_match_detail = False; eval_msgs.append(f"FAILED: Invalid expected call structure {i} for {service_name}.{method_name}."); 
                                        current_method_eval_details["details"].append(f"Invalid expected call structure {i}"); break
                                    expected_args = expected_call_detail.get('args', "ANY"); expected_kwargs = expected_call_detail.get('kwargs', "ANY");
                                    actual_args_val = actual_call_detail.get('args', []); actual_kwargs_val = actual_call_detail.get('kwargs', {}); 
                                    args_match = (expected_args == "ANY" or repr(expected_args) == repr(actual_args_val))
                                    kwargs_match = (expected_kwargs == "ANY" or repr(expected_kwargs) == repr(actual_kwargs_val))
                                    if not args_match or not kwargs_match: 
                                        calls_match_detail = False; 
                                        mismatch_info = f"Arg/Kwarg mismatch call {i+1}. Exp: args={expected_args}, kwargs={expected_kwargs}. Act: args={actual_args_val}, kwargs={actual_kwargs_val}"; 
                                        eval_msgs.append(f"FAILED: Mock call {mismatch_info} for {service_name}.{method_name}."); 
                                        current_method_eval_details["details"].append(mismatch_info); break
                                if calls_match_detail and expected_call_list: 
                                    current_method_eval_details["match"] = True; eval_msgs.append(f"PASSED: Mock calls (args/kwargs) matched for {service_name}.{method_name}.")
                                elif not calls_match_detail : passed = False
                            sandbox_result['evaluation_details'][service_name][method_name] = current_method_eval_details
                        else: 
                            passed = False; eval_msgs.append(f"FAILED: Invalid expected_calls value for '{service_name}.{method_name}'. Must be list of dicts or int.");
                            sandbox_result['evaluation_details'][service_name][method_name] = {"error": "Invalid expectation type"}

                for service_name_actual, actual_methods in actual_calls_all.items(): 
                     if service_name_actual not in expected_calls: 
                          if actual_methods: 
                              passed=False; eval_msgs.append(f"FAILED: Unexpected calls to service '{service_name_actual}'. Details: {actual_methods}");
                              if service_name_actual not in sandbox_result['evaluation_details']: sandbox_result['evaluation_details'][service_name_actual] = {}
                              sandbox_result['evaluation_details'][service_name_actual]["__UNEXPECTED_SERVICE_CALLS__"] = actual_methods
                     elif isinstance(expected_calls.get(service_name_actual), dict): 
                          for method_name_actual, calls in actual_methods.items(): 
                              if method_name_actual not in expected_calls[service_name_actual]:
                                   if calls: 
                                       passed=False; eval_msgs.append(f"FAILED: Unexpected calls to method '{service_name_actual}.{method_name_actual}'. Details: {calls}");
                                       if service_name_actual not in sandbox_result['evaluation_details']: sandbox_result['evaluation_details'][service_name_actual] = {}
                                       if method_name_actual not in sandbox_result['evaluation_details'][service_name_actual]: sandbox_result['evaluation_details'][service_name_actual][method_name_actual] = {}
                                       sandbox_result['evaluation_details'][service_name_actual][method_name_actual]["__UNEXPECTED_METHOD_CALLS__"] = calls
                
                sandbox_result['success'] = passed; 
                sandbox_result['message'] = "; ".join(eval_msgs); 
                sandbox_result['mock_calls'] = actual_calls_all;
            except Exception as eval_err:
                sandbox_result['success'] = False; sandbox_result['error'] = f"SandboxError: {eval_err}"; sandbox_result['message'] = "Test sandbox failed internally during setup or evaluation."; logger.error(f"Core Code Test Failed (Sandbox Setup/Eval): {eval_err}", exc_info=True)
            result_queue.put(sandbox_result)
        
        result_queue_obj = queue.Queue(); 
        test_thread = threading.Thread(target=_run_test_in_sandbox, args=(result_queue_obj,)) 
        test_thread.daemon = True; test_thread.start()
        timeout_ms = test_scenario.get('max_test_duration_ms', CORE_CODE_TEST_DEFAULT_TIMEOUT_MS)
        timeout_sec = timeout_ms / 1000.0; test_thread.join(timeout=timeout_sec)
        
        if test_thread.is_alive():
            result['success'] = False; result['message'] = f"Test failed: Timed out after {timeout_sec:.1f} seconds."; result['details']['timed_out'] = True; result['reason'] = 'timeout'; logger.warning(f"Core Code Test TIMED OUT for target '{target_name}'.")
        else:
            try: sandbox_res = result_queue_obj.get_nowait(); 
            except queue.Empty: sandbox_res = {"success": False, "message": "Test failed: Result queue empty after thread join.", "error":"QueueEmpty", "evaluation_details":{}}
            except Exception as q_err: sandbox_res = {"success": False, "message": f"Test failed: Error getting result from queue: {q_err}", "error":str(q_err), "evaluation_details":{}}
            result['success'] = sandbox_res.get('success', False);
            result['message'] = sandbox_res.get('message', 'Test completed, result format invalid.');
            result['details'].update(sandbox_res); result['details']['timed_out'] = False 
            result['reason'] = 'test_passed' if result['success'] else 'test_failed'
        return result

    def _execute_verify_core_code_change(self, action_params: Dict, cycle_id: str) -> ActionResult:
        result: ActionResult = {"success": False, "message": "Verification failed.", "details": {}, "reason": "unknown"}
        result['details']['params'] = {k:v for k,v in action_params.items() if k not in ['new_logic', 'new_content']}
        file_rel_path = action_params.get("file_path"); mod_hash = self._hash_modification_params(action_params)
        verification_level = action_params.get("verification_level", "basic")
        logger.info(f"Executing CORE CODE VERIFICATION: File='{file_rel_path}', Hash='{mod_hash[:8]}...', Level='{verification_level}'")
        try:
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
        try:
            verify_success, verify_message, verify_details = run_verification_suite(
                project_root=self.project_root, modification_params=action_params, verification_level=verification_level
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
        hash_content = {}
        path_key = "file_path"
        keys_to_hash = [path_key, "modification_type", "target_name", "target_line_content", "new_logic", "new_content"]
        for key in sorted(keys_to_hash):
            if key in action_params: hash_content[key] = action_params[key]
        content_str = json.dumps(hash_content, sort_keys=True)
        return hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    def _get_fallback_action(self, reason: str) -> Dict:
        logger.warning(f"Seed Fallback triggered: {reason}.")
        query = f"Analyze state and recent failures given fallback reason: {reason}"
        reasoning = f"Fallback Action (LLM Query): {reason}. Analyzing memory via LLM to understand context."
        return {
            "action_type": "ANALYZE_MEMORY",
            "query": query,
            "reasoning": reasoning
        }