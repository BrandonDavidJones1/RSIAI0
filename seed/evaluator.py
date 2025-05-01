# --- START OF FILE seed/evaluator.py ---

# RSIAI/seed/evaluator.py
"""
Defines the Seed_SuccessEvaluator class for assessing Seed action effectiveness.
Uses configured weights for evaluation components.
Accepts actual post-action sensory input for accurate evaluation.
"""
import numpy as np
import math
import time
import traceback
import logging
import json
import copy
from typing import Dict, Any, Optional, TYPE_CHECKING

# Prevent circular import for type hinting
if TYPE_CHECKING:
    from .sensory import Seed_SensoryRefiner, RefinedInput # Corrected relative import

# Import necessary config constants using relative import within the package
from .config import (
    SEED_LEARNING_PARAMETERS # Import the whole dict, evaluator uses a sub-key
)

logger = logging.getLogger(__name__)

# Renamed class
class Seed_SuccessEvaluator:
    """ Evaluates Seed action success using configured weights. """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 sensory_refiner: Optional['Seed_SensoryRefiner'] = None):
        """
        Args:
            config (dict, optional): Configuration overrides (not currently used).
            sensory_refiner (Seed_SensoryRefiner, optional): Not currently used.
        """
        self.config = config if config else {}
        self.sensory_refiner = sensory_refiner

        # Get evaluation weights from the imported parameters dictionary
        self.seed_score_weights: Dict[str, float] = {
            k: v.get('value', 0.0)
            for k, v in SEED_LEARNING_PARAMETERS.get("evaluation_weights", {}).items()
            if isinstance(v, dict)
        }
        logger.info(f"Seed Evaluator Weights loaded: {json.dumps(self.seed_score_weights, indent=2)}")

        logger.info("Seed Success Evaluator Initialized.")

    def evaluate_seed_action_success(self,
                                     initial_state_snapshot: dict,
                                     post_action_sensory_input: Optional['RefinedInput'],
                                     execution_result: dict,
                                     action_taken: dict,
                                     current_goal: dict,
                                     evaluation_weights: Optional[Dict[str, float]] = None) -> dict:
        """
        Evaluates success of a Seed action using weighted components and actual post-action state.

        Args:
            initial_state_snapshot (dict): Snapshot of system state *before* the action.
                                           Expected to contain 'seedSensory' key.
            post_action_sensory_input (Optional[RefinedInput]): Refined sensory input *after* the action.
            execution_result (dict): The result dictionary from Seed_Core._execute_seed_action.
            action_taken (dict): The Seed action dictionary decided by the LLM.
            current_goal (dict): The current Seed goal dictionary.
            evaluation_weights (Optional[Dict]): If provided, overrides the instance's weights for this evaluation.

        Returns:
            dict: A dictionary containing evaluation scores ('overall_success', component scores) and a message.
        """
        scores: Dict[str, Any] = {
            'actionExecutionSuccess': 0.0,
            'goalProgress': 0.0,
            'resourceEfficiency': 0.0,
            'overall_success': 0.0,
            'message': "",
            'action_summary': f"{action_taken.get('action_type', 'Unknown')}",
        }
        if action_taken.get('action_type') == "EXECUTE_VM_COMMAND": scores['action_summary'] += f": {action_taken.get('command', '')[:30]}..."
        elif action_taken.get('action_type') in ["READ_FILE", "WRITE_FILE", "MODIFY_CORE_CODE"]: scores['action_summary'] += f": {action_taken.get('file_path', action_taken.get('path',''))}"

        current_weights = evaluation_weights if evaluation_weights else self.seed_score_weights

        # 1. Execution Success Score
        exec_success = execution_result.get('success', False)
        scores['actionExecutionSuccess'] = 1.0 if exec_success else 0.0
        action_type = action_taken.get('action_type', 'UnknownAction')
        if not exec_success:
            reason = execution_result.get('reason')
            reason_str = f" (Reason: {reason})" if reason else ""
            scores['message'] = f"Exec failed ({action_type}): {execution_result.get('message', 'No message')}{reason_str}"
            scores['overall_success'] = scores['actionExecutionSuccess'] * current_weights.get('execution', 0.0)
            scores['details'] = execution_result.get('details', {})
            return scores

        scores['message'] = f"Exec OK ({action_type})."

        # 2. Goal Progress Score
        try:
            initial_sensory = initial_state_snapshot.get('seedSensory')
            if initial_sensory and post_action_sensory_input:
                # Pass action_taken to the helper method
                scores['goalProgress'] = self._calculate_goal_progress(initial_sensory, post_action_sensory_input, current_goal, action_taken)
                scores['message'] += f" GoalProg={scores['goalProgress']:.2f}."
            elif initial_sensory and not post_action_sensory_input:
                 logger.warning("Cannot calculate goal progress accurately: Post-action sensory data unavailable.")
                 scores['goalProgress'] = 0.0
                 scores['message'] += " GoalProg=N/A (No Post-State)."
            else:
                logger.warning("Cannot calculate goal progress: Missing initial sensory state data.")
                scores['goalProgress'] = 0.0
                scores['message'] += " GoalProg=N/A (No Pre-State)."
        except Exception as e:
            logger.warning(f"Goal progress calculation error: {e}", exc_info=True)
            scores['goalProgress'] = 0.0
            scores['message'] += " (GoalProg Error)."

        # 3. Resource Efficiency Score
        scores['resourceEfficiency'] = 0.5
        duration = execution_result.get('details', {}).get('seed_action_duration_sec')
        if duration is not None and isinstance(duration, (int, float)) and duration >= 0:
            scores['resourceEfficiency'] = max(0.0, 1.0 - (duration / 60.0))
            scores['message'] += f" Eff={scores['resourceEfficiency']:.2f} (Dur={duration:.1f}s)."
        else:
            scores['message'] += " Eff=N/A."

        # 4. Overall Weighted Score
        overall = (scores['actionExecutionSuccess'] * current_weights.get('execution', 0.0) +
                   scores['goalProgress'] * current_weights.get('goal_prog', 0.0) +
                   scores['resourceEfficiency'] * current_weights.get('efficiency', 0.0))
        scores['overall_success'] = max(0.0, min(1.0, overall))
        scores['message'] += f" Overall={scores['overall_success']:.3f}."
        scores['details'] = execution_result.get('details', {})
        return scores


    # Modify definition to accept action_taken
    def _calculate_goal_progress(self, pre_sensory: dict, post_sensory: dict, goal: dict, action_taken: dict) -> float:
        """
        Calculates goal progress based on actual pre/post state changes towards the current goal.
        Returns a score between -1.0 (moved away) and 1.0 (achieved).
        """
        pre_target = pre_sensory.get('target_status', {})
        post_target = post_sensory.get('target_status', {})
        goal_type = goal.get('target', 'unknown')
        goal_path = goal.get('path')
        goal_hint = goal.get('content_hint')
        progress = 0.0

        if goal_type in ['create_file', 'modify_file', 'delete_file', 'create_directory']:
            if not goal_path:
                logger.warning(f"Cannot calculate goal progress for type '{goal_type}': Missing 'path' in goal definition.")
                return 0.0

            pre_exists = pre_target.get('exists', False)
            post_exists = post_target.get('exists', False)
            pre_type = pre_target.get('type')
            post_type = post_target.get('type')

            if goal_type == 'create_file':
                target_type = 'file'
                pre_state_met = pre_exists and pre_type == target_type
                post_state_met = post_exists and post_type == target_type
                if goal_hint:
                    pre_hint_ok = pre_target.get('hint_present', False)
                    post_hint_ok = post_target.get('hint_present', False)
                    pre_state_met = pre_state_met and pre_hint_ok
                    post_state_met = post_state_met and post_hint_ok
                if post_state_met and not pre_state_met: progress = 1.0
                elif post_state_met and pre_state_met: progress = 0.1
                elif not post_state_met and pre_state_met: progress = -0.5
                elif post_exists and post_type == target_type and not pre_exists: progress = 0.5
                else: progress = 0.0

            elif goal_type == 'delete_file':
                target_type = 'file'
                pre_state_met = not (pre_exists and pre_type == target_type)
                post_state_met = not (post_exists and post_type == target_type)
                if post_state_met and not pre_state_met: progress = 1.0
                elif post_state_met and pre_state_met: progress = 0.1
                elif not post_state_met and pre_state_met: progress = -0.5
                else: progress = 0.0

            elif goal_type == 'modify_file':
                 if pre_exists and post_exists and pre_type == 'file' and post_type == 'file':
                      pre_mtime = pre_target.get('mtime'); post_mtime = post_target.get('mtime')
                      # Use mtime comparison if available, otherwise assume modification if file still exists
                      if pre_mtime is not None and post_mtime is not None and pre_mtime != post_mtime: progress = 0.8
                      elif pre_mtime is None or post_mtime is None : progress = 0.5 # Assume modified if time unknown but file exists
                      else: progress = 0.1 # Exists, but not modified based on time
                 elif post_exists and not pre_exists: progress = 0.2
                 elif not post_exists and pre_exists: progress = -0.5
                 else: progress = 0.0

            elif goal_type == 'create_directory':
                target_type = 'directory'
                pre_state_met = pre_exists and pre_type == target_type
                post_state_met = post_exists and post_type == target_type
                if post_state_met and not pre_state_met: progress = 1.0
                elif post_state_met and pre_state_met: progress = 0.1
                elif not post_state_met and pre_state_met: progress = -0.5
                else: progress = 0.0

        elif goal_type == 'bootstrap_intelligence':
            # Now we can access action_taken here
            action_type = action_taken.get('action_type')
            if action_type in ["MODIFY_CORE_CODE", "VERIFY_CORE_CODE_CHANGE", "TEST_CORE_CODE_MODIFICATION"]:
                # Give higher progress if the action was actually successful
                if action_taken.get('success', False): # Check success from execution_result passed via action_taken? No, check evaluate_seed_action_success context
                     # We need execution_result here too, or assume success if we reach here?
                     # Let's assume if evaluate_seed_action_success called us, exec was successful.
                     progress = 0.3 # Higher boost for successful RSI attempt
                else:
                     progress = 0.1 # Small boost just for attempting RSI
            elif action_type == "UPDATE_LEARNING_PARAMETER": progress = 0.05 # Smaller boost for learning adaptation
            elif action_type == "INDUCE_BEHAVIORAL_RULE": progress = 0.05 # Smaller boost for learning adaptation
            else:
                progress = 0.0 # Neutral for other actions

        else:
            logger.debug(f"Goal progress calculation does not handle goal type '{goal_type}'.")
            progress = 0.0

        return np.clip(progress, -1.0, 1.0)

# --- END OF FILE seed/evaluator.py ---