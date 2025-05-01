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
    # Assumes sensory refiner will also be renamed and moved
    from .sensory import Seed_SensoryRefiner, RefinedInput

# Import necessary config constants using relative import within the package
from .config import (
    SEED_LEARNING_PARAMETERS # Import the whole dict, evaluator uses a sub-key
)

logger = logging.getLogger(__name__)

# Renamed class
class Seed_SuccessEvaluator:
    """ Evaluates Seed action success using configured weights. """
    # Updated type hint
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
        # Access the 'value' for each weight within the 'evaluation_weights' category
        self.seed_score_weights: Dict[str, float] = {
            k: v.get('value', 0.0) # Default to 0.0 if 'value' key is missing (shouldn't happen with validation)
            for k, v in SEED_LEARNING_PARAMETERS.get("evaluation_weights", {}).items()
            if isinstance(v, dict) # Ensure the config item is a dictionary
        }
        logger.info(f"Seed Evaluator Weights loaded: {json.dumps(self.seed_score_weights, indent=2)}") # Updated log

        # NN logic was never implemented, removed related flags/checks

        logger.info("Seed Success Evaluator Initialized.") # Updated log

    def evaluate_seed_action_success(self,
                                     initial_state_snapshot: dict,
                                     post_action_sensory_input: Optional['RefinedInput'], # Updated type hint
                                     execution_result: dict,
                                     action_taken: dict,
                                     current_goal: dict,
                                     evaluation_weights: Optional[Dict[str, float]] = None) -> dict: # Added optional override
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
            'action_summary': f"{action_taken.get('action_type', 'Unknown')}", # Add summary for logging
        }
        # Add specific param details to summary for certain actions
        if action_taken.get('action_type') == "EXECUTE_VM_COMMAND":
            scores['action_summary'] += f": {action_taken.get('command', '')[:30]}..."
        elif action_taken.get('action_type') in ["READ_FILE", "WRITE_FILE", "MODIFY_CORE_CODE"]:
             scores['action_summary'] += f": {action_taken.get('file_path', action_taken.get('path',''))}"

        # Use provided weights if available, otherwise use instance default
        current_weights = evaluation_weights if evaluation_weights else self.seed_score_weights

        # 1. Execution Success Score
        exec_success = execution_result.get('success', False)
        scores['actionExecutionSuccess'] = 1.0 if exec_success else 0.0
        action_type = action_taken.get('action_type', 'UnknownAction')
        if not exec_success:
            # Include failure reason if available
            reason = execution_result.get('reason')
            reason_str = f" (Reason: {reason})" if reason else ""
            scores['message'] = f"Exec failed ({action_type}): {execution_result.get('message', 'No message')}{reason_str}"
            # Calculate score based only on execution failure
            scores['overall_success'] = scores['actionExecutionSuccess'] * current_weights.get('execution', 0.0)
            scores['details'] = execution_result.get('details', {}) # Pass details through
            return scores # Return early

        scores['message'] = f"Exec OK ({action_type})."

        # 2. Goal Progress Score
        try:
            # Assumes snapshot key is 'seedSensory'
            initial_sensory = initial_state_snapshot.get('seedSensory')
            if initial_sensory and post_action_sensory_input:
                scores['goalProgress'] = self._calculate_goal_progress(initial_sensory, post_action_sensory_input, current_goal)
                scores['message'] += f" GoalProg={scores['goalProgress']:.2f}."
            elif initial_sensory and not post_action_sensory_input:
                 logger.warning("Cannot calculate goal progress accurately: Post-action sensory data unavailable.")
                 scores['goalProgress'] = 0.0 # Neutral score if post-state unknown
                 scores['message'] += " GoalProg=N/A (No Post-State)."
            else:
                logger.warning("Cannot calculate goal progress: Missing initial sensory state data.")
                scores['goalProgress'] = 0.0 # Neutral score if pre-state unknown
                scores['message'] += " GoalProg=N/A (No Pre-State)."
        except Exception as e:
            logger.warning(f"Goal progress calculation error: {e}", exc_info=True)
            scores['goalProgress'] = 0.0 # Penalize if calculation fails
            scores['message'] += " (GoalProg Error)."

        # 3. Resource Efficiency Score
        scores['resourceEfficiency'] = 0.5 # Neutral default if duration unavailable
        duration = execution_result.get('details', {}).get('seed_action_duration_sec')
        if duration is not None and isinstance(duration, (int, float)) and duration >= 0:
            # Normalize against 60s - longer actions get lower score. Adjust 60.0 as needed.
            scores['resourceEfficiency'] = max(0.0, 1.0 - (duration / 60.0))
            scores['message'] += f" Eff={scores['resourceEfficiency']:.2f} (Dur={duration:.1f}s)."
        else:
            scores['message'] += " Eff=N/A."

        # 4. Overall Weighted Score
        overall = (scores['actionExecutionSuccess'] * current_weights.get('execution', 0.0) +
                   scores['goalProgress'] * current_weights.get('goal_prog', 0.0) +
                   scores['resourceEfficiency'] * current_weights.get('efficiency', 0.0))
        # Normalize by sum of weights used (in case some components are missing/zeroed)
        # total_weight = current_weights.get('execution', 0.0) + current_weights.get('goal_prog', 0.0) + current_weights.get('efficiency', 0.0)
        # if total_weight > 0: overall /= total_weight # Normalize? Or assume weights sum to ~1? Assuming weights are managed to sum appropriately.

        scores['overall_success'] = max(0.0, min(1.0, overall)) # Clamp to [0, 1]
        scores['message'] += f" Overall={scores['overall_success']:.3f}."
        scores['details'] = execution_result.get('details', {}) # Pass details through
        return scores


    def _calculate_goal_progress(self, pre_sensory: dict, post_sensory: dict, goal: dict) -> float:
        """
        Calculates goal progress based on actual pre/post state changes towards the current goal.
        Returns a score between -1.0 (moved away) and 1.0 (achieved).
        """
        # This logic remains the same, compares pre/post target status based on goal type
        pre_target = pre_sensory.get('target_status', {})
        post_target = post_sensory.get('target_status', {})
        goal_type = goal.get('target', 'unknown')
        goal_path = goal.get('path') # Required for file/dir goals
        goal_hint = goal.get('content_hint') # Optional content check
        progress = 0.0

        # --- File/Directory Based Goals ---
        if goal_type in ['create_file', 'modify_file', 'delete_file', 'create_directory']:
            if not goal_path:
                logger.warning(f"Cannot calculate goal progress for type '{goal_type}': Missing 'path' in goal definition.")
                return 0.0 # Cannot evaluate without path

            # Check if the target path matches the path reported in sensory data
            # (Sensory refiner should put info under the correct path key)
            pre_exists = pre_target.get('exists', False)
            post_exists = post_target.get('exists', False)
            pre_type = pre_target.get('type')
            post_type = post_target.get('type')

            if goal_type == 'create_file':
                target_type = 'file'
                pre_state_met = pre_exists and pre_type == target_type
                post_state_met = post_exists and post_type == target_type
                # Optional: Check content hint
                if goal_hint:
                    pre_hint_ok = pre_target.get('hint_present', False)
                    post_hint_ok = post_target.get('hint_present', False)
                    pre_state_met = pre_state_met and pre_hint_ok
                    post_state_met = post_state_met and post_hint_ok

                if post_state_met and not pre_state_met: progress = 1.0 # Goal achieved
                elif post_state_met and pre_state_met: progress = 0.1 # Goal already met
                elif not post_state_met and pre_state_met: progress = -0.5 # Moved away from goal
                elif post_exists and post_type == target_type and not pre_exists: progress = 0.5 # File created, but hint wrong/missing
                else: progress = 0.0 # No relevant change

            elif goal_type == 'delete_file':
                target_type = 'file'
                pre_state_met = not (pre_exists and pre_type == target_type) # Goal is non-existence
                post_state_met = not (post_exists and post_type == target_type)

                if post_state_met and not pre_state_met: progress = 1.0 # Goal achieved (was file, now isn't)
                elif post_state_met and pre_state_met: progress = 0.1 # Goal already met (wasn't file, still isn't)
                elif not post_state_met and pre_state_met: progress = -0.5 # Moved away (wasn't file, now is)
                else: progress = 0.0 # No relevant change

            # Add other goal types like 'modify_file' (check mtime or hash change?), 'create_directory' here
            elif goal_type == 'modify_file':
                 # Check if file exists before and after, and if mtime changed? Or hash?
                 if pre_exists and post_exists and pre_type == 'file' and post_type == 'file':
                      pre_mtime = pre_target.get('mtime')
                      post_mtime = post_target.get('mtime')
                      if pre_mtime != post_mtime: progress = 0.8 # Modified
                      else: progress = 0.1 # Exists, but not modified
                 elif post_exists and not pre_exists: progress = 0.2 # Created instead of modified?
                 elif not post_exists and pre_exists: progress = -0.5 # Deleted instead of modified
                 else: progress = 0.0

            elif goal_type == 'create_directory':
                target_type = 'directory'
                pre_state_met = pre_exists and pre_type == target_type
                post_state_met = post_exists and post_type == target_type
                if post_state_met and not pre_state_met: progress = 1.0 # Goal achieved
                elif post_state_met and pre_state_met: progress = 0.1 # Goal already met
                elif not post_state_met and pre_state_met: progress = -0.5 # Moved away from goal (deleted?)
                else: progress = 0.0 # No relevant change

        # --- Non-Filesystem Goals ---
        elif goal_type == 'bootstrap_intelligence':
            # How to measure this? Look for successful core mod actions?
            # Or maybe LLM sets a more specific sub-goal?
            # For now, maybe give small progress for successful test/verify/modify actions?
            action_type = action_taken.get('action_type')
            if action_type in ["MODIFY_CORE_CODE", "VERIFY_CORE_CODE_CHANGE", "TEST_CORE_CODE_MODIFICATION"]:
                progress = 0.2 # Small boost for attempting RSI
            else:
                progress = 0.0 # Neutral for other actions

        # Add other goal types here...

        else:
            logger.debug(f"Goal progress calculation does not handle goal type '{goal_type}'.")
            progress = 0.0 # Neutral for unknown goal types

        return np.clip(progress, -1.0, 1.0)

# --- END OF FILE seed/evaluator.py ---