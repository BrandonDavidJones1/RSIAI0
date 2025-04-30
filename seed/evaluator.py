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

# Import necessary config constants
from ..config import ( # Adjusted relative import
    SEED_EVALUATION_WEIGHTS, # Use updated constant name
    # L3_EVALUATOR_USE_NN constant removed as NN logic was not implemented
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

        # Use Seed action evaluation weights directly from config
        self.seed_score_weights: Dict[str, float] = SEED_EVALUATION_WEIGHTS # Renamed variable and constant
        logger.info(f"Seed Evaluator Weights loaded: {json.dumps(self.seed_score_weights, indent=2)}") # Updated log

        # NN logic was never implemented, removed related flags/checks
        # self.use_nn = L3_EVALUATOR_USE_NN
        # self.model = None
        # if self.use_nn: logger.warning("NN-based Evaluation not implemented. Using rule-based."); self.use_nn = False

        logger.info("Seed Success Evaluator Initialized.") # Updated log

    # Renamed method
    def evaluate_seed_action_success(self,
                                     initial_state_snapshot: dict,
                                     post_action_sensory_input: Optional['RefinedInput'], # Updated type hint
                                     execution_result: dict,
                                     action_taken: dict,
                                     current_goal: dict) -> dict:
        """
        Evaluates success of a Seed action using weighted components and actual post-action state.

        Args:
            initial_state_snapshot (dict): Snapshot of system state *before* the action.
                                           Expected to contain 'seedSensory' key.
            post_action_sensory_input (Optional[RefinedInput]): Refined sensory input *after* the action.
            execution_result (dict): The result dictionary from Seed_Core._execute_seed_action.
            action_taken (dict): The Seed action dictionary decided by the LLM.
            current_goal (dict): The current Seed goal dictionary.

        Returns:
            dict: A dictionary containing evaluation scores ('overall_success', component scores) and a message.
        """
        scores: Dict[str, Any] = {
            'actionExecutionSuccess': 0.0,
            'goalProgress': 0.0,
            'resourceEfficiency': 0.0,
            'overall_success': 0.0,
            'message': ""
        }

        # 1. Execution Success Score
        exec_success = execution_result.get('success', False)
        scores['actionExecutionSuccess'] = 1.0 if exec_success else 0.0
        action_type = action_taken.get('action_type', 'UnknownAction')
        if not exec_success:
            scores['message'] = f"Exec failed ({action_type}): {execution_result.get('message', 'No message')}"
            scores['overall_success'] = scores['actionExecutionSuccess'] * self.seed_score_weights.get('execution', 0.0) # Use renamed weights dict
            return scores # Return early

        scores['message'] = f"Exec OK ({action_type})."

        # 2. Goal Progress Score
        try:
            # Assumes snapshot key is 'seedSensory' (needs consistency check w/ core.py)
            initial_sensory = initial_state_snapshot.get('seedSensory')
            if initial_sensory and post_action_sensory_input:
                scores['goalProgress'] = self._calculate_goal_progress(initial_sensory, post_action_sensory_input, current_goal)
                scores['message'] += f" GoalProg={scores['goalProgress']:.2f}."
            elif initial_sensory and not post_action_sensory_input:
                 logger.warning("Cannot calculate goal progress accurately: Post-action sensory data unavailable.")
                 scores['goalProgress'] = 0.0; scores['message'] += " GoalProg=N/A (No Post-State)."
            else:
                logger.warning("Cannot calculate goal progress: Missing initial sensory state data.")
                scores['goalProgress'] = 0.0
        except Exception as e:
            logger.warning(f"Goal progress calculation error: {e}", exc_info=True)
            scores['goalProgress'] = 0.0; scores['message'] += " (GoalProg Error)."

        # 3. Resource Efficiency Score
        scores['resourceEfficiency'] = 0.5 # Neutral default
        # Use renamed duration key from core.py
        duration = execution_result.get('details', {}).get('seed_action_duration_sec')
        if duration is not None and isinstance(duration, (int, float)) and duration >= 0:
            scores['resourceEfficiency'] = max(0.0, 1.0 - (duration / 60.0)) # Normalize against 60s
            scores['message'] += f" Eff={scores['resourceEfficiency']:.2f} (Dur={duration:.1f}s)."

        # 4. Overall Weighted Score
        # Use renamed weights dict
        overall = (scores['actionExecutionSuccess'] * self.seed_score_weights.get('execution', 0.0) +
                   scores['goalProgress'] * self.seed_score_weights.get('goal_prog', 0.0) +
                   scores['resourceEfficiency'] * self.seed_score_weights.get('efficiency', 0.0))
        scores['overall_success'] = max(0.0, min(1.0, overall)) # Clamp to [0, 1]
        scores['message'] += f" Overall={scores['overall_success']:.3f}."
        return scores


    def _calculate_goal_progress(self, pre_sensory: dict, post_sensory: dict, goal: dict) -> float:
        """ Calculates goal progress based on actual pre/post state changes towards the current goal. """
        # This logic remains the same, compares pre/post target status based on goal type
        pre_target = pre_sensory.get('target_status', {})
        post_target = post_sensory.get('target_status', {})
        goal_type = goal.get('target', 'unknown')
        goal_path = goal.get('path')
        goal_hint = goal.get('content_hint')
        progress = 0.0
        target_path_matches = goal_path and post_target.get('path') == goal_path
        if not target_path_matches: return 0.0

        if goal_type == 'create_file':
            pre_exists = pre_target.get('exists', False) and pre_target.get('type') == 'file'; post_exists = post_target.get('exists', False) and post_target.get('type') == 'file'
            pre_hint = pre_target.get('hint_present', False) if goal_hint else True; post_hint = post_target.get('hint_present', False) if goal_hint else True
            if post_hint is None and goal_hint: post_hint = False # Treat unknown as not present
            post_state_achieved = post_exists and post_hint; pre_state_achieved = pre_exists and pre_hint
            if post_state_achieved and not pre_state_achieved: progress = 1.0
            elif post_state_achieved and pre_state_achieved: progress = 0.1
            elif post_exists and not pre_exists: progress = 0.5
            elif post_exists and post_hint and not pre_hint: progress = 0.7
            elif not post_exists and pre_exists: progress = -0.5
            elif post_exists and not post_hint and pre_hint: progress = -0.3
        elif goal_type == 'delete_file':
            pre_exists = pre_target.get('exists', False); post_exists = post_target.get('exists', False)
            if not post_exists and pre_exists: progress = 1.0
            elif not post_exists and not pre_exists: progress = 0.1
            elif post_exists and not pre_exists: progress = -0.5
        else: logger.debug(f"Goal progress calculation does not handle goal type '{goal_type}'.")
        return np.clip(progress, -1.0, 1.0)

# --- END OF FILE seed/evaluator.py ---