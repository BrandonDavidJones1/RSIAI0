# --- START OF FILE RSIAI0/seed/tests/test_core.py ---

# seed/tests/test_core.py
"""
Unit tests for the Seed_Core component.
Focuses on testing the logic within Seed_Core methods in isolation,
using mocked dependencies.
"""

import pytest
import time
import copy
import collections # For deque in Seed_Core internal state
# >>> FIX: Import 'call' and 'ANY' from unittest.mock <<< # Already imported via pytest fixtures? Keep for clarity.
from unittest.mock import MagicMock, patch, call, ANY

# Import the class to be tested
from seed.core import Seed_Core

# Import other necessary components for type hinting or potential real instances
# from seed.memory_system import MemorySystem # Example
from seed.config import SEED_LEARNING_PARAMETERS # Needed for default learning params

# --- Fixtures ---

@pytest.fixture
def mock_dependencies():
    """ Creates a dictionary of mocked dependencies for Seed_Core. """
    mocks = {
        "llm_service": MagicMock(name="MockLLMService"),
        "vm_service": MagicMock(name="MockVMService"),
        "memory_system": MagicMock(name="MockMemorySystem"),
        "success_evaluator": MagicMock(name="MockSuccessEvaluator"),
        "sensory_refiner": MagicMock(name="MockSensoryRefiner"),
    }
    # Default configurations for mocked methods
    # Ensure side_effect handles empty name for get_learning_parameter
    def get_param_side_effect(param_name):
        if not param_name: # Handle request for all params
             # Return a deep copy resembling the structure loaded/saved
             return copy.deepcopy(SEED_LEARNING_PARAMETERS)
        if param_name == "llm_query_temperature.value":
            if hasattr(mocks["memory_system"], '_mock_temp_return'):
                return mocks["memory_system"]._mock_temp_return
            # Fallback to default if not specifically mocked for a test
            return SEED_LEARNING_PARAMETERS.get("llm_query_temperature", {}).get("value", 0.6)
        elif param_name == "evaluation_weights":
             return copy.deepcopy(SEED_LEARNING_PARAMETERS.get("evaluation_weights", {}))
        elif param_name == "rule_application_mode.value":
            return SEED_LEARNING_PARAMETERS.get("rule_application_mode", {}).get("value", "log_suggestion")
        # Add other specific param mocks if needed
        # Simulate nested access for evaluation weights if requested directly
        elif param_name.startswith("evaluation_weights.") and param_name.endswith(".value"):
             parts = param_name.split('.')
             if len(parts) == 3:
                 weight_name = parts[1]
                 return SEED_LEARNING_PARAMETERS.get("evaluation_weights", {}).get(weight_name, {}).get("value", 0.0)
        return None # Default for unhandled params
    mocks["memory_system"].get_learning_parameter.side_effect = get_param_side_effect
    mocks["memory_system"].update_learning_parameter.return_value = True # Assume success
    mocks["memory_system"].get_behavioral_rules.return_value = {} # Default empty rules
    # Default side effect for find_lifelong_by_criteria, might be overridden in specific tests
    mocks["memory_system"].find_lifelong_by_criteria.return_value = []

    mocks["sensory_refiner"].refine.return_value = {"summary": {}, "cwd": "/app", "target_status": {}} # Basic refine result
    mocks["vm_service"].get_state.return_value = {"mode": "simulation", "cwd": "/app", "filesystem": {}, "resources": {}} # Basic vm state
    mocks["success_evaluator"].evaluate_seed_action_success.return_value = {"overall_success": 1.0, "message": "Mock Eval OK"} # Default success

    return mocks

@pytest.fixture
def seed_core_instance(mock_dependencies):
    """ Creates a Seed_Core instance with mocked dependencies. """
    core = Seed_Core(
        llm_service=mock_dependencies["llm_service"],
        vm_service=mock_dependencies["vm_service"],
        memory_system=mock_dependencies["memory_system"],
        success_evaluator=mock_dependencies["success_evaluator"],
        sensory_refiner=mock_dependencies["sensory_refiner"],
    )
    core.set_initial_state({"target": "test_goal", "description": "Initial goal for testing"})
    # Reset internal state for tests if needed
    core._recent_eval_scores = collections.deque(maxlen=5)
    return core

# --- Basic Tests (Unchanged) ---

def test_seed_core_instantiation(seed_core_instance, mock_dependencies):
    """ Test if Seed_Core can be instantiated correctly with mocks. """
    assert seed_core_instance is not None
    assert seed_core_instance.llm_service == mock_dependencies["llm_service"]
    assert seed_core_instance.vm_service == mock_dependencies["vm_service"]
    assert seed_core_instance.memory == mock_dependencies["memory_system"]
    assert seed_core_instance.success_evaluator == mock_dependencies["success_evaluator"]
    assert seed_core_instance.sensory_refiner == mock_dependencies["sensory_refiner"]
    assert seed_core_instance.cycle_count == 0

def test_set_goal(seed_core_instance, mock_dependencies):
    """ Test the set_goal method. """
    mock_memory = mock_dependencies["memory_system"]
    initial_goal = copy.deepcopy(seed_core_instance.current_goal)
    new_goal = {"target": "new_test_target", "description": "Updated goal for testing"}
    success = seed_core_instance.set_goal(new_goal)
    assert success is True
    assert seed_core_instance.current_goal == new_goal
    mock_memory.log.assert_called_with(
        "seed_goal_set",
        {"old_goal": initial_goal, "new_goal": new_goal},
        tags=['Seed', 'Goal']
    )

def test_set_invalid_goal(seed_core_instance, mock_dependencies):
    """ Test setting an invalid goal. """
    mock_memory = mock_dependencies["memory_system"]
    initial_goal = copy.deepcopy(seed_core_instance.current_goal)
    invalid_goal = {"target": "missing_description"}
    log_call_count_before = mock_memory.log.call_count
    success = seed_core_instance.set_goal(invalid_goal)
    assert success is False
    assert seed_core_instance.current_goal == initial_goal
    assert mock_memory.log.call_count == log_call_count_before

# --- Tests for NEW Internal Methods ---

# Test _analyze_memory_patterns
# >>> START OF MODIFIED TEST (simplified robust mock side_effect) <<<
def test_analyze_memory_patterns_basic(seed_core_instance, mock_dependencies):
    """ Test basic functionality of _analyze_memory_patterns with simplified robust mock. """
    mock_memory = mock_dependencies["memory_system"]

    # Setup mock memory entries
    mock_evals = [
        {'key': 'SEED_Evaluation_1', 'data': {'action_summary': 'READ_FILE: x', 'overall_success': 1.0}},
        {'key': 'SEED_Evaluation_2', 'data': {'action_summary': 'WRITE_FILE: y', 'overall_success': 0.2}},
        {'key': 'SEED_Evaluation_3', 'data': {'action_summary': 'READ_FILE: z', 'overall_success': 0.8}},
        {'key': 'SEED_Evaluation_4', 'data': {'action_summary': 'EXECUTE_VM_COMMAND: ls', 'overall_success': 0.0}},
    ]
    mock_errors = [
        {'key': 'SEED_Action_EXECUTE_VM_COMMAND_Error', 'data': {'result_reason': 'permission_denied'}, 'tags': ['Error']},
        {'key': 'SEED_Action_WRITE_FILE_Error', 'data': {'result_msg': 'Disk full'}, 'tags': ['Error']},
        {'key': 'SEED_LLMError_1', 'data': {'error': 'API Timeout'}, 'tags': ['Error', 'LLM']},
        {'key': 'SEED_Action_EXECUTE_VM_COMMAND_Error2', 'data': {'result_reason': 'permission_denied'}, 'tags': ['Error']},
    ]

    # --- Simplified Robust Mock Side Effect ---
    dummy_eval_entry = {'key': 'SEED_Evaluation_dummy', 'tags': []}
    # No need for dummy_error_entry with simplified logic

    def simplified_robust_mock_find(filter_func, limit=None, newest_first=False):
        try:
            # Check if the filter matches an evaluation entry
            if filter_func(dummy_eval_entry):
                # print("DEBUG: Mock returning mock_evals") # Optional
                return copy.deepcopy(mock_evals[:limit] if limit else mock_evals)
            else:
                # Assume it's the error filter for this specific test context
                # print("DEBUG: Mock assuming error filter, returning mock_errors") # Optional
                actual_errors = [copy.deepcopy(e) for e in mock_errors if filter_func(e)]
                return actual_errors[:limit] if limit else actual_errors
        except Exception as e:
            print(f"ERROR in simplified_robust_mock_find applying filter: {e}")
            pytest.fail(f"Filter function call failed within mock side effect: {e}")
        return [] # Should not be reached if filter logic is sound

    mock_memory.find_lifelong_by_criteria.side_effect = simplified_robust_mock_find
    # --- End Simplified Robust Mock ---

    # Call the method
    analysis_result = seed_core_instance._analyze_memory_patterns(history_limit=10)

    # Assertions
    assert analysis_result is not None
    assert analysis_result.get("error") is None, f"Analysis resulted in error: {analysis_result.get('error')}"

    # Check success rates
    rates = analysis_result.get("action_success_rates", {})
    expected_rates = {
        "READ_FILE": {"avg_success": 0.9, "count": 2},
        "WRITE_FILE": {"avg_success": 0.2, "count": 1},
        "EXECUTE_VM_COMMAND": {"avg_success": 0.0, "count": 1}
    }
    assert rates == expected_rates

    # Check common errors
    errors = analysis_result.get("common_errors", [])
    assert len(errors) == 3 # Expecting top 3, error filter should now work
    # Convert list of tuples to set of tuples for easier comparison if order isn't guaranteed
    errors_set = set(errors)
    # Check counts and presence (order might vary based on Counter implementation details)
    assert errors[0] == ('permission_denied', 2) # Top error should have count 2
    assert len(errors_set) == 3 # Ensure 3 unique errors were found
    assert any(e[0].startswith('Msg: Disk full') for e in errors_set)
    assert any(e[0].startswith('Error: API Timeout') for e in errors_set)

    # Check memory log call
    mock_memory.log.assert_any_call(
        "SEED_InternalAnalysis",
        analysis_result,
        tags=["Seed", "Analysis", "InternalState"]
    )
# >>> END OF MODIFIED TEST <<<


def test_analyze_memory_patterns_no_data(seed_core_instance, mock_dependencies):
    """ Test _analyze_memory_patterns when memory returns no relevant entries. """
    mock_memory = mock_dependencies["memory_system"]
    # Use the robust mock, but ensure it returns empty lists when called
    def empty_side_effect(filter_func, limit=None, newest_first=False):
        return []
    mock_memory.find_lifelong_by_criteria.side_effect = empty_side_effect

    analysis_result = seed_core_instance._analyze_memory_patterns()

    assert analysis_result is not None
    assert analysis_result.get("error") is None
    assert analysis_result.get("action_success_rates") == {}
    assert analysis_result.get("common_errors") == []

# Test _generate_failure_hypotheses
def test_generate_failure_hypotheses_file_not_found(seed_core_instance, mock_dependencies):
    """ Test failure hypotheses generation for file_not_found reason. """
    mock_memory = mock_dependencies["memory_system"]
    last_action = {"action_type": "READ_FILE", "path": "nonexistent/file.txt"}
    last_eval = {"overall_success": 0.0, "details": {"reason": "file_not_found"}}

    hypotheses = seed_core_instance._generate_failure_hypotheses(last_action, last_eval)

    assert len(hypotheses) >= 1
    assert any("path 'nonexistent/file.txt' might be incorrect" in h for h in hypotheses)
    mock_memory.log.assert_called_with(
        "SEED_FailureHypotheses",
        {"hypotheses": hypotheses, "failed_action": last_action, "evaluation": last_eval},
        tags=["Seed", "Hypothesis", "Error"]
    )

def test_generate_failure_hypotheses_permission(seed_core_instance, mock_dependencies):
    """ Test failure hypotheses generation for permission_denied reason. """
    last_action = {"action_type": "WRITE_FILE", "path": "/root/secret.txt"}
    last_eval = {"overall_success": 0.1, "details": {"reason": "permission_denied"}}
    hypotheses = seed_core_instance._generate_failure_hypotheses(last_action, last_eval)
    assert len(hypotheses) >= 1
    assert any("lacked necessary permissions" in h for h in hypotheses)

def test_generate_failure_hypotheses_no_failure(seed_core_instance, mock_dependencies):
    """ Test that no hypotheses are generated for successful actions. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory.log.reset_mock()
    last_action = {"action_type": "NO_OP"}
    last_eval = {"overall_success": 1.0}
    hypotheses = seed_core_instance._generate_failure_hypotheses(last_action, last_eval)
    assert len(hypotheses) == 0
    found_call = False
    for actual_call in mock_memory.log.call_args_list:
        args, kwargs = actual_call
        if args and args[0] == "SEED_FailureHypotheses":
            found_call = True
            break
    assert not found_call


# Test _propose_improvement_hypotheses
def test_propose_improvement_hypotheses_low_success(seed_core_instance, mock_dependencies):
    """ Test improvement hypotheses for actions with low success rates. """
    mock_memory = mock_dependencies["memory_system"]
    analysis = {
        "action_success_rates": {
            "READ_FILE": {"avg_success": 0.9, "count": 10},
            "MODIFY_CORE_CODE": {"avg_success": 0.1, "count": 5},
            "TEST_CORE_CODE": {"avg_success": 0.4, "count": 2}
        },
        "common_errors": []
    }
    hypotheses = seed_core_instance._propose_improvement_hypotheses(analysis)
    assert len(hypotheses) == 1
    assert "Action 'MODIFY_CORE_CODE' has low success (0.10 over 5 tries)" in hypotheses[0]
    mock_memory.log.assert_called_with(
        "SEED_ImprovementHypotheses",
        {"hypotheses": hypotheses, "triggering_analysis": analysis},
        tags=["Seed", "Hypothesis", "Improvement"]
    )

def test_propose_improvement_hypotheses_common_error(seed_core_instance, mock_dependencies):
    """ Test improvement hypotheses for common errors. """
    analysis = {
        "action_success_rates": {},
        "common_errors": [("timeout", 5), ("invalid_argument", 2)]
    }
    hypotheses = seed_core_instance._propose_improvement_hypotheses(analysis)
    assert len(hypotheses) == 1
    assert "most common error reason is 'timeout'" in hypotheses[0]

def test_propose_improvement_hypotheses_analysis_error(seed_core_instance, mock_dependencies):
    """ Test improvement hypotheses when analysis itself failed. """
    analysis = {"error": "Memory read failed"}
    hypotheses = seed_core_instance._propose_improvement_hypotheses(analysis)
    assert len(hypotheses) == 1
    assert "Internal analysis function itself failed ('Memory read failed')" in hypotheses[0]


# Test _perform_automated_learning
def test_perform_automated_learning_increase_temp(seed_core_instance, mock_dependencies):
    """ Test that LLM temperature increases with low average success. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory._mock_temp_return = 0.3
    seed_core_instance._recent_eval_scores = collections.deque([0.1, 0.2, 0.1, 0.0, 0.2], maxlen=5)

    last_eval = {"overall_success": 0.25}
    seed_core_instance._perform_automated_learning(last_eval)

    expected_temp = 0.3 + 0.05
    mock_memory.update_learning_parameter.assert_called_once_with(
        "llm_query_temperature.value",
        pytest.approx(expected_temp)
    )
    if hasattr(mock_memory, '_mock_temp_return'):
        delattr(mock_memory, '_mock_temp_return')


def test_perform_automated_learning_decrease_temp(seed_core_instance, mock_dependencies):
    """ Test that LLM temperature decreases with high average success. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory._mock_temp_return = 0.9
    seed_core_instance._recent_eval_scores = collections.deque([0.9, 0.85, 0.95, 1.0, 0.8], maxlen=5)

    last_eval = {"overall_success": 0.88}
    seed_core_instance._perform_automated_learning(last_eval)

    expected_temp = 0.9 - 0.05
    mock_memory.update_learning_parameter.assert_called_once_with(
        "llm_query_temperature.value",
        pytest.approx(expected_temp)
    )
    if hasattr(mock_memory, '_mock_temp_return'):
        delattr(mock_memory, '_mock_temp_return')

def test_perform_automated_learning_no_change(seed_core_instance, mock_dependencies):
    """ Test that LLM temperature doesn't change with moderate average success. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory._mock_temp_return = 0.6
    seed_core_instance._recent_eval_scores = collections.deque([0.5, 0.6, 0.7, 0.5, 0.4], maxlen=5)

    last_eval = {"overall_success": 0.55}
    seed_core_instance._perform_automated_learning(last_eval)

    mock_memory.update_learning_parameter.assert_not_called()
    if hasattr(mock_memory, '_mock_temp_return'):
        delattr(mock_memory, '_mock_temp_return')

def test_perform_automated_learning_buffer_not_full(seed_core_instance, mock_dependencies):
    """ Test that no learning happens until the score buffer is full. """
    mock_memory = mock_dependencies["memory_system"]
    seed_core_instance._recent_eval_scores = collections.deque([0.1, 0.9], maxlen=5)

    last_eval = {"overall_success": 0.5}
    seed_core_instance._perform_automated_learning(last_eval)

    mock_memory.update_learning_parameter.assert_not_called()

# --- run_strategic_cycle Test Placeholder ---

@pytest.mark.skip(reason="run_strategic_cycle tests require complex integration setup.")
def test_run_strategic_cycle_basic(seed_core_instance, mock_dependencies):
    """ TODO: Implement integration tests for run_strategic_cycle. """
    pass

# --- END OF FILE RSIAI0/seed/tests/test_core.py ---