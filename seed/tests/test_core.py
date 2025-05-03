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
from unittest.mock import MagicMock, patch, call # Using standard unittest.mock

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
    mocks["memory_system"].get_learning_parameter.return_value = 0.6 # Default temp
    # Make get_learning_parameter return specific values based on input string
    def get_param_side_effect(param_name):
        if param_name == "llm_query_temperature.value":
            # Allow tests to configure this via the mock if needed, else default
            if hasattr(mocks["memory_system"], '_mock_temp_return'):
                return mocks["memory_system"]._mock_temp_return
            return 0.6
        elif param_name == "evaluation_weights":
             # Return a structure similar to the config for evaluation tests
             return copy.deepcopy(SEED_LEARNING_PARAMETERS["evaluation_weights"])
        # Add other specific param mocks if needed by future tests
        return None # Default for unhandled params
    mocks["memory_system"].get_learning_parameter.side_effect = get_param_side_effect
    mocks["memory_system"].update_learning_parameter.return_value = True # Assume success
    mocks["memory_system"].get_behavioral_rules.return_value = {} # Default empty rules
    mocks["memory_system"].find_lifelong_by_criteria.return_value = [] # Default empty memory search

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
# @pytest.mark.skip(reason="Analysis method _analyze_memory_patterns not yet implemented or tests not written")
def test_analyze_memory_patterns_basic(seed_core_instance, mock_dependencies):
    """ Test basic functionality of _analyze_memory_patterns. """
    mock_memory = mock_dependencies["memory_system"]

    # Setup mock memory entries
    mock_evals = [
        {'key': 'SEED_Evaluation_1', 'data': {'action_summary': 'READ_FILE: x', 'overall_success': 1.0}},
        {'key': 'SEED_Evaluation_2', 'data': {'action_summary': 'WRITE_FILE: y', 'overall_success': 0.2}},
        {'key': 'SEED_Evaluation_3', 'data': {'action_summary': 'READ_FILE: z', 'overall_success': 0.8}},
        {'key': 'SEED_Evaluation_4', 'data': {'action_summary': 'EXECUTE_VM_COMMAND: ls', 'overall_success': 0.0}}, # Failed command
    ]
    mock_errors = [
        {'key': 'SEED_Action_EXECUTE_VM_COMMAND_Error', 'data': {'result_reason': 'permission_denied'}, 'tags': ['Error']},
        {'key': 'SEED_Action_WRITE_FILE_Error', 'data': {'result_msg': 'Disk full'}, 'tags': ['Error']},
        {'key': 'SEED_LLMError_1', 'data': {'error': 'API Timeout'}, 'tags': ['Error', 'LLM']},
        {'key': 'SEED_Action_EXECUTE_VM_COMMAND_Error2', 'data': {'result_reason': 'permission_denied'}, 'tags': ['Error']}, # Repeated error
    ]

    # Configure mock find_lifelong_by_criteria
    def mock_find_side_effect(filter_func, limit=None, newest_first=False):
        # Crude check based on string representation of lambda
        if "SEED_Evaluation" in repr(filter_func):
            return mock_evals
        elif "Error" in repr(filter_func) or "Critical" in repr(filter_func):
            return mock_errors
        return []
    mock_memory.find_lifelong_by_criteria.side_effect = mock_find_side_effect

    # Call the method
    analysis_result = seed_core_instance._analyze_memory_patterns(history_limit=10) # Use limit

    # Assertions
    assert analysis_result is not None
    assert analysis_result.get("error") is None

    # Check success rates
    rates = analysis_result.get("action_success_rates", {})
    assert "READ_FILE" in rates
    assert rates["READ_FILE"]["count"] == 2
    assert pytest.approx(rates["READ_FILE"]["avg_success"]) == 0.9 # (1.0 + 0.8) / 2
    assert "WRITE_FILE" in rates
    assert rates["WRITE_FILE"]["count"] == 1
    assert pytest.approx(rates["WRITE_FILE"]["avg_success"]) == 0.2
    assert "EXECUTE_VM_COMMAND" in rates
    assert rates["EXECUTE_VM_COMMAND"]["count"] == 1
    assert pytest.approx(rates["EXECUTE_VM_COMMAND"]["avg_success"]) == 0.0

    # Check common errors (should prioritize 'reason', then 'error', then 'msg')
    errors = analysis_result.get("common_errors", [])
    assert len(errors) > 0
    # Check if 'permission_denied' is the top error (count 2)
    assert errors[0] == ('permission_denied', 2)
    # Check presence of others (order might vary if counts are equal)
    assert ('Msg: Disk full...' in [e[0] for e in errors]) or ('Msg: Disk full' in [e[0] for e in errors]) # Handle potential truncation
    assert ('Error: API Timeout...' in [e[0] for e in errors]) or ('Error: API Timeout' in [e[0] for e in errors])

    # Check that memory.log was called to store the analysis
    mock_memory.log.assert_any_call(
        "SEED_InternalAnalysis",
        analysis_result, # Should log the result it calculated
        tags=["Seed", "Analysis", "InternalState"]
    )

def test_analyze_memory_patterns_no_data(seed_core_instance, mock_dependencies):
    """ Test _analyze_memory_patterns when memory returns no relevant entries. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory.find_lifelong_by_criteria.return_value = [] # Simulate no history

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

    assert len(hypotheses) >= 1 # Expect at least one hypothesis
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
    last_action = {"action_type": "NO_OP"}
    last_eval = {"overall_success": 1.0}
    hypotheses = seed_core_instance._generate_failure_hypotheses(last_action, last_eval)
    assert len(hypotheses) == 0
    # Ensure log was NOT called
    assert call("SEED_FailureHypotheses", unittest.mock.ANY, tags=unittest.mock.ANY) not in mock_memory.log.call_args_list


# Test _propose_improvement_hypotheses
def test_propose_improvement_hypotheses_low_success(seed_core_instance, mock_dependencies):
    """ Test improvement hypotheses for actions with low success rates. """
    mock_memory = mock_dependencies["memory_system"]
    analysis = {
        "action_success_rates": {
            "READ_FILE": {"avg_success": 0.9, "count": 10},
            "MODIFY_CORE_CODE": {"avg_success": 0.1, "count": 5}, # Low success
            "TEST_CORE_CODE": {"avg_success": 0.4, "count": 2} # Low count
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
    # Set initial temp low
    mock_memory._mock_temp_return = 0.3
    seed_core_instance._recent_eval_scores = collections.deque([0.1, 0.2, 0.1, 0.0, 0.2], maxlen=5) # Avg < 0.3

    last_eval = {"overall_success": 0.25} # Current eval doesn't matter as much as avg
    seed_core_instance._perform_automated_learning(last_eval)

    # Check update_learning_parameter was called correctly
    expected_temp = 0.3 + 0.05 # Initial temp + step
    mock_memory.update_learning_parameter.assert_called_once_with(
        "llm_query_temperature.value",
        pytest.approx(expected_temp)
    )

def test_perform_automated_learning_decrease_temp(seed_core_instance, mock_dependencies):
    """ Test that LLM temperature decreases with high average success. """
    mock_memory = mock_dependencies["memory_system"]
    # Set initial temp high
    mock_memory._mock_temp_return = 0.9
    seed_core_instance._recent_eval_scores = collections.deque([0.9, 0.85, 0.95, 1.0, 0.8], maxlen=5) # Avg > 0.8

    last_eval = {"overall_success": 0.88}
    seed_core_instance._perform_automated_learning(last_eval)

    # Check update_learning_parameter was called correctly
    expected_temp = 0.9 - 0.05 # Initial temp - step
    mock_memory.update_learning_parameter.assert_called_once_with(
        "llm_query_temperature.value",
        pytest.approx(expected_temp)
    )

def test_perform_automated_learning_no_change(seed_core_instance, mock_dependencies):
    """ Test that LLM temperature doesn't change with moderate average success. """
    mock_memory = mock_dependencies["memory_system"]
    mock_memory._mock_temp_return = 0.6
    seed_core_instance._recent_eval_scores = collections.deque([0.5, 0.6, 0.7, 0.5, 0.4], maxlen=5) # Avg between 0.3 and 0.8

    last_eval = {"overall_success": 0.55}
    seed_core_instance._perform_automated_learning(last_eval)

    # Check update_learning_parameter was NOT called
    mock_memory.update_learning_parameter.assert_not_called()

def test_perform_automated_learning_buffer_not_full(seed_core_instance, mock_dependencies):
    """ Test that no learning happens until the score buffer is full. """
    mock_memory = mock_dependencies["memory_system"]
    seed_core_instance._recent_eval_scores = collections.deque([0.1, 0.9], maxlen=5) # Only 2 scores

    last_eval = {"overall_success": 0.5}
    seed_core_instance._perform_automated_learning(last_eval)

    mock_memory.update_learning_parameter.assert_not_called()

# --- run_strategic_cycle Test Placeholder (Keep Skipped) ---

@pytest.mark.skip(reason="run_strategic_cycle tests require complex integration setup.")
def test_run_strategic_cycle_basic(seed_core_instance, mock_dependencies):
    """ TODO: Implement integration tests for run_strategic_cycle. """
    pass


# TODO: Add tests for remaining methods if logic becomes complex:
# - _build_llm_prompt (verify context inclusion)
# - _validate_direct_action_llm_response (more edge cases)
# - _execute_seed_action (more action types, error conditions)
# - _check_behavioral_rules (more complex patterns)
# - Core code modification helpers (if modified from original)

# --- END OF FILE RSIAI0/seed/tests/test_core.py ---