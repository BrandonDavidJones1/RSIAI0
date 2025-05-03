# --- START OF FILE RSIAI0/seed/tests/test_memory.py ---

# seed/tests/test_memory.py
"""
Unit tests for the MemorySystem component.
"""

import pytest
import os
import time
import copy
import uuid
import pathlib
import pickle

# Import the class to be tested and relevant config
# Assuming pytest runs from the project root (RSIAI0/)
from seed.memory_system import MemorySystem, MemoryEntry
from seed.config import (
    MEMORY_MAX_EPISODIC_SIZE, MEMORY_MAX_LIFELONG_SIZE,
    SEED_LEARNING_PARAMETERS, MEMORY_LIFELONG_EVENT_TYPES,
    MEMORY_LIFELONG_TAGS, VECTOR_SEARCH_ENABLED_CONFIG # Check if vector search is meant to be enabled
)

# --- Fixtures ---

@pytest.fixture
def temp_save_file(tmp_path: pathlib.Path) -> str:
    """ Provides a temporary file path for saving/loading memory. """
    # Use a subdirectory within tmp_path if needed, e.g., tmp_path / "memory_tests"
    # tmp_path.mkdir(exist_ok=True)
    return str(tmp_path / "test_memory.pkl")

@pytest.fixture
def default_config():
    """ Returns a copy of the default learning parameters for comparison. """
    return copy.deepcopy(SEED_LEARNING_PARAMETERS)

@pytest.fixture
def memory_system(temp_save_file: str) -> MemorySystem:
    """ Creates a fresh MemorySystem instance using a temporary save file. """
    # Pass the temp file path in config override
    # Ensure vector search is explicitly disabled for most tests unless specifically testing it
    # (which requires libraries and setup not assumed here)
    ms = MemorySystem(config={'save_file_path': temp_save_file, 'vector_search_enabled': False})
    # Clear any state potentially loaded from a leftover file in rare cases
    ms.clear_all_memory()
    return ms

# --- Initialization Tests ---

def test_memory_system_init_defaults(memory_system: MemorySystem, temp_save_file: str):
    """ Test basic initialization with default settings. """
    assert memory_system.max_episodic_size == MEMORY_MAX_EPISODIC_SIZE
    assert memory_system.max_lifelong_size == MEMORY_MAX_LIFELONG_SIZE
    assert memory_system.save_file_path == temp_save_file
    assert len(memory_system._episodic_memory) == 0
    assert len(memory_system._lifelong_memory) == 2 # Should only contain learning params & rules state initially
    assert len(memory_system._lifelong_keys_by_age) == 2
    assert memory_system.vector_search_enabled is False # Based on fixture config override
    assert memory_system._learning_params_key in memory_system._lifelong_memory
    assert memory_system._behavioral_rules_key in memory_system._lifelong_memory

def test_memory_system_init_loads_defaults_first_time(temp_save_file: str, default_config: dict):
    """ Test that defaults are loaded and saved if no file exists. """
    assert not os.path.exists(temp_save_file)
    ms = MemorySystem(config={'save_file_path': temp_save_file})
    assert os.path.exists(temp_save_file) # Save should happen on init if file doesn't exist
    assert ms._learning_parameters == default_config
    assert ms._behavioral_rules == {}
    # Check if internal state is in memory
    assert ms.get_lifelong_memory(ms._learning_params_key) is not None
    assert ms.get_lifelong_memory(ms._behavioral_rules_key) is not None
    assert ms.get_lifelong_memory(ms._learning_params_key)['data'] == default_config
    assert ms.get_lifelong_memory(ms._behavioral_rules_key)['data'] == {}


# --- Episodic Memory Tests ---

def test_add_episodic_basic(memory_system: MemorySystem):
    """ Test adding a single episodic memory entry. """
    data = {"event": "test", "value": 1}
    tags = ["test_tag"]
    entry_id = memory_system.add_episodic_memory(data, tags)
    assert entry_id is not None
    assert len(memory_system._episodic_memory) == 1
    latest = memory_system.get_latest_episodic(1)[0]
    assert latest['id'] == entry_id
    assert latest['data'] == data
    assert latest['tags'] == tags
    assert time.time() - latest['timestamp'] < 1 # Check timestamp is recent

def test_add_episodic_none(memory_system: MemorySystem):
    """ Test adding None data. """
    entry_id = memory_system.add_episodic_memory(None)
    assert entry_id is None
    assert len(memory_system._episodic_memory) == 0

def test_episodic_max_size(memory_system: MemorySystem):
    """ Test that episodic memory respects max_size. """
    max_size = memory_system.max_episodic_size
    for i in range(max_size + 5):
        memory_system.add_episodic_memory({"index": i})
    assert len(memory_system._episodic_memory) == max_size
    latest = memory_system.get_latest_episodic(1)[0]
    oldest = list(memory_system._episodic_memory)[0]
    assert latest['data']['index'] == max_size + 4
    assert oldest['data']['index'] == 5 # The first 5 should have been evicted

def test_find_episodic(memory_system: MemorySystem):
    """ Test finding episodic entries by criteria. """
    memory_system.add_episodic_memory({"type": "A", "value": 1}, ["tagA"])
    memory_system.add_episodic_memory({"type": "B", "value": 2}, ["tagB"])
    memory_system.add_episodic_memory({"type": "A", "value": 3}, ["tagA", "tagC"])

    results_A = memory_system.find_episodic_by_criteria(lambda e: e['data'].get('type') == 'A')
    assert len(results_A) == 2
    assert results_A[0]['data']['value'] == 1
    assert results_A[1]['data']['value'] == 3

    results_tagA = memory_system.find_episodic_by_criteria(lambda e: "tagA" in e.get('tags', []))
    assert len(results_tagA) == 2

    results_limit = memory_system.find_episodic_by_criteria(lambda e: e['data'].get('type') == 'A', limit=1)
    assert len(results_limit) == 1
    assert results_limit[0]['data']['value'] == 1

    results_newest = memory_system.find_episodic_by_criteria(lambda e: e['data'].get('type') == 'A', newest_first=True)
    assert len(results_newest) == 2
    assert results_newest[0]['data']['value'] == 3 # Newest first

def test_get_episodic_by_id(memory_system: MemorySystem):
    """ Test retrieving episodic entry by ID. """
    id1 = memory_system.add_episodic_memory({"v": 1})
    id2 = memory_system.add_episodic_memory({"v": 2})
    assert memory_system.get_episodic_by_id(id1)['data']['v'] == 1
    assert memory_system.get_episodic_by_id(id2)['data']['v'] == 2
    assert memory_system.get_episodic_by_id("nonexistent") is None

# --- Lifelong Memory Tests ---

def test_add_lifelong_basic(memory_system: MemorySystem):
    """ Test adding a new lifelong memory entry. """
    key = "test_key_1"
    data = {"value": "hello"}
    tags = ["lifelong_test"]
    entry_id = memory_system.add_lifelong_memory(key, data, tags)
    assert entry_id is not None
    assert key in memory_system._lifelong_memory
    entry = memory_system.get_lifelong_memory(key)
    assert entry is not None
    assert entry['id'] == entry_id
    assert entry['key'] == key
    assert entry['data'] == data
    assert entry['tags'] == tags
    assert time.time() - entry['timestamp'] < 1
    assert list(memory_system._lifelong_keys_by_age)[-1] == key # Should be newest

def test_add_lifelong_update(memory_system: MemorySystem):
    """ Test updating an existing lifelong memory entry. """
    key = "test_key_update"
    data1 = {"v": 1, "original": True}
    data2 = {"v": 2, "updated": True}
    id1 = memory_system.add_lifelong_memory(key, data1, ["tag1"])
    ts1 = memory_system.get_lifelong_memory(key)['timestamp']
    time.sleep(0.01) # Ensure timestamp changes
    id2 = memory_system.add_lifelong_memory(key, data2, ["tag2"])

    assert id2 == id1 # ID should be preserved on update
    entry = memory_system.get_lifelong_memory(key)
    assert entry is not None
    assert entry['data'] == data2 # Data should be updated
    assert entry['tags'] == ["tag2"] # Tags should be replaced
    assert entry['timestamp'] > ts1 # Timestamp should update
    assert list(memory_system._lifelong_keys_by_age)[-1] == key # Should be newest

def test_add_lifelong_eviction(memory_system: MemorySystem):
    """ Test lifelong memory eviction when max_size is reached. """
    # Temporarily reduce max_lifelong_size for this test
    original_max = memory_system.max_lifelong_size
    # Account for the 2 internal state keys
    test_max = 3 + 2 # Allow 3 user entries + 2 internal
    memory_system.max_lifelong_size = test_max
    memory_system._lifelong_keys_by_age = deque(memory_system._lifelong_keys_by_age, maxlen=test_max)

    keys = [f"key_{i}" for i in range(test_max + 1)] # Add one more than allowed
    for key in keys:
        memory_system.add_lifelong_memory(key, {"data": key})

    assert len(memory_system._lifelong_memory) == test_max
    assert len(memory_system._lifelong_keys_by_age) == test_max
    # key_0 should have been evicted (assuming internal keys were added first)
    assert "key_0" not in memory_system._lifelong_memory
    assert keys[1] in memory_system._lifelong_memory # key_1 should still be there
    assert keys[-1] in memory_system._lifelong_memory # Last key added should be there

    # Restore original max size
    memory_system.max_lifelong_size = original_max
    memory_system._lifelong_keys_by_age = deque(memory_system._lifelong_keys_by_age, maxlen=original_max)

def test_remove_lifelong(memory_system: MemorySystem):
    """ Test removing a lifelong entry. """
    key = "to_remove"
    memory_system.add_lifelong_memory(key, {"data": "test"})
    assert key in memory_system._lifelong_memory
    removed = memory_system.remove_lifelong_memory(key)
    assert removed is True
    assert key not in memory_system._lifelong_memory
    assert key not in memory_system._lifelong_keys_by_age
    removed_again = memory_system.remove_lifelong_memory(key)
    assert removed_again is False

def test_remove_internal_state_key_fails(memory_system: MemorySystem):
    """ Test that internal state keys cannot be removed via remove_lifelong_memory. """
    params_key = memory_system._learning_params_key
    rules_key = memory_system._behavioral_rules_key
    assert params_key in memory_system._lifelong_memory
    assert rules_key in memory_system._lifelong_memory

    removed_params = memory_system.remove_lifelong_memory(params_key)
    removed_rules = memory_system.remove_lifelong_memory(rules_key)

    assert removed_params is False
    assert removed_rules is False
    assert params_key in memory_system._lifelong_memory # Should still exist
    assert rules_key in memory_system._lifelong_memory

# --- Persistence Tests ---

def test_save_load_memory(memory_system: MemorySystem, temp_save_file: str, default_config: dict):
    """ Test saving and loading the entire memory state. """
    # 1. Add diverse data
    epi_id = memory_system.add_episodic_memory({"type": "ep1"}, ["tag_epi"])
    ll_key1 = "ll_key_persist"
    ll_id = memory_system.add_lifelong_memory(ll_key1, {"persist": True}, ["tag_ll"])
    param_name = "llm_query_temperature.value"
    memory_system.update_learning_parameter(param_name, 0.99)
    rule_id = memory_system.add_behavioral_rule({"trigger_pattern": {"goal.target": "test"}, "suggested_response": "do_test"})

    # Get state before saving
    episodic_before = list(memory_system._episodic_memory)
    lifelong_before = copy.deepcopy(memory_system._lifelong_memory)
    keys_before = list(memory_system._lifelong_keys_by_age)
    params_before = copy.deepcopy(memory_system._learning_parameters)
    rules_before = copy.deepcopy(memory_system._behavioral_rules)

    # 2. Save memory
    memory_system.save_memory()
    assert os.path.exists(temp_save_file)

    # 3. Create a new instance and load
    ms_loaded = MemorySystem(config={'save_file_path': temp_save_file})
    # load_memory is called during init

    # 4. Assert state matches
    assert list(ms_loaded._episodic_memory) == episodic_before
    # Compare lifelong dicts excluding potential ephemeral fields like 'embedding' if added later
    assert {k: {ik: iv for ik, iv in v.items() if ik != 'embedding'} for k, v in ms_loaded._lifelong_memory.items()} == \
           {k: {ik: iv for ik, iv in v.items() if ik != 'embedding'} for k, v in lifelong_before.items()}
    assert list(ms_loaded._lifelong_keys_by_age) == keys_before
    assert ms_loaded._learning_parameters == params_before
    assert ms_loaded.get_learning_parameter(param_name) == 0.99 # Check specific loaded value
    assert ms_loaded._behavioral_rules == rules_before
    assert rule_id in ms_loaded._behavioral_rules

    # Check loaded internal keys directly
    assert ms_loaded.get_lifelong_memory(ms_loaded._learning_params_key)['data'] == params_before
    assert ms_loaded.get_lifelong_memory(ms_loaded._behavioral_rules_key)['data'] == rules_before

# --- Learning Parameter Tests ---

def test_get_learning_parameter(memory_system: MemorySystem, default_config: dict):
    """ Test retrieving learning parameters. """
    # Test getting specific value
    assert memory_system.get_learning_parameter("llm_query_temperature.value") == default_config["llm_query_temperature"]["value"]
    # Test getting nested value
    assert memory_system.get_learning_parameter("evaluation_weights.goal_prog.value") == default_config["evaluation_weights"]["goal_prog"]["value"]
    # Test getting category dict
    assert memory_system.get_learning_parameter("evaluation_weights") == default_config["evaluation_weights"]
    # Test getting all params
    assert memory_system.get_learning_parameter("") == default_config
    # Test getting non-existent
    assert memory_system.get_learning_parameter("nonexistent.value") is None
    assert memory_system.get_learning_parameter("evaluation_weights.nonexistent.value") is None

def test_update_learning_parameter_valid(memory_system: MemorySystem):
    """ Test updating parameters with valid values. """
    assert memory_system.update_learning_parameter("llm_query_temperature.value", 0.8) is True
    assert memory_system.get_learning_parameter("llm_query_temperature.value") == 0.8
    assert memory_system.update_learning_parameter("evaluation_weights.goal_prog.value", 0.1) is True
    assert memory_system.get_learning_parameter("evaluation_weights.goal_prog.value") == 0.1
    assert memory_system.update_learning_parameter("rule_application_mode.value", "pre_llm_filter") is True
    assert memory_system.get_learning_parameter("rule_application_mode.value") == "pre_llm_filter"

def test_update_learning_parameter_invalid(memory_system: MemorySystem, default_config: dict):
    """ Test updating parameters with invalid values (type, bounds, options). """
    temp_key = "llm_query_temperature.value"
    weight_key = "evaluation_weights.goal_prog.value"
    mode_key = "rule_application_mode.value"
    initial_temp = default_config["llm_query_temperature"]["value"]
    initial_weight = default_config["evaluation_weights"]["goal_prog"]["value"]
    initial_mode = default_config["rule_application_mode"]["value"]

    # Invalid type
    assert memory_system.update_learning_parameter(temp_key, "not_a_float") is False
    assert memory_system.get_learning_parameter(temp_key) == initial_temp
    assert memory_system.update_learning_parameter(mode_key, 123) is False
    assert memory_system.get_learning_parameter(mode_key) == initial_mode

    # Out of bounds
    assert memory_system.update_learning_parameter(temp_key, 2.0) is True # Should clamp
    assert memory_system.get_learning_parameter(temp_key) == 1.5 # Max bound
    memory_system.update_learning_parameter(temp_key, initial_temp) # Reset for next test
    assert memory_system.update_learning_parameter(weight_key, -0.5) is True # Should clamp
    assert memory_system.get_learning_parameter(weight_key) == 0.0 # Min bound
    memory_system.update_learning_parameter(weight_key, initial_weight) # Reset

    # Invalid option
    assert memory_system.update_learning_parameter(mode_key, "invalid_option") is False
    assert memory_system.get_learning_parameter(mode_key) == initial_mode

    # Invalid name format
    assert memory_system.update_learning_parameter("llm_query_temperature", 0.7) is False # Missing .value
    assert memory_system.update_learning_parameter("evaluation_weights.goal_prog", 0.7) is False # Missing .value

# --- Behavioral Rule Tests ---

def test_add_behavioral_rule(memory_system: MemorySystem):
    """ Test adding a new behavioral rule. """
    rule_data = {"trigger_pattern": {"key": "value"}, "suggested_response": "action1"}
    rule_id = memory_system.add_behavioral_rule(rule_data)
    assert rule_id is not None
    assert rule_id.startswith("rule_")
    rules = memory_system.get_behavioral_rules()
    assert rule_id in rules
    assert rules[rule_id]['trigger_pattern'] == rule_data['trigger_pattern']
    assert rules[rule_id]['suggested_response'] == rule_data['suggested_response']
    assert rules[rule_id]['trigger_count'] == 0
    assert rules[rule_id]['last_triggered_timestamp'] is None

def test_add_behavioral_rule_with_id(memory_system: MemorySystem):
    """ Test adding a rule specifying an ID. """
    rule_id = "custom_rule_1"
    rule_data = {"trigger_pattern": {"a": 1}, "suggested_response": "action2", "rule_id": rule_id}
    returned_id = memory_system.add_behavioral_rule(rule_data)
    assert returned_id == rule_id
    rules = memory_system.get_behavioral_rules()
    assert rule_id in rules
    assert rules[rule_id]['trigger_pattern'] == rule_data['trigger_pattern']

def test_update_behavioral_rule(memory_system: MemorySystem):
    """ Test updating an existing rule by providing the same ID. """
    rule_id = "rule_to_update"
    rule_data1 = {"trigger_pattern": {"a": 1}, "suggested_response": "actionA", "rule_id": rule_id}
    rule_data2 = {"trigger_pattern": {"b": 2}, "suggested_response": "actionB", "rule_id": rule_id}
    memory_system.add_behavioral_rule(rule_data1)
    ts1 = memory_system.get_behavioral_rules()[rule_id]['creation_timestamp']
    memory_system.update_rule_trigger_stats(rule_id) # Increment count
    time.sleep(0.01)
    memory_system.add_behavioral_rule(rule_data2) # This should update

    rules = memory_system.get_behavioral_rules()
    assert len(rules) == 1 # Should still be only one rule
    assert rule_id in rules
    assert rules[rule_id]['trigger_pattern'] == rule_data2['trigger_pattern']
    assert rules[rule_id]['suggested_response'] == rule_data2['suggested_response']
    assert rules[rule_id]['trigger_count'] == 1 # Stats should persist through update
    assert rules[rule_id]['creation_timestamp'] == ts1 # Creation time preserved
    assert rules[rule_id]['last_updated_timestamp'] > ts1 # Update time changed

def test_remove_behavioral_rule(memory_system: MemorySystem):
    """ Test removing a rule. """
    rule_id = memory_system.add_behavioral_rule({"trigger_pattern": {}, "suggested_response": "test"})
    assert rule_id in memory_system.get_behavioral_rules()
    removed = memory_system.remove_behavioral_rule(rule_id)
    assert removed is True
    assert rule_id not in memory_system.get_behavioral_rules()
    removed_again = memory_system.remove_behavioral_rule(rule_id)
    assert removed_again is False

def test_update_rule_trigger_stats(memory_system: MemorySystem):
    """ Test updating trigger stats for a rule. """
    rule_id = memory_system.add_behavioral_rule({"trigger_pattern": {}, "suggested_response": "test"})
    rule_before = memory_system.get_behavioral_rules()[rule_id]
    assert rule_before['trigger_count'] == 0
    assert rule_before['last_triggered_timestamp'] is None

    memory_system.update_rule_trigger_stats(rule_id)
    rule_after = memory_system.get_behavioral_rules()[rule_id]
    assert rule_after['trigger_count'] == 1
    assert rule_after['last_triggered_timestamp'] is not None
    ts1 = rule_after['last_triggered_timestamp']
    time.sleep(0.01)

    memory_system.update_rule_trigger_stats(rule_id)
    rule_after2 = memory_system.get_behavioral_rules()[rule_id]
    assert rule_after2['trigger_count'] == 2
    assert rule_after2['last_triggered_timestamp'] > ts1

# --- General Log / Search / Clear Tests ---

def test_log_routing(memory_system: MemorySystem):
    """ Test that log routes events correctly based on config. """
    # Event type configured as lifelong
    lifelong_type = list(MEMORY_LIFELONG_EVENT_TYPES)[0]
    ll_key_prefix = lifelong_type.split('_')[0] # Use prefix for check
    memory_system.log(lifelong_type, {"data": 1})
    assert len(memory_system._episodic_memory) == 0
    assert any(k.startswith(ll_key_prefix) for k in memory_system._lifelong_memory.keys())

    # Tag configured as lifelong
    lifelong_tag = list(MEMORY_LIFELONG_TAGS)[0]
    memory_system.log("NonLifelongEventType", {"data": 2}, tags=[lifelong_tag, "other_tag"])
    assert len(memory_system._episodic_memory) == 0
    assert any(k.startswith("NonLifelongEventType") for k in memory_system._lifelong_memory.keys())

    # Neither type nor tag is lifelong
    memory_system.log("EpisodicEventType", {"data": 3}, tags=["other_tag"])
    assert len(memory_system._episodic_memory) == 1
    assert not any(k.startswith("EpisodicEventType") for k in memory_system._lifelong_memory.keys())
    assert memory_system.get_latest_episodic(1)[0]['data'] == {"event_type": "EpisodicEventType", "details": {"data": 3}}

def test_search_memory_text(memory_system: MemorySystem):
    """ Test basic text search. """
    memory_system.add_episodic_memory({"text": "unique_episodic_phrase"}, ["searchable"])
    memory_system.add_lifelong_memory("ll_search_key", {"text": "unique_lifelong_phrase"}, ["searchable"])
    memory_system.add_lifelong_memory("ll_other_key", {"other": "content"}) # Non-matching

    # Search both
    results_both = memory_system.search_memory_text("unique", search_type='both')
    assert len(results_both) == 2
    assert any("unique_episodic_phrase" in str(e) for e in results_both)
    assert any("unique_lifelong_phrase" in str(e) for e in results_both)

    # Search episodic only
    results_epi = memory_system.search_memory_text("unique_episodic_phrase", search_type='episodic')
    assert len(results_epi) == 1
    assert "unique_episodic_phrase" in str(results_epi[0])

    # Search lifelong only
    results_ll = memory_system.search_memory_text("unique_lifelong_phrase", search_type='lifelong')
    assert len(results_ll) == 1
    assert "unique_lifelong_phrase" in str(results_ll[0])

    # No results
    results_none = memory_system.search_memory_text("nonexistent_term")
    assert len(results_none) == 0

def test_clear_all_memory(memory_system: MemorySystem, default_config: dict):
    """ Test clearing all memory including learning state. """
    memory_system.add_episodic_memory({"data": "ep"})
    memory_system.add_lifelong_memory("ll_key", {"data": "ll"})
    memory_system.update_learning_parameter("llm_query_temperature.value", 1.0)
    memory_system.add_behavioral_rule({"trigger_pattern": {}, "suggested_response": "test"})

    assert len(memory_system._episodic_memory) > 0
    assert len(memory_system._lifelong_memory) > 2 # More than just internal state
    assert len(memory_system._behavioral_rules) > 0

    memory_system.clear_all_memory()

    assert len(memory_system._episodic_memory) == 0
    # Should contain ONLY the internal state keys after clear and reset
    assert len(memory_system._lifelong_memory) == 2
    assert len(memory_system._lifelong_keys_by_age) == 2
    assert memory_system._learning_params_key in memory_system._lifelong_memory
    assert memory_system._behavioral_rules_key in memory_system._lifelong_memory
    # Check learning state reset to defaults
    assert memory_system._learning_parameters == default_config
    assert memory_system._behavioral_rules == {}

# TODO: Add tests for retrieve_context (more complex setup)
# TODO: Add tests for vector search functionality IF it gets enabled later

# --- END OF FILE RSIAI0/seed/tests/test_memory.py ---