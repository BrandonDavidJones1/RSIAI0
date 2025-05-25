# RSIAI/seed/vm_service.py
"""
Defines the Seed_VMService class for simulating or interacting with
an external system (VM, container, OS). Executes commands, reads/writes files,
and probes state to provide snapshots for Seed core reasoning.
Uses configured command whitelist and timeouts for safety in real mode.
Includes case-insensitive path handling for robustness.
"""
import os
import time
import json
import traceback
import copy
import shlex
import logging
import subprocess
import re
from pathlib import PurePosixPath, Path # Added Path for local CWD check
from typing import Dict, Any, Optional, Tuple, Callable, List, Union # Added Union

# --- Configuration ---
# Use relative import now
from .config import (
    VM_SERVICE_USE_REAL, VM_SERVICE_DOCKER_CONTAINER,
    VM_SERVICE_COMMAND_TIMEOUT_SEC, VM_SERVICE_ALLOWED_REAL_COMMANDS,
    VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE # <<< --- IMPORT YOUR MAGIC VALUE
)

# --- Docker Integration (Optional) ---
try:
    import docker
    from docker.errors import NotFound as DockerNotFound, APIError as DockerAPIError
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None # type: ignore
    DockerNotFound = Exception # type: ignore
    DockerAPIError = Exception # type: ignore
    logging.getLogger(__name__).info("Docker library not found, Docker interaction disabled for VMService.") # Use logging directly before logger is set

logger = logging.getLogger(__name__)

# Renamed class
class Seed_VMService:
    """ Simulates or interacts with an external system (VM/Docker/Subprocess). """
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 use_real_system: bool = VM_SERVICE_USE_REAL,
                 docker_container_name: Optional[str] = VM_SERVICE_DOCKER_CONTAINER):
        """ Initializes the VMService, choosing mode and setting up connections/state. """
        self.config = config if config else {}
        # Core commands needed for probing and basic file ops in real mode
        self._core_real_probes = ['sh', 'pwd', 'ls', 'df', 'stat', 'grep', 'head', 'tail', 'top', 'cat', 'echo', 'rm', 'touch', 'mkdir', 'mv', 'cp', 'printf', 'python3', 'pip', 'pytest']

        # Initialize self.allowed_real_commands based on the global config.
        # The actual permission check in _execute_real_command will use the global config value directly.
        if isinstance(VM_SERVICE_ALLOWED_REAL_COMMANDS, str) and VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
            self.allowed_real_commands_config_value: Union[List[str], str] = VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE
            # For local storage/reference if needed, but primary check uses global
            self._effective_allowed_commands_list: List[str] = list(set(self._core_real_probes)) # Fallback/reference, not used for check if magic value active
            logger.warning("VMService initialized with 'allow all commands' mode (magic value in config).")
        elif isinstance(VM_SERVICE_ALLOWED_REAL_COMMANDS, list):
            self.allowed_real_commands_config_value = VM_SERVICE_ALLOWED_REAL_COMMANDS
            self._effective_allowed_commands_list = list(set(self.allowed_real_commands_config_value + self._core_real_probes))
        else:
            logger.error(f"VM_SERVICE_ALLOWED_REAL_COMMANDS has an unexpected type: {type(VM_SERVICE_ALLOWED_REAL_COMMANDS)}. Defaulting to core probes only.")
            self.allowed_real_commands_config_value = list(set(self._core_real_probes)) # Ensure it's a list
            self._effective_allowed_commands_list = list(set(self.allowed_real_commands_config_value))


        # Determine actual mode based on config and library availability
        self.docker_container_name: Optional[str] = docker_container_name if DOCKER_AVAILABLE and use_real_system else None
        self.use_real_system: bool = use_real_system and (DOCKER_AVAILABLE or not self.docker_container_name) # Real mode only if Docker available or no container specified (for subprocess)

        self.docker_client = None
        self.docker_container = None
        self._simulated_state: Optional[Dict] = None # Renamed internal var

        self.command_timeout_sec: int = VM_SERVICE_COMMAND_TIMEOUT_SEC
        # CWD initialization
        self._real_system_cwd: str = '/app' # Default assumed CWD for real systems/containers
        self._simulated_system_cwd: str = '/app' # Default sim CWD

        if self.use_real_system:
            mode_name = "Subprocess"
            if self.docker_container_name:
                # This check now happens only if docker was specified and available
                if self._connect_docker():
                     mode_name = f"Docker ({self.docker_container_name})"
                else:
                    logger.error(f"Failed to connect to configured Docker container '{self.docker_container_name}'. Falling back to Subprocess mode.")
                    self.docker_container_name = None # Ensure docker isn't used
                    mode_name = "Subprocess (Docker Fallback)"
            else:
                # Using subprocess mode, verify initial CWD exists locally
                logger.info("Using Subprocess mode. Verifying initial CWD...")
                try:
                    local_cwd_path = Path(self._real_system_cwd)
                    if not local_cwd_path.is_dir():
                         logger.warning(f"Initial Subprocess CWD '{self._real_system_cwd}' does not exist or is not a directory locally. Defaulting CWD to current directory.")
                         self._real_system_cwd = str(Path.cwd())
                    else:
                         # Resolve to absolute path
                         self._real_system_cwd = str(local_cwd_path.resolve())
                except Exception as e:
                     logger.error(f"Error verifying local CWD '{self._real_system_cwd}': {e}. Defaulting to current directory.")
                     self._real_system_cwd = str(Path.cwd())
                logger.info(f"Subprocess mode initial CWD set to: {self._real_system_cwd}")

            effective_allowance_msg = 'ALL COMMANDS (Magic Value)' if self.allowed_real_commands_config_value == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE else sorted(self._effective_allowed_commands_list)
            logger.info(f"Initializing Seed VMService (Real System Mode: {mode_name}). Effective command allowance: {effective_allowance_msg}")
        else:
            logger.info("Initializing Seed VMService (Simulation Mode)...")
            self._initialize_simulation()

    def _connect_docker(self) -> bool:
        """ Attempts to connect to the specified Docker container. """
        # Should only be called if self.docker_container_name is set and DOCKER_AVAILABLE is True
        if not self.docker_container_name or not DOCKER_AVAILABLE:
             logger.error("Internal error: _connect_docker called inappropriately.")
             return False

        logger.info(f"Attempting connection to Docker container: {self.docker_container_name}")
        try:
            self.docker_client = docker.from_env(timeout=20)
            if not self.docker_client.ping():
                logger.error("Docker daemon is not responding.")
                self.docker_client = None
                return False

            self.docker_container = self.docker_client.containers.get(self.docker_container_name)

            if self.docker_container.status.lower() != 'running':
                logger.error(f"Docker container '{self.docker_container_name}' not running (Status: {self.docker_container.status}).")
                self.docker_container = None
                return False

            # Test connectivity and get initial CWD if possible
            test_result_dict = self._docker_exec_context("pwd")
            if test_result_dict.get('success'):
                container_cwd = test_result_dict.get('stdout','').strip()
                if container_cwd: # Only set if pwd returns something valid
                    logger.info(f"Docker container initial CWD detected as: '{container_cwd}'. Setting internal CWD.")
                    self._real_system_cwd = container_cwd # Set internal CWD to match container's default
                else:
                    logger.warning(f"Docker 'pwd' command returned empty. Using default CWD '{self._real_system_cwd}'.")
                logger.info(f"Successfully connected to Docker container '{self.docker_container_name}'.")
                return True
            else:
                logger.error(f"Docker test command 'pwd' failed. Exit Code: {test_result_dict.get('exit_code')}. Stderr: {test_result_dict.get('stderr','')}")
                self.docker_container = None
                return False
        except DockerNotFound:
            logger.error(f"Docker container '{self.docker_container_name}' not found.")
            self.docker_container = None; self.docker_client = None
            return False
        except DockerAPIError as api_err:
            logger.error(f"Docker API error connecting: {api_err}")
            self.docker_container = None; self.docker_client = None
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Docker: {e}", exc_info=True)
            self.docker_container = None; self.docker_client = None
            return False

    def _initialize_simulation(self):
        """ Sets up the initial simulated environment state including test files. """
        logger.info("Setting up simulated filesystem and state...")
        self._simulated_system_cwd = '/app'
        # Simplified base commands for simulation mode (core file ops)
        self._sim_available_commands = ['ls', 'pwd', 'cat', 'echo', 'touch', 'mkdir', 'rm', 'cd', 'cp', 'mv', 'pytest'] # Added pytest

        test_core_content = """
# RSIAI0/seed/tests/test_core.py
import pytest

def test_basic_core_functionality_alpha(): # Matches basic_core_tests
    assert True

def test_basic_core_functionality_beta(): # Matches basic_core_tests
    assert True

def test_advanced_core_feature_one():
    assert 1 == 1

def test_core_specific_module_interaction(): # Matches 'core' if path target is test_core.py
    assert "core" in __file__

def test_internal_analysis_component_basic(): # Matches 'internal_analysis'
    assert True

def test_internal_learning_mechanism_initial(): # Matches 'internal_learning'
    assert True

def test_core_edge_case_expected_fail_low_memory(): # Expected to fail by name
    assert False
"""
        test_memory_content = """
# RSIAI0/seed/tests/test_memory.py
import pytest

def test_memory_read_write(): # Matches 'memory' if path target is test_memory.py
    assert True

def test_memory_persistence_layer():
    assert True

def test_memory_query_performance_expected_fail_large_dataset(): # Expected to fail by name
    assert False
"""
        # Using standard read/write permissions for user now for simplicity
        self._simulated_state = {
            'timestamp': time.time(), 'cwd': self._simulated_system_cwd,
            'filesystem': {
                '/': {'type': 'directory', 'owner': 'root', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/tmp': {'type': 'directory', 'owner': 'user', 'perms': 'rwxrwxrwx', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/seed': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/seed/__init__.py': {'type': 'file', 'content': '# Seed package init\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 20},
                '/app/seed/core.py': {'type': 'file', 'content': '# RSIAI/seed/core.py\nprint("Simulated core.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 60},
                '/app/seed/memory_system.py': {'type': 'file', 'content': '# RSIAI/seed/memory_system.py\nprint("Simulated memory_system.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 70},
                '/app/seed/config.py': {'type': 'file', 'content': '# RSIAI/seed/config.py\nprint("Simulated config.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 64},
                '/app/seed/vm_service.py': {'type': 'file', 'content': '# RSIAI/seed/vm_service.py\nprint("Simulated vm_service.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 66},
                '/app/seed/evaluator.py': {'type': 'file', 'content': '# RSIAI/seed/evaluator.py\nprint("Simulated evaluator.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 68},
                '/app/seed/llm_service.py': {'type': 'file', 'content': '# RSIAI/seed/llm_service.py\nprint("Simulated llm_service.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 70},
                '/app/seed/main.py': {'type': 'file', 'content': '# RSIAI/seed/main.py\nprint("Simulated main.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 62},
                '/app/seed/sensory.py': {'type': 'file', 'content': '# RSIAI/seed/sensory.py\nprint("Simulated sensory.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 66},
                '/app/seed/verification.py': {'type': 'file', 'content': '# RSIAI/seed/verification.py\nprint("Simulated verification.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 72},
                '/app/seed/tests': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/seed/tests/__init__.py': {'type': 'file', 'content': '', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0},
                '/app/seed/tests/test_core.py': {'type': 'file', 'content': test_core_content, 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(test_core_content.encode())},
                '/app/seed/tests/test_memory.py': {'type': 'file', 'content': test_memory_content, 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(test_memory_content.encode())},
            },
            'resources': {'cpu_load_percent': 1.0, 'memory_usage_percent': 5.0, 'disk_usage_percent': 10.0},
            'last_command_result': None,
            'available_commands': list(self._sim_available_commands)
        }
        logger.info(f"Simulation Initialized. CWD: {self._simulated_system_cwd}. Sim commands: {self._sim_available_commands}")

    def _get_current_cwd(self) -> str:
        return self._real_system_cwd if self.use_real_system else self._simulated_system_cwd

    def _set_current_cwd(self, new_cwd: str):
        """ Sets the internal current working directory, ensuring it's absolute. """
        try:
            posix_path = PurePosixPath(new_cwd)
            if not posix_path.is_absolute():
                 current_path = PurePosixPath(self._get_current_cwd())
                 # Resolve path relative to current_path, handling '..' and '.'
                 # os.path.normpath can be used on the string representation for this.
                 # However, since we are strictly posix, PurePosixPath.resolve() (with strict=False if needed)
                 # or manual normalization is better.
                 # Example of manual normalization:
                 combined_path = current_path.joinpath(posix_path)
                 parts = []
                 for part in combined_path.parts:
                     if part == '.':
                         continue
                     if part == '..':
                         if parts and parts[-1] != '/': # Ensure we don't pop root
                             parts.pop()
                         elif not parts or parts == ['/']: # Already at root or empty
                             continue # Cannot go above root
                     else:
                         parts.append(part)

                 if not parts or parts == ['/']: # Handle cases like '/../' or just '/'
                     posix_path = PurePosixPath('/')
                 else:
                     # Reconstruct the path, ensuring it starts with '/' if parts[0] isn't '/'
                     # and join other parts. PurePosixPath handles multiple slashes.
                     posix_path = PurePosixPath('/' + '/'.join(p for p in parts if p != '/'))
                 logger.debug(f"Resolved relative CWD '{new_cwd}' to '{posix_path}'")

            normalized_cwd = str(posix_path)

            if not self.use_real_system and self._simulated_state:
                fs = self._simulated_state['filesystem']
                if normalized_cwd not in fs or fs[normalized_cwd].get('type') != 'directory':
                     logger.warning(f"Attempted to set CWD to non-existent/non-directory path in simulation: '{normalized_cwd}'. CWD unchanged.")
                     return # Do not change CWD if invalid in simulation

            # If real system, CWD change is optimistic here; actual commands will use this new CWD.
            # If a 'cd' command fails in real mode, CWD might not actually change on the target system.
            # This internal CWD is primarily for constructing absolute paths for subsequent commands.
            if self.use_real_system:
                self._real_system_cwd = normalized_cwd
            else:
                self._simulated_system_cwd = normalized_cwd
                if self._simulated_state: # Update simulation state if it exists
                    self._simulated_state['cwd'] = normalized_cwd

        except (TypeError, ValueError) as e:
            logger.error(f"Attempted to set invalid CWD '{new_cwd}': {e}")


    def _resolve_path(self, path_str: str) -> Optional[str]:
        current_dir = self._get_current_cwd()
        if not path_str: return current_dir # No path given, return current CWD
        try:
            cwd_path = PurePosixPath(current_dir)
            target_path: PurePosixPath

            if path_str == '~':
                target_path = PurePosixPath('/app') # Define your "home" directory
            elif path_str.startswith('~/'):
                target_path = PurePosixPath('/app') / path_str[2:]
            else:
                input_path = PurePosixPath(path_str)
                if input_path.is_absolute():
                    target_path = input_path
                else:
                    target_path = cwd_path / input_path

            # Normalize the path (handles .., ., //)
            # os.path.normpath is generally for the OS it runs on.
            # For consistent Posix paths, PurePosixPath operations are better.
            # A simple way to normalize with PurePosixPath like functionality:
            resolved_parts = []
            for part in target_path.parts:
                if part == '.':
                    continue
                if part == '..':
                    if resolved_parts and resolved_parts[-1] != '/': # Don't pop root
                        resolved_parts.pop()
                else:
                    resolved_parts.append(part)

            if not resolved_parts or (len(resolved_parts) == 1 and resolved_parts[0] == '/'):
                resolved_path_str = '/'
            else:
                # Join parts, skip leading '/' if present in the first part from target_path.parts
                # and prepend a single '/'
                # PurePosixPath('/' + '/'.join(...)) handles this well.
                # Filter out any empty strings that might result from multiple slashes if not handled by PurePosixPath
                actual_path_components = [p for p in resolved_parts if p and p != '/']
                if not actual_path_components: # e.g. target was '/' or '/./'
                    resolved_path_str = '/'
                else:
                    resolved_path_str = '/' + '/'.join(actual_path_components)


            logger.debug(f"Resolved path '{path_str}' from '{current_dir}' to '{resolved_path_str}'")
            return resolved_path_str
        except Exception as e:
            logger.error(f"Path resolution error for '{path_str}' from '{current_dir}': {e}", exc_info=True)
            return None

    def _sim_check_permissions(self, path: str, action: str = 'read') -> Tuple[bool, str]:
        # Simplified: In simulation, assume all actions are permitted if path exists.
        # More complex permission model could be added if needed.
        logger.debug(f"Sim permission check skipped for action '{action}' on '{path}' (Always returning True).")
        return (True, "")


    def _find_case_insensitive_match(self, parent_path_str: str, filename_str: str) -> Optional[str]:
        logger.debug(f"Attempting case-insensitive search for '{filename_str}' in '{parent_path_str}'")
        matches = []
        target_filename_lower = filename_str.lower()

        if not self.use_real_system: # Simulation mode
            if not self._simulated_state: return None
            fs = self._simulated_state['filesystem']
            # Ensure parent_path_str is normalized for comparison
            norm_parent_path = str(PurePosixPath(parent_path_str)) # Normalize to ensure consistent format like /app not /app/
            if norm_parent_path == '.': norm_parent_path = self._get_current_cwd() # Handle relative parent path

            for path_key in fs.keys():
                try:
                    entry_path = PurePosixPath(path_key)
                    # Compare normalized parent of entry_path with norm_parent_path
                    if str(entry_path.parent) == norm_parent_path:
                         entry_filename = entry_path.name
                         if entry_filename.lower() == target_filename_lower:
                             matches.append(path_key) # path_key is already the correct case from simulation keys
                except Exception as e:
                    logger.warning(f"Error processing path '{path_key}' during case-insensitive search: {e}")
                    continue
        else: # Real system mode
            # This requires 'ls' command to be allowed
            if 'ls' not in self._effective_allowed_commands_list and VM_SERVICE_ALLOWED_REAL_COMMANDS != VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
                logger.warning("Cannot perform case-insensitive search: 'ls' command not allowed.")
                return None # Or fallback to exact match only

            # Use a command that lists one item per line, including hidden files (except . and ..)
            ls_cmd = f"ls -1a {shlex.quote(parent_path_str)}"
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            ls_res = self._execute_real_command(ls_cmd, exec_func) # Use internal exec to respect CWD and context

            if ls_res.get('success'):
                potential_files = [line for line in ls_res.get('stdout', '').splitlines() if line and line not in ['.', '..']]
                for potential_file in potential_files:
                    if potential_file.lower() == target_filename_lower:
                        # Construct the full path with the case found by 'ls'
                        try:
                           full_match_path = str(PurePosixPath(parent_path_str) / potential_file)
                           matches.append(full_match_path)
                        except Exception as path_e:
                            logger.error(f"Error constructing path for matched file '{potential_file}' in '{parent_path_str}': {path_e}")
            else:
                logger.warning(f"Case-insensitive search: 'ls' command failed in '{parent_path_str}'. Stderr: {ls_res.get('stderr')}")

        if len(matches) == 1:
            logger.info(f"Found unique case-insensitive match for '{filename_str}' in '{parent_path_str}': '{matches[0]}'")
            return matches[0]
        elif len(matches) > 1:
            logger.warning(f"Found multiple ({len(matches)}) case-insensitive matches for '{filename_str}' in '{parent_path_str}': {matches}. Ambiguous.")
            # Optionally, you could return the list of matches if the caller can handle ambiguity.
            # For now, returning None as it's safer for single-target operations.
            return None
        else:
            logger.debug(f"Found no case-insensitive match for '{filename_str}' in '{parent_path_str}'.")
            return None


    def _get_potential_matches(self, parent_path_str: str, filename_str: str) -> List[str]:
        """ Internal helper to list all case-insensitive matches for a filename within a parent path. """
        matches = []
        target_filename_lower = filename_str.lower()
        if not self.use_real_system: # Simulation
            if not self._simulated_state: return []
            fs = self._simulated_state['filesystem']
            norm_parent_path = str(PurePosixPath(parent_path_str))
            if norm_parent_path == '.': norm_parent_path = self._get_current_cwd()

            for path_key in fs.keys():
                if path_key == parent_path_str or path_key == '/': continue # Skip the parent itself or root if it's the parent
                try:
                    entry_path = PurePosixPath(path_key)
                    if str(entry_path.parent) == norm_parent_path:
                         entry_filename = entry_path.name
                         if entry_filename.lower() == target_filename_lower:
                             matches.append(path_key)
                except Exception: continue # Ignore errors from malformed paths if any
        else: # Real system
            if 'ls' not in self._effective_allowed_commands_list and VM_SERVICE_ALLOWED_REAL_COMMANDS != VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
                return []
            ls_cmd = f"ls -1a {shlex.quote(parent_path_str)}" # -1 for one per line, -a for hidden (excluding . and .. by filtering)
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            ls_res = self._execute_real_command(ls_cmd, exec_func)
            if ls_res.get('success'):
                potential_files = [line for line in ls_res.get('stdout', '').splitlines() if line and line not in ['.', '..']]
                for potential_file in potential_files:
                    if potential_file.lower() == target_filename_lower:
                        try:
                           full_match_path = str(PurePosixPath(parent_path_str) / potential_file)
                           matches.append(full_match_path)
                        except Exception: continue # Ignore errors constructing path
        return matches

    def execute_command(self, command_str: str) -> Dict[str, Any]:
        if self.use_real_system:
            exec_context = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            return self._execute_real_command(command_str, exec_context)
        else:
            return self._execute_simulated_command(command_str)

    def _sim_discover_tests(self, fs_state: Dict, base_paths: List[str], keyword_filter: Optional[str]) -> List[Tuple[str, str]]:
        """ Helper to discover simulated tests. """
        discovered_tests = []
        # Default to /app/seed/tests if no paths given or paths are empty
        search_roots_str = base_paths if base_paths and any(bp.strip() for bp in base_paths) else ["seed/tests"] # Ensure not empty list of empty strings

        search_roots_resolved = []
        for sr_str in search_roots_str:
            resolved = self._resolve_path(sr_str)
            if resolved:
                search_roots_resolved.append(resolved)
            else:
                logger.warning(f"Could not resolve pytest target path: {sr_str}")

        candidate_files = []
        for root_path_str in search_roots_resolved:
            # Check if the root_path_str itself is a file
            if root_path_str in fs_state and fs_state[root_path_str].get('type') == 'file':
                if PurePosixPath(root_path_str).name.startswith("test_") and root_path_str.endswith(".py"):
                    candidate_files.append(root_path_str)
            # Else, if it's a directory, search within it
            elif root_path_str in fs_state and fs_state[root_path_str].get('type') == 'directory':
                for path_key, item_info in fs_state.items():
                    if item_info.get('type') == 'file' and \
                       path_key.startswith(root_path_str + ('/' if root_path_str != '/' else '')) and \
                       PurePosixPath(path_key).name.startswith("test_") and \
                       path_key.endswith(".py"):
                        candidate_files.append(path_key)

        candidate_files = sorted(list(set(candidate_files))) # Unique and sorted files

        for test_file_path in candidate_files:
            content = fs_state.get(test_file_path, {}).get('content', '')
            # Simple regex to find test functions (can be improved)
            for line_num, line in enumerate(content.splitlines()):
                match = re.match(r'^\s*def\s+(test_[a-zA-Z0-9_]+)\s*\(', line)
                if match:
                    test_name = match.group(1)
                    if keyword_filter:
                        # Keyword check: in test name OR in file path (relative to CWD for user friendliness)
                        # OR if the keyword filter is a substring of the full node ID (file::test)
                        rel_file_path = str(PurePosixPath(test_file_path).relative_to(PurePosixPath(self._simulated_system_cwd)))
                        node_id = f"{rel_file_path}::{test_name}"
                        if keyword_filter in test_name or keyword_filter in rel_file_path or keyword_filter in node_id:
                            discovered_tests.append((test_file_path, test_name))
                    else:
                        discovered_tests.append((test_file_path, test_name))
        return discovered_tests

    def _sim_run_one_test(self, filepath: str, testname: str, keyword_filter: Optional[str]) -> Tuple[str, str]:
        """ Simulates running a single test and returns (status, testname_with_path). """
        status = "PASSED" # Default to PASSED
        # Apply heuristic pass/fail logic
        if "expected_fail" in testname.lower():
            status = "FAILED"

        # Specific keyword matching for suites (can be expanded)
        # This is a simplified heuristic. Real pytest has more complex collection.
        if keyword_filter:
            rel_file_path_for_nodeid = str(PurePosixPath(filepath).relative_to(PurePosixPath(self._simulated_system_cwd)))
            node_id = f"{rel_file_path_for_nodeid}::{testname}"

            # Check if the keyword filter DESELECTS the test.
            # This is a negation of the collection logic: if a filter is provided,
            # and the test DOESN'T match it, it's effectively deselected.
            # Pytest usually shows these as "deselected". Here, we just won't run them if not matched by filter.
            # The discover_tests should handle this initial filtering.
            # This _sim_run_one_test assumes it's a test that *was* selected.

            # Example of overriding status based on filter (if needed, but discover should handle selection)
            if "basic_core_tests" == keyword_filter:
                if not ("basic_core" in testname.lower() or "basic_core" in filepath.lower()):
                    pass # This test wouldn't have been discovered if filter was applied correctly
            elif "internal_analysis" == keyword_filter:
                 if not ("internal_analysis" in testname.lower()): pass
            elif "internal_learning" == keyword_filter:
                 if not ("internal_learning" in testname.lower()): pass
            elif "core" == keyword_filter:
                if not ("test_core.py" in filepath): pass
            elif "memory" == keyword_filter:
                if not ("test_memory.py" in filepath): pass

        # Construct the full test name as pytest would display it (relative path)
        relative_filepath = str(PurePosixPath(filepath).relative_to(PurePosixPath(self._simulated_system_cwd)))
        return status, f"{relative_filepath}::{testname}"


    def _sim_generate_pytest_output(self, collected_tests: List[Tuple[str,str]], ran_tests_results: List[Tuple[str,str]], verbose: bool, duration: float) -> Tuple[str, str, int]:
        """ Generates simulated pytest stdout, stderr, and exit code. """
        stdout_lines = [f"============================= test session starts =============================="]
        stdout_lines.append(f"platform sim -- Python sim, pytest sim, pluggy sim")
        stdout_lines.append(f"rootdir: {self._get_current_cwd()}")
        # plugins: ... (can be added)
        # collected N items / M deselected / P errors
        deselected_count = len(collected_tests) - len(ran_tests_results)
        collected_line = f"collected {len(collected_tests)} items"
        if deselected_count > 0:
            collected_line += f" / {deselected_count} deselected"
        stdout_lines.append(collected_line)
        stdout_lines.append("")

        passed_count = 0
        failed_count = 0

        # Output for each test that ran
        for status, testname_full in ran_tests_results:
            short_status_char = "." if status == "PASSED" else "F"
            if status == "PASSED": passed_count += 1
            elif status == "FAILED": failed_count +=1

            if verbose:
                stdout_lines.append(f"{testname_full} {status}")
            else:
                # In non-verbose, pytest prints chars, then full names for failures in summary
                # We'll append to a temporary line of status characters
                if 'status_line' not in locals() or not status_line: status_line = ""
                status_line += short_status_char
        if not verbose and 'status_line' in locals() and status_line:
            stdout_lines.append(status_line) # Print the accumulated status characters

        # Summary section
        if not ran_tests_results and deselected_count > 0:
            stdout_lines.append(f"============================ no tests ran due to deselection ============================")
            exit_code = 5 # No tests collected/ran due to deselection
        elif not ran_tests_results:
            stdout_lines.append(f"================================= no tests found ==================================")
            # stdout_lines.append(f"ERROR: InvocationError: ... (pytest exited with code 5).") # More detailed for no tests found
            exit_code = 5 # No tests found
        else:
            stdout_lines.append(f"============================== short test summary info ===============================")
            if failed_count > 0:
                 for status, testname_full in ran_tests_results:
                      if status == "FAILED":
                           stdout_lines.append(f"FAILED {testname_full}")
                           # Add simplified failure details
                           stdout_lines.append(f">       assert False, 'Simulated failure: {testname_full} was marked as expected_fail or did not match positive criteria'")
                           stdout_lines.append("") # Blank line after each failure detail block

            summary_parts = []
            if failed_count > 0: summary_parts.append(f"{failed_count} failed")
            if passed_count > 0: summary_parts.append(f"{passed_count} passed")
            if deselected_count > 0: summary_parts.append(f"{deselected_count} deselected")

            stdout_lines.append(f"========================= {', '.join(summary_parts)} in {duration:.2f}s =========================")
            exit_code = 1 if failed_count > 0 else 0

        return "\n".join(stdout_lines) + "\n", "", exit_code


    def _execute_simulated_command(self, command_str: str) -> Dict[str, Any]:
        current_cwd = self._get_current_cwd()
        logger.debug(f"VM Sim Exec (CWD: {current_cwd}): '{command_str}'")
        result={'success':False,'stdout':'','stderr':'','command':command_str,'reason':'','exit_code':1}
        if not self._simulated_state: result['stderr'],result['reason']="Sim not init.",'internal_error'; return result
        try: parts = shlex.split(command_str)
        except ValueError as e: result['stderr'],result['reason']=f"Command parse error: {e}",'parse_error'; return result
        cmd=parts[0] if parts else ''; args=parts[1:]
        if not cmd or cmd not in self._sim_available_commands: result['stderr'],result['reason']=f"Command not found/allowed in sim: {cmd}",'illegal_command'; return result

        try:
            fs = self._simulated_state['filesystem']; resolve_path_func = self._resolve_path; find_match_func = self._find_case_insensitive_match
            if cmd == 'pwd':
                result['stdout'], result['success'], result['exit_code'] = current_cwd, True, 0
            elif cmd == 'cd':
                target_dir_str = args[0] if args else '~'; resolved_path = resolve_path_func(target_dir_str); target_path = resolved_path
                if resolved_path and resolved_path not in fs:
                    parent = str(PurePosixPath(resolved_path).parent); filename = PurePosixPath(resolved_path).name; found_match = find_match_func(parent, filename)
                    if found_match: target_path = found_match; logger.debug(f"cd: Using corrected path '{target_path}' for '{target_dir_str}'")
                    else: result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found'; return result
                if target_path and target_path in fs and fs[target_path].get('type') == 'directory': self._set_current_cwd(target_path); result['success'], result['exit_code'] = True, 0 # _set_current_cwd updates _simulated_state['cwd']
                elif target_path and target_path in fs: result['stderr'], result['reason'] = f"cd: Not a directory: {target_dir_str}", 'is_not_directory'
                elif not target_path: result['stderr'], result['reason'] = f"cd: Invalid path resolution for: {target_dir_str}", 'invalid_path'
                else: result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found'
            elif cmd == 'ls':
                target_path_str = args[0] if args else '.'; resolved_path = resolve_path_func(target_path_str); target_path = resolved_path
                if resolved_path and resolved_path not in fs:
                    parent = str(PurePosixPath(resolved_path).parent); filename = PurePosixPath(resolved_path).name; found_match = find_match_func(parent, filename)
                    if found_match: target_path = found_match; logger.debug(f"ls: Using corrected path '{target_path}' for '{target_path_str}'")
                    else: result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found'; return result
                if target_path and target_path in fs:
                    item_info = fs[target_path]
                    if item_info.get('type') == 'directory':
                        parent_path_for_ls = target_path
                        # Correctly list contents of the target directory
                        contents = [PurePosixPath(n).name for n, f_entry in fs.items()
                                    if str(PurePosixPath(n).parent) == parent_path_for_ls and n != parent_path_for_ls] # exclude self if parent is /
                        result['stdout'], result['success'], result['exit_code'] = "\n".join(sorted(contents)), True, 0
                    else: result['stdout'], result['success'], result['exit_code'] = PurePosixPath(target_path).name, True, 0
                elif not target_path: result['stderr'], result['reason'] = f"ls: invalid path resolution for '{target_path_str}'", 'invalid_path'
                else: result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found'
            elif cmd == 'cat':
                if not args: result['stderr'], result['reason'] = "cat: missing file operand", 'missing_args'; return result
                p = resolve_path_func(args[0]); target_p = p
                if p and p not in fs:
                    parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name; found_match = find_match_func(parent, filename)
                    if found_match: target_p = found_match
                    else: result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found'; return result
                if target_p and target_p in fs:
                    info = fs[target_p]
                    if info.get('type') == 'file': result['stdout'], result['success'], result['exit_code'] = info.get('content',''), True, 0
                    else: result['stderr'], result['reason'] = f"cat: {args[0]}: Is a directory", 'is_directory'
                elif not target_p: result['stderr'], result['reason'] = f"cat: invalid path resolution for '{args[0]}'", 'invalid_path'
                else: result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found'
            elif cmd == 'touch':
                if not args: result['stderr'], result['reason'] = "touch: missing file operand", 'missing_args'; return result
                p = resolve_path_func(args[0]); target_p = p
                if p and p not in fs:
                    parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name; found_match = find_match_func(parent, filename)
                    if found_match: target_p = found_match; logger.debug(f"touch: Targeting existing file/dir '{target_p}'")

                if target_p: # target_p could be the original p or the found_match
                    if target_p in fs: fs[target_p]['mtime'] = time.time()
                    else: fs[target_p] = {'type': 'file', 'content': '', 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0}
                    result['success'], result['exit_code'] = True, 0
                else: # Path resolution failed or no match found for creation
                    result['stderr'], result['reason'] = f"touch: Invalid path resolution for: {args[0]} or cannot create", 'invalid_path'
            elif cmd == 'mkdir':
                 if not args: result['stderr'], result['reason'] = "mkdir: missing operand", 'missing_args'; return result
                 p = resolve_path_func(args[0])
                 if p:
                     if p in fs: result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': File exists", 'file_exists'
                     else:
                         parent_p_str = str(PurePosixPath(p).parent)
                         actual_parent_p = parent_p_str
                         if parent_p_str not in fs: # If direct parent string not in fs, try case-insensitive match
                             grandparent = str(PurePosixPath(parent_p_str).parent); parent_name = PurePosixPath(parent_p_str).name
                             parent_match = find_match_func(grandparent, parent_name)
                             if parent_match: actual_parent_p = parent_match
                             else: result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': No such file or directory (parent '{parent_p_str}' missing)", 'file_not_found'; return result

                         if actual_parent_p not in fs or fs[actual_parent_p].get('type') != 'directory':
                            result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': Parent '{actual_parent_p}' is not a directory or does not exist", 'file_not_found'; return result

                         fs[p] = {'type': 'directory', 'owner': 'user', 'perms':'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096}; result['success'], result['exit_code'] = True, 0
                 else: result['stderr'], result['reason'] = f"mkdir: Invalid path resolution for: {args[0]}", 'invalid_path'
            elif cmd == 'rm':
                 if not args: result['stderr'], result['reason'] = "rm: missing operand", 'missing_args'; return result
                 p = resolve_path_func(args[0]); target_p = p
                 if p and p not in fs:
                     parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name; found_match = find_match_func(parent, filename)
                     if found_match: target_p = found_match; logger.debug(f"rm: Targeting existing file/dir '{target_p}'")
                     else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found'; return result

                 if target_p and target_p in fs:
                     info = fs[target_p]
                     if info.get('type') == 'directory':
                         # Check if directory is empty
                         is_empty = not any(str(PurePosixPath(n).parent)==target_p for n in fs if n != target_p)
                         if is_empty: del fs[target_p]; result['success'], result['exit_code'] = True, 0
                         else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': Directory not empty", 'directory_not_empty'
                     else: # It's a file
                         del fs[target_p]; result['success'], result['exit_code'] = True, 0
                 elif target_p == '/': result['stderr'], result['reason'] = "rm: cannot remove root directory", 'permission_denied'
                 elif not target_p: result['stderr'], result['reason'] = f"rm: invalid path resolution for '{args[0]}'", 'invalid_path'
                 else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found' # Should have been caught by match
            elif cmd == 'echo':
                content_to_echo = ""; target_file_str = None; redirect_mode = None
                idx = 0
                # Simple parsing for redirection (doesn't handle complex cases like `echo "foo" > "bar baz"`)
                while idx < len(args):
                    if args[idx] == '>' or args[idx] == '>>':
                        redirect_mode = args[idx]
                        if idx + 1 < len(args):
                            target_file_str = args[idx+1]
                            content_to_echo = " ".join(args[:idx]) # Content is everything before redirect operator
                            idx = len(args) # Break loop
                        else:
                            result['stderr'],result['reason']="echo: missing target file for redirection",'missing_args'; return result
                        break
                    idx += 1
                if not redirect_mode: # No redirection found
                    content_to_echo = " ".join(args)

                # Handle quotes (simple removal if entire string is quoted)
                if content_to_echo.startswith('"') and content_to_echo.endswith('"'): content_to_echo = content_to_echo[1:-1]
                elif content_to_echo.startswith("'") and content_to_echo.endswith("'"): content_to_echo = content_to_echo[1:-1]

                if not redirect_mode:
                    result['stdout'], result['success'], result['exit_code'] = content_to_echo, True, 0
                else: # Handle redirection
                    if not target_file_str: result['stderr'],result['reason']="echo: missing target file for redirection",'missing_args'; return result
                    resolved_target_path = resolve_path_func(target_file_str)
                    if not resolved_target_path:
                        result['stderr'], result['reason'] = f"echo: Invalid path resolution for redirection target: {target_file_str}", 'invalid_path'; return result

                    final_write_path = resolved_target_path
                    # Check for existing case-insensitive match for the target file
                    parent_dir = str(PurePosixPath(resolved_target_path).parent)
                    filename = PurePosixPath(resolved_target_path).name
                    existing_match_path = find_match_func(parent_dir, filename)

                    if existing_match_path:
                        if existing_match_path in fs and fs[existing_match_path].get('type') == 'directory':
                            result['stderr'], result['reason'] = f"echo: {target_file_str}: Is a directory", 'is_directory'; return result
                        final_write_path = existing_match_path # Write to the existing cased path

                    current_file_content = ""
                    if redirect_mode == '>>': # Append mode
                        if final_write_path in fs and fs[final_write_path].get('type') == 'file':
                            current_file_content = fs[final_write_path].get('content', '')
                            if current_file_content and not current_file_content.endswith('\n') and content_to_echo: # Add newline if appending to non-empty file without trailing newline
                                current_file_content += '\n'

                    new_content = current_file_content + content_to_echo # New content is appended or overwrites
                    # Ensure newline at the end of echoed content if it's typical shell behavior
                    if not new_content.endswith('\n'): new_content += '\n'

                    fs[final_write_path] = {'type': 'file', 'content': new_content, 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(new_content.encode('utf-8'))}
                    result['success'], result['exit_code'] = True, 0
            elif cmd == 'cp':
                if len(args) != 2: result['stderr'], result['reason'] = "cp: missing file operand or too many arguments", 'missing_args'; return result
                src_arg, dest_arg = args[0], args[1]
                resolved_src = resolve_path_func(src_arg)
                if not resolved_src: result['stderr'], result['reason'] = f"cp: Invalid source path: {src_arg}", 'invalid_path'; return result
                actual_src_path = resolved_src
                if resolved_src not in fs:
                    match = find_match_func(str(PurePosixPath(resolved_src).parent), PurePosixPath(resolved_src).name)
                    if match: actual_src_path = match
                    else: result['stderr'], result['reason'] = f"cp: cannot stat '{src_arg}': No such file or directory", 'file_not_found'; return result
                if actual_src_path not in fs: result['stderr'],result['reason']=f"cp: source '{actual_src_path}' disappeared after check",'internal_error'; return result
                src_info = fs[actual_src_path]
                if src_info.get('type') == 'directory': result['stderr'], result['reason'] = f"cp: omitting directory '{src_arg}' (simulating no -r)", 'is_directory'; return result
                if src_info.get('type') != 'file': result['stderr'], result['reason'] = f"cp: '{src_arg}' is not a regular file", 'invalid_type'; return result

                resolved_dest_arg = resolve_path_func(dest_arg)
                if not resolved_dest_arg: result['stderr'], result['reason'] = f"cp: Invalid destination path: {dest_arg}", 'invalid_path'; return result

                final_dest_path = resolved_dest_arg
                # Check if destination is an existing directory
                existing_dest_match = find_match_func(str(PurePosixPath(resolved_dest_arg).parent), PurePosixPath(resolved_dest_arg).name)
                if existing_dest_match and existing_dest_match in fs and fs[existing_dest_match].get('type') == 'directory':
                    # Dest is a directory, copy file into it
                    src_basename = PurePosixPath(actual_src_path).name
                    path_in_dir = str(PurePosixPath(existing_dest_match) / src_basename)
                    # Check if file with same name (case-insensitive) exists in target dir
                    file_match_in_target_dir = find_match_func(existing_dest_match, src_basename)
                    if file_match_in_target_dir: final_dest_path = file_match_in_target_dir # Overwrite existing cased file
                    else: final_dest_path = path_in_dir # New file in target dir
                elif existing_dest_match and existing_dest_match in fs :
                    final_dest_path = existing_dest_match # Overwrite existing cased file (not a dir)

                if actual_src_path == final_dest_path: result['stderr'], result['reason'] = f"cp: '{src_arg}' and '{dest_arg}' are the same file", 'invalid_argument'; return result

                fs[final_dest_path] = copy.deepcopy(src_info); fs[final_dest_path]['owner'] = 'user'; fs[final_dest_path]['mtime'] = time.time()
                result['success'], result['exit_code'] = True, 0
            elif cmd == 'mv':
                if len(args) != 2: result['stderr'], result['reason'] = "mv: missing file operand or too many arguments", 'missing_args'; return result
                src_arg, dest_arg = args[0], args[1]
                resolved_src = resolve_path_func(src_arg)
                if not resolved_src: result['stderr'], result['reason'] = f"mv: Invalid source path: {src_arg}", 'invalid_path'; return result
                actual_src_path = resolved_src
                if resolved_src not in fs: # Source must exist (case-insensitive)
                    match = find_match_func(str(PurePosixPath(resolved_src).parent), PurePosixPath(resolved_src).name)
                    if match: actual_src_path = match
                    else: result['stderr'], result['reason'] = f"mv: cannot stat '{src_arg}': No such file or directory", 'file_not_found'; return result
                if actual_src_path not in fs: result['stderr'],result['reason']=f"mv: source '{actual_src_path}' disappeared after check",'internal_error'; return result # Should not happen
                src_info_copy = copy.deepcopy(fs[actual_src_path]) # Keep copy before removing

                resolved_dest_arg = resolve_path_func(dest_arg)
                if not resolved_dest_arg: result['stderr'], result['reason'] = f"mv: Invalid destination path: {dest_arg}", 'invalid_path'; return result

                final_dest_path = resolved_dest_arg
                # Check if dest is an existing directory
                existing_dest_match = find_match_func(str(PurePosixPath(resolved_dest_arg).parent), PurePosixPath(resolved_dest_arg).name)
                if existing_dest_match and existing_dest_match in fs and fs[existing_dest_match].get('type') == 'directory':
                    # Dest is a directory, move file into it
                    src_basename = PurePosixPath(actual_src_path).name
                    path_in_dir = str(PurePosixPath(existing_dest_match) / src_basename)
                    # Check if file with same name (case-insensitive) exists in target dir
                    file_match_in_target_dir = find_match_func(existing_dest_match, src_basename)
                    if file_match_in_target_dir: final_dest_path = file_match_in_target_dir # Overwrite existing cased file
                    else: final_dest_path = path_in_dir
                elif existing_dest_match and existing_dest_match in fs:
                     final_dest_path = existing_dest_match # Overwrite existing cased file (not a dir)
                # If no existing match, final_dest_path remains resolved_dest_arg (rename/new file)

                if actual_src_path == final_dest_path: result['stderr'], result['reason'] = f"mv: '{src_arg}' and '{dest_arg}' are the same file", 'invalid_argument'; return result
                if src_info_copy.get('type') == 'directory' and final_dest_path.startswith(actual_src_path + '/'): result['stderr'], result['reason'] = f"mv: cannot move '{src_arg}' to a subdirectory of itself, '{dest_arg}'", 'invalid_argument'; return result

                # Check for overwrite conditions if final_dest_path exists and is different from actual_src_path
                if final_dest_path in fs and final_dest_path != actual_src_path:
                    dest_info_at_final = fs[final_dest_path]
                    if src_info_copy.get('type') == 'directory':
                        if dest_info_at_final.get('type') == 'file': result['stderr'], result['reason'] = f"mv: cannot overwrite non-directory '{final_dest_path}' with directory '{src_arg}'", 'overwrite_file_with_dir'; return result
                        elif dest_info_at_final.get('type') == 'directory': # Moving dir into existing dir (handled by dest is dir logic above)
                            is_dest_empty = not any(str(PurePosixPath(k).parent) == final_dest_path for k in fs if k != final_dest_path and k != actual_src_path)
                            if not is_dest_empty: result['stderr'], result['reason'] = f"mv: failed to move '{src_arg}' to '{dest_arg}': Directory '{final_dest_path}' not empty", 'directory_not_empty'; return result
                    elif src_info_copy.get('type') == 'file':
                        if dest_info_at_final.get('type') == 'directory': result['stderr'], result['reason'] = f"mv: cannot overwrite directory '{final_dest_path}' with non-directory '{src_arg}'", 'overwrite_dir_with_file'; return result
                # If all checks pass, perform the move
                data_to_move = fs.pop(actual_src_path) # Remove source
                fs[final_dest_path] = data_to_move; fs[final_dest_path]['mtime'] = time.time() # Place at destination
                result['success'], result['exit_code'] = True, 0
            elif cmd == 'pytest':
                pytest_start_time = time.monotonic()
                keyword_filter = None
                verbose_mode = False
                target_paths_args_raw = [] # Files or directories specified in pytest command

                idx = 0
                while idx < len(args):
                    if args[idx] == '-k':
                        if idx + 1 < len(args): keyword_filter = args[idx+1]; idx += 1
                        else: result['stderr'], result['reason'] = "pytest: argument -k requires a value", 'missing_args'; return result
                    elif args[idx] == '-v' or args[idx] == '--verbose':
                        verbose_mode = True
                    elif not args[idx].startswith('-'):
                        target_paths_args_raw.append(args[idx])
                    idx += 1

                logger.debug(f"Simulating pytest: TargetsRaw={target_paths_args_raw}, Keyword='{keyword_filter}', Verbose={verbose_mode}")

                # _sim_discover_tests expects list of paths relative to CWD or absolute
                # If target_paths_args_raw is empty, discover_tests uses default test dir.
                # If paths are given, they are resolved by discover_tests.

                collected_sim_tests = self._sim_discover_tests(fs, target_paths_args_raw, None) # Collect all first
                ran_sim_tests_results = []

                if not collected_sim_tests and not target_paths_args_raw : # No tests found in default locations
                    stdout_output, stderr_output, exit_code_output = self._sim_generate_pytest_output([], [], verbose_mode, time.monotonic() - pytest_start_time)
                    result['stdout'] = stdout_output
                    result['stderr'] = stderr_output
                    result['exit_code'] = 5 # No tests found
                    result['success'] = True # Pytest ran, but found no tests
                    result['reason'] = 'no_tests_found'
                elif not collected_sim_tests and target_paths_args_raw: # Specific paths given, but no tests found there
                    stdout_output, stderr_output, exit_code_output = self._sim_generate_pytest_output([], [], verbose_mode, time.monotonic() - pytest_start_time)
                    result['stdout'] = stdout_output
                    result['stderr'] = stderr_output
                    result['exit_code'] = 5 # No tests found
                    result['success'] = True
                    result['reason'] = 'no_tests_found_in_paths'
                else:
                    # Filter collected tests by keyword if provided
                    tests_to_run = []
                    if keyword_filter:
                        for test_file_path, test_name in collected_sim_tests:
                            rel_file_path = str(PurePosixPath(test_file_path).relative_to(PurePosixPath(self._simulated_system_cwd)))
                            node_id = f"{rel_file_path}::{test_name}"
                            if keyword_filter in test_name or keyword_filter in rel_file_path or keyword_filter in node_id:
                                tests_to_run.append((test_file_path, test_name))
                    else:
                        tests_to_run = collected_sim_tests

                    if not tests_to_run and keyword_filter : # Collected tests, but filter deselected all
                        stdout_output, stderr_output, exit_code_output = self._sim_generate_pytest_output(collected_sim_tests, [], verbose_mode, time.monotonic() - pytest_start_time)
                        result['stdout'] = stdout_output
                        result['stderr'] = stderr_output
                        result['exit_code'] = 5 # No tests selected
                        result['success'] = True
                        result['reason'] = 'no_tests_selected_by_filter'
                    else :
                        for test_file_path, test_name in tests_to_run:
                            status, full_name = self._sim_run_one_test(test_file_path, test_name, keyword_filter) # keyword_filter passed for context if needed by run_one_test
                            ran_sim_tests_results.append((status, full_name))

                        pytest_duration = time.monotonic() - pytest_start_time
                        stdout_res, stderr_res, exit_code_res = self._sim_generate_pytest_output(collected_sim_tests, ran_sim_tests_results, verbose_mode, pytest_duration)

                        result['stdout'] = stdout_res
                        result['stderr'] = stderr_res
                        result['exit_code'] = exit_code_res
                        result['success'] = (exit_code_res == 0 or exit_code_res == 5) # Exit code 5 (no tests run) is not a command failure
                        result['reason'] = 'tests_ran' if exit_code_res in [0,1] else 'no_tests_collected_or_selected' if exit_code_res == 5 else 'execution_error'

            else:
                result['stderr'], result['reason'] = f"Sim command '{cmd}' logic not implemented.", 'not_implemented'
        except Exception as e:
            logger.error(f"Sim Internal Error exec '{command_str}': {e}",exc_info=True)
            result.update({'stderr':f"Internal sim error: {e}",'reason':'internal_error'})

        if self._simulated_state: # Ensure state exists before updating
            self._simulated_state['timestamp']=time.time()
            self._simulated_state['last_command_result']=copy.deepcopy(result)
        logger.debug(f"VM Sim Res: Succ={result['success']}, Code={result['exit_code']}, Out='{result['stdout'][:30]}...', Err='{result['stderr'][:30]}...'")
        return result

    def _execute_real_command(self, command_str: str, execution_context: Callable[[str], Dict]) -> Dict[str, Any]:
        result: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'command': command_str, 'reason': '', 'exit_code': -1}
        try:
            parts = shlex.split(command_str)
        except ValueError as e:
            result['stderr'], result['reason'] = f"Command parse error: {e}", 'parse_error'
            return result

        command = parts[0] if parts else ''

        # --- MODIFIED SECTION TO HANDLE ALLOW_ALL_COMMANDS_MAGIC_VALUE ---
        # Check if the configured global variable (VM_SERVICE_ALLOWED_REAL_COMMANDS from config)
        # is set to the magic value.
        if VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
            logger.warning(f"VM COMMAND WHITELIST BYPASSED (Magic Value Active) for command: '{command_str}'")
        elif command not in self._effective_allowed_commands_list: # Check against the list derived in __init__
            result['stderr'], result['reason'] = f"Command '{command}' not allowed for real execution.", 'safety_violation'
            logger.warning(f"Safety Violation: Blocked real system execution: '{command_str}'")
            return result
        # --- END MODIFIED SECTION ---

        try:
            # Handle 'cd' separately as it modifies internal state (CWD) rather than just executing
            if command == 'cd':
                if len(parts) == 1: target_dir_str = '~' # cd with no args goes to home
                elif len(parts) == 2: target_dir_str = parts[1]
                else:
                    result['stderr'], result['reason'] = "cd: too many arguments", 'invalid_argument'
                    return result

                if target_dir_str == '~': target_dir = '/app' # Define your "home"
                else: target_dir = self._resolve_path(target_dir_str)

                if not target_dir:
                    result['stderr'], result['reason'] = f"cd: Invalid path resolution for: {target_dir_str}", 'invalid_path'
                    return result

                # To verify if cd is possible, we can try to list the target directory
                # This is a common way to check existence and permissions in a shell-like manner
                check_cmd = f"ls -ld {shlex.quote(target_dir)}"
                check_res = execution_context(check_cmd) # Use the provided execution_context

                if check_res.get('success') and check_res.get('stdout','').strip().startswith('d'):
                    self._set_current_cwd(target_dir)
                    result['success'], result['exit_code'] = True, 0
                    logger.info(f"Real system CWD updated to: {self._get_current_cwd()}")
                elif check_res.get('exit_code') != 0:
                    stderr_output = check_res.get('stderr', '').lower()
                    reason = 'execution_error' # Default reason
                    if 'no such file or directory' in stderr_output:
                        reason = 'file_not_found'
                    elif 'permission denied' in stderr_output:
                        reason = 'permission_denied'
                    result.update({
                        'success':False,
                        'exit_code': check_res.get('exit_code', 1),
                        'stderr':f"cd: cannot access '{target_dir_str}': {check_res.get('stderr','Unknown error')}",
                        'reason': reason
                    })
                else: # ls -ld succeeded but it's not a directory
                    result.update({
                        'success':False,
                        'exit_code':1, # Or another appropriate non-zero code
                        'stderr':f"cd: not a directory: {target_dir_str}",
                        'reason':'is_not_directory'
                    })
                return result
            else:
                # Normal command execution for non-'cd' commands
                return execution_context(command_str)
        except Exception as e:
            logger.error(f"Unexpected error during real command prep/exec '{command_str}': {e}", exc_info=True)
            result['stderr'], result['reason'] = f"Unexpected internal error: {e}", 'internal_error'
            return result

    def _docker_exec_context(self, command_str: str) -> Dict[str, Any]:
        res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
        if not self.docker_container:
            res['stderr']='Docker container unavailable'
            res['reason']='docker_error'
            return res
        try:
            # Ensure command_str is properly quoted for `sh -c`
            # shlex.quote is good for this, but `sh -c` itself needs careful handling
            # The command given to sh -c should be a single string.
            # Example: sh -c "ls -l /path/with spaces"
            full_cmd = f"sh -c {shlex.quote(command_str)}" # This correctly quotes the *entire* command_str for sh -c
            logger.debug(f"Docker Exec Run: cmd='{full_cmd}', workdir='{self._get_current_cwd()}'")

            exit_code, output_stream = self.docker_container.exec_run(
                cmd=full_cmd,
                workdir=self._get_current_cwd(),
                stream=False, # Get all output at once
                demux=False,  # Get combined stdout/stderr as a single byte string
                user='root'   # Or specific user if needed
            )
            output_bytes: bytes = output_stream if isinstance(output_stream, bytes) else b''
            output_str = output_bytes.decode('utf-8', errors='replace').strip()

            res['exit_code'] = exit_code
            res['success'] = (exit_code == 0)

            if res['success']:
                res['stdout'] = output_str
                res['stderr'] = '' # No stderr if successful (or it was part of output_str if demux=False and error occurred)
            else:
                res['stdout'] = '' # No stdout if failed (or it was part of output_str)
                res['stderr'] = output_str
                res['reason'] = 'execution_error' # Generic reason
                # Try to be more specific if possible, though Docker exec_run doesn't give fine-grained reasons
                if "command not found" in output_str.lower():
                    res['reason'] = 'command_not_found'
                elif "permission denied" in output_str.lower():
                    res['reason'] = 'permission_denied'
                elif "no such file or directory" in output_str.lower():
                    res['reason'] = 'file_not_found'

            if not res['success'] and not res['stderr']: # Safety net if error but no output
                res['stderr'] = f"Command failed (Code {res['exit_code']}) with no output."

            return res
        except DockerAPIError as api_err:
            logger.error(f"Docker API error executing '{command_str}': {api_err}", exc_info=True)
            res.update({'stderr':f"Docker API error: {api_err}", 'reason':'docker_api_error'})
            return res
        except Exception as e: # Catch other unexpected errors
            logger.error(f"Docker exec_run unexpected error '{command_str}': {e}", exc_info=True)
            res.update({'stderr':f"Docker exec_run error: {e}", 'reason':'docker_error'})
            return res

    def _subprocess_exec_context(self, command_str: str) -> Dict[str, Any]:
         res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
         current_vm_cwd = self._get_current_cwd() # This is the Seed's internal CWD, might be /app

         effective_host_cwd = None # This will be the CWD actually passed to subprocess.run

         if os.name == 'nt': # Targeting Windows specifically for this fix
            # Determine a valid CWD for Windows
            # Priority:
            # 1. If current_vm_cwd is already a valid Windows directory, use it.
            # 2. If current_vm_cwd is '/app' (common default) or another POSIX-like root path,
            #    and it's NOT a valid directory on Windows, try project root.
            # 3. If project root is also not valid (unlikely but possible), fallback to C:\ or temp.

            if Path(current_vm_cwd).is_dir(): # Check if the VM's CWD is directly usable on host
                effective_host_cwd = current_vm_cwd
                logger.debug(f"_subprocess_exec_context: Using VM CWD '{current_vm_cwd}' as it's a valid host directory.")
            elif current_vm_cwd.startswith('/') : # Common indicator of POSIX-style path from sim/docker
                # Try to use the project's root directory as a more sensible default on Windows host
                try:
                    # Assuming vm_service.py is in RSIAI0/seed/
                    project_root_path = Path(__file__).resolve().parent.parent
                    if project_root_path.is_dir():
                        effective_host_cwd = str(project_root_path)
                        logger.warning(f"_subprocess_exec_context: VM CWD '{current_vm_cwd}' seems POSIX-like or invalid on Windows. Falling back to project root: {effective_host_cwd}")
                    else: # Project root itself isn't a dir (edge case)
                        effective_host_cwd = 'C:\\' # Ultimate fallback
                        logger.warning(f"_subprocess_exec_context: VM CWD '{current_vm_cwd}' and project root invalid. Falling back to CWD: {effective_host_cwd}")
                except Exception as e_path:
                    effective_host_cwd = 'C:\\' # Ultimate fallback on path error
                    logger.error(f"_subprocess_exec_context: Error determining project root for fallback CWD: {e_path}. Using 'C:\\\\'.")
            else: # current_vm_cwd is not POSIX-like, but also not a dir (e.g. "D:\nonexistent")
                effective_host_cwd = 'C:\\'
                logger.warning(f"_subprocess_exec_context: VM CWD '{current_vm_cwd}' is not a valid directory. Falling back to CWD: {effective_host_cwd}")
         else: # Non-Windows (POSIX-like systems)
            effective_host_cwd = current_vm_cwd # Assume current_vm_cwd is valid

         logger.debug(f"Subprocess Exec Run: cmd='{command_str}', effective_host_cwd='{effective_host_cwd}' (Original VM CWD: '{current_vm_cwd}')")

         try:
            # Using shell=True. Be mindful of command injection if command_str is not controlled.
            # Given the "allow all" context, this risk is inherent.
            proc = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True, # Decodes output as text
                encoding='utf-8', # Explicit encoding
                errors='replace', # Handles decoding errors gracefully
                timeout=self.command_timeout_sec,
                check=False, # We handle the return code manually
                cwd=effective_host_cwd # Use the determined effective CWD
            )
            res['exit_code'] = proc.returncode
            res['stdout'] = proc.stdout.strip() if proc.stdout else ''
            res['stderr'] = proc.stderr.strip() if proc.stderr else ''
            res['success'] = (proc.returncode == 0)

            if not res['success']:
                res['reason'] = 'execution_error' # Generic reason
                stderr_lower = res['stderr'].lower()
                if "the directory name is invalid" in stderr_lower: # Catch the specific error
                    res['reason'] = 'win_invalid_directory'
                elif "command not found" in stderr_lower or "not recognized" in stderr_lower:
                    res['reason'] = 'command_not_found'
                elif "permission denied" in stderr_lower:
                    res['reason'] = 'permission_denied'
                elif "no such file or directory" in stderr_lower:
                    res['reason'] = 'file_not_found'
                # Add more specific error reason parsing if needed

            if not res['success'] and not res['stderr']: # Safety net if error but no stderr
                res['stderr'] = f"Command failed (Code {proc.returncode}) with no stderr. CWD was '{effective_host_cwd}'."
            elif res['success'] and res['stderr']: # Log stderr even on success, as some tools use it for warnings
                logger.info(f"Command '{command_str}' succeeded but produced stderr (CWD: {effective_host_cwd}): {res['stderr']}")


            return res
         except FileNotFoundError:
            # This typically means the shell itself (if shell=True) or the command (if shell=False and command isn't a path) wasn't found.
            # With shell=True, it's less likely for the shell, more likely for command within the shell string if not in PATH.
            res['stderr'] = f"Command or essential component not found: {command_str.split()[0]} (Attempted CWD: {effective_host_cwd})"
            res['reason'] = 'command_not_found'
            res['exit_code'] = 127 # Common exit code for command not found
            return res
         except subprocess.TimeoutExpired:
            res['stderr'] = f"Command execution timed out after {self.command_timeout_sec} seconds (Attempted CWD: {effective_host_cwd})"
            res['reason'] = 'timeout'
            res['exit_code'] = -9 # Arbitrary code for timeout
            return res
         except PermissionError as pe: # Catch OS-level permission errors if subprocess.run itself fails to launch
            logger.error(f"OS PermissionError launching subprocess for '{command_str}' in CWD '{effective_host_cwd}': {pe}", exc_info=True)
            res['stderr'] = f"OS PermissionError: {pe} (Attempted CWD: {effective_host_cwd})"
            res['reason'] = 'permission_denied_os_level'
            return res
         except Exception as e:
            logger.error(f"Unexpected subprocess error executing '{command_str}' in CWD '{effective_host_cwd}': {e}", exc_info=True)
            res['stderr'] = f"Unexpected subprocess execution error: {e} (Attempted CWD: {effective_host_cwd})"
            res['reason'] = 'internal_error' # Or a more specific 'subprocess_exception'
            return res

    def read_file(self, path: str) -> Dict[str, Any]:
        logger.info(f"Seed VMService: Reading file '{path}'")
        result = {'success': False, 'content': None, 'message': '', 'details': {'path': path}, 'reason': ''}
        original_abs_path = self._resolve_path(path)

        if not original_abs_path:
            result['message'] = "Invalid path resolution."
            result['reason'] = 'invalid_path'
            return result
        result['details']['absolute_path'] = original_abs_path
        abs_path_to_try = original_abs_path # Start with the resolved path

        def _perform_read_attempt(read_path: str) -> Dict[str, Any]:
            read_result = {'success': False, 'content': None, 'message': '', 'reason': '', 'details': {}}
            if self.use_real_system:
                if 'cat' not in self._effective_allowed_commands_list and VM_SERVICE_ALLOWED_REAL_COMMANDS != VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
                    read_result['message'] = "Cannot read file: 'cat' command not allowed."
                    read_result['reason'] = 'safety_violation'
                    return read_result

                exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
                cat_cmd = f"cat {shlex.quote(read_path)}" # Ensure path is quoted
                cat_res = self._execute_real_command(cat_cmd, exec_func) # Use internal _execute_real_command

                read_result['reason'] = cat_res.get('reason', 'execution_error')
                read_result['details'] = {'exit_code': cat_res.get('exit_code', -1), 'stderr': cat_res.get('stderr')}
                if cat_res.get('success'):
                    read_result['success'] = True
                    read_result['content'] = cat_res.get('stdout', '') # stdout from cat is the file content
                    read_result['message'] = "File read successfully."
                else:
                    read_result['message'] = f"Failed to read file: {cat_res.get('stderr', 'Unknown error')}"
                    stderr_lower = (cat_res.get('stderr') or '').lower()
                    if 'no such file or directory' in stderr_lower: read_result['reason'] = 'file_not_found'
                    elif 'is a directory' in stderr_lower: read_result['reason'] = 'is_directory'
                    elif 'permission denied' in stderr_lower: read_result['reason'] = 'permission_denied'
            else: # Simulation mode
                if not self._simulated_state:
                    read_result['message'] = "Simulation not initialized."
                    read_result['reason'] = 'internal_error'
                    return read_result
                fs = self._simulated_state['filesystem']
                if read_path in fs:
                    item_info = fs[read_path]
                    if item_info.get('type') == 'file':
                        read_result['success'] = True
                        read_result['content'] = item_info.get('content', '')
                        read_result['message'] = "File read successfully (simulation)."
                    elif item_info.get('type') == 'directory':
                        read_result['message'] = "Cannot read: Is a directory."
                        read_result['reason'] = 'is_directory'
                    else:
                        read_result['message'] = f"Cannot read: Not a file (Type: {item_info.get('type')})."
                        read_result['reason'] = 'invalid_type'
                else: # File not found at this exact path
                    read_result['message'] = "File not found (simulation)."
                    read_result['reason'] = 'file_not_found'
            return read_result

        # First attempt with the resolved path
        attempt1_result = _perform_read_attempt(abs_path_to_try)

        # If first attempt failed with 'file_not_found', try case-insensitive lookup
        if not attempt1_result['success'] and attempt1_result['reason'] == 'file_not_found':
            logger.info(f"Read failed for exact path '{abs_path_to_try}', attempting case-insensitive lookup.")
            try:
                parent_path = str(PurePosixPath(abs_path_to_try).parent)
                filename = PurePosixPath(abs_path_to_try).name
                corrected_path = self._find_case_insensitive_match(parent_path, filename)
                if corrected_path:
                    logger.info(f"Found corrected path '{corrected_path}', retrying read.")
                    result['details']['corrected_path_used'] = corrected_path
                    attempt2_result = _perform_read_attempt(corrected_path)
                    result.update(attempt2_result) # Update main result with the second attempt
                    # Append to message if it exists from attempt2
                    if result.get('message'):
                         result['message'] += f" (used corrected path '{PurePosixPath(corrected_path).name}')"
                    else: # Should not happen if attempt2_result is valid
                         result['message'] = f"Read with corrected path '{PurePosixPath(corrected_path).name}'."
                else: # No corrected path found
                    result.update(attempt1_result) # Use the result of the first attempt
            except Exception as lookup_e:
                logger.error(f"Error during case-insensitive lookup for read: {lookup_e}", exc_info=True)
                result.update(attempt1_result) # Fallback to first attempt result
                result['message'] = str(result.get('message','')) + " (Case-insensitive lookup failed with error)"
                if 'reason' not in result or not result['reason']: result['reason'] = 'internal_error' # Ensure reason is set
        else: # First attempt was successful or failed for a reason other than 'file_not_found'
            result.update(attempt1_result)

        # Ensure details always contains original path and attempted absolute path
        result['details']['path'] = path
        result['details']['absolute_path'] = original_abs_path
        return result

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        logger.info(f"Seed VMService: Writing to file '{path}' (Content length: {len(content)})")
        result = {'success': False, 'message': '', 'details': {'path': path, 'content_length': len(content)}, 'reason': ''}
        resolved_path = self._resolve_path(path)

        if not resolved_path:
            result['message'] = "Invalid path resolution."
            result['reason'] = 'invalid_path'
            return result
        result['details']['absolute_path'] = resolved_path
        write_target_path = resolved_path # Default to the resolved path

        try:
            # Check for existing case-insensitive match to decide if we are overwriting an existing cased file
            # or creating a new one with the specified casing.
            parent_path_str = str(PurePosixPath(resolved_path).parent)
            filename_str = PurePosixPath(resolved_path).name
            existing_match_path = self._find_case_insensitive_match(parent_path_str, filename_str)

            if existing_match_path:
                # If a match exists, we should write to that exact path to overwrite it.
                logger.info(f"Found existing case-insensitive match '{existing_match_path}'. Write will target this path.")
                write_target_path = existing_match_path
                result['details']['corrected_path_used'] = existing_match_path
            elif len(self._get_potential_matches(parent_path_str, filename_str)) > 1 :
                 # This case should be rare if _find_case_insensitive_match returns None for multiple matches
                 result['message'] = f"Write failed: Ambiguous path, multiple case-insensitive matches exist for '{filename_str}' in '{parent_path_str}' and no exact match provided."
                 result['reason'] = 'ambiguous_path'
                 return result
            # If no existing match, write_target_path remains resolved_path (create new file with this casing).

        except Exception as lookup_e:
            logger.error(f"Error during case-insensitive lookup for write: {lookup_e}", exc_info=True)
            result['message'] = f"Write failed: Error during path lookup: {lookup_e}"
            result['reason'] = 'internal_error'
            return result

        result['details']['final_write_path'] = write_target_path

        if self.use_real_system:
            if not (('sh' in self._effective_allowed_commands_list and 'printf' in self._effective_allowed_commands_list) or VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE) :
                result['message'] = "Cannot write file: 'sh' or 'printf' command not allowed."
                result['reason'] = 'safety_violation'
                return result

            # Using printf for writing content. Content is passed as an argument.
            # Need to be extremely careful with quoting content if it contains shell metacharacters.
            # shlex.quote should handle the content safely for printf's format string if it's simple.
            # For complex content, a here-string or piping echo might be safer, but printf is often available.
            # A common pattern: printf "%s" "$CONTENT_VAR" > file
            # Here, we pass content directly. If content is very large, this might hit argument length limits.
            # A more robust method for large content would be to pipe `echo` or use a temporary file with `cat`.
            # For now, using printf as it's often a core utility.
            # The content itself should be quoted by shlex.quote if it's part of the command string being built for `sh -c`.
            # So the structure for `sh -c` would be: sh -c 'printf "%s" "my content with spaces and \"quotes\"" > "/path/to/file"'
            # The command for `_execute_real_command` will be: `printf %s ${quoted_content} > ${quoted_path}`
            # where `_execute_real_command` will then wrap this in `sh -c`.
            # Correct construction:
            # Command to be executed by `sh -c`: printf "%s" "..." > "..."
            # The content itself needs to be escaped for the `printf "%s" content` part.
            # Simplest: use `cat <<EOF > file \n content \n EOF` pattern or pipe echo.
            # Let's use a more robust echo pipe for now.
            # echo -n is to avoid trailing newline from echo itself, content should have its own newlines.
            # However, printf is better for precise content.
            # Using `sh -c 'printf "%s" "$1" > "$2"' sh_cmd_name content_var path_var` is safer with exec_run if it supported args like that.
            # Given current exec_run which takes a single command string for `sh -c`:
            # We need to ensure content is properly escaped for the shell string.
            # One way: base64 encode content, then decode in shell.
            # `echo 'BASE64_CONTENT' | base64 -d > 'FILE'`
            # This requires base64 to be an allowed command.
            # Let's try a simpler printf, assuming content isn't excessively hostile.
            # The `shlex.quote` on the *entire* `printf ... > ...` command for `sh -c` is key.
            # No, `sh -c` takes ONE string. The command *within* that string needs quoting.

            # Safest for arbitrary content with printf: use hex escapes for tricky bytes.
            # Or, ensure `sh -c` receives the content as a separate argument if the execution context supported it.
            # Since it doesn't, we must embed. A common trick is:
            # CMD="cat > '$FILE'" then pipe content to this command's stdin.
            # Our current `execution_context` doesn't support stdin piping easily.

            # Fallback to a simple redirect, hoping content is not too complex.
            # This is a known limitation if content has many shell metacharacters.
            # A heredoc approach is more robust for `sh -c`.
            # Example: sh -c "cat > '$(echo "$FILE_PATH_ESCAPED")' <<'EOF_MARKER'\n$CONTENT_ESCAPED_FOR_HEREDOC\nEOF_MARKER\n"
            # This is getting very complex to build reliably.

            # Simpler: use `echo -n ... >> file` and `rm file` first for overwrite.
            # Or `tee` if available.
            # Let's try to stick with `printf` and rely on `shlex.quote` for the path.
            # For content, it's tricky. If content contains single quotes, shlex.quote will wrap with double quotes.
            # If content contains `$(...)` or backticks, they could be evaluated by the outer `sh -c`.

            # Most reliable for `sh -c "command content path"` where content is complex:
            # 1. Create a command string that reads from stdin: `cat > quoted_path`
            # 2. This isn't directly supported by current `_docker_exec_context` or `_subprocess_exec_context`
            #    as they don't take stdin input separately.

            # Let's try `bash -c 'echo "$1" > "$2"' -- "$CONTENT" "$FILE"` if the context supports it. It doesn't.

            # Back to basics: printf and hope for the best with quoting.
            # The risk is content like `$(reboot)` if not perfectly quoted.
            # `shlex.quote(content)` is for a single shell token.
            # `printf "%s" content > path` -> content must be safe for printf's first arg if not quoted.
            # `printf %s content > path` -> content is multiple shell tokens.
            # `printf '%s' "$CONTENT_VAR" > "$PATH_VAR"` is safer if we could set vars.

            # Simplest that might work for reasonable content:
            # Ensure `content` does not break out of the `sh -c '...'` single quotes.
            # This requires escaping single quotes within content.
            # content_for_sh_c = content.replace("'", "'\\''") # Escape single quotes for `sh -c '...'`
            # write_cmd_for_sh_c = f"printf %s '{content_for_sh_c}' > {shlex.quote(write_target_path)}"
            # This is for `printf %s 'content'` (content as a single arg to printf)
            # If content has newlines, printf %s will print them.

            # Using base64 encoding for content is safer if `base64` is available.
            if ('base64' in self._effective_allowed_commands_list or VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE) and \
               ('sh' in self._effective_allowed_commands_list or VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE):
                import base64 as b64lib
                encoded_content = b64lib.b64encode(content.encode('utf-8')).decode('ascii')
                # Command: sh -c "echo 'ENCODED' | base64 -d > 'FILE_PATH'"
                write_cmd = f"echo {shlex.quote(encoded_content)} | base64 -d > {shlex.quote(write_target_path)}"
                logger.debug(f"Using base64 for write_file. Command: {write_cmd[:100]}...")
            else: # Fallback to less safe printf if base64 not allowed
                logger.warning("Using direct printf for write_file; complex content might fail or be insecure. Allow 'base64' for more robust writes.")
                # Escape for printf's format string and for shell.
                # printf format specifiers: %
                content_escaped_printf = content.replace('%', '%%')
                # Now escape for shell single quotes within sh -c '...'
                content_escaped_shell = content_escaped_printf.replace("'", "'\\''")
                write_cmd = f"printf -- '{content_escaped_shell}' > {shlex.quote(write_target_path)}"
                # Using -- to mark end of options for printf, making it safer if content starts with -.


            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            write_res = self._execute_real_command(write_cmd, exec_func) # Use internal _execute_real_command

            result['details']['exit_code'] = write_res.get('exit_code', -1)
            result['details']['stderr'] = write_res.get('stderr')
            result['reason'] = write_res.get('reason', 'execution_error')

            if write_res.get('success'):
                result['success'] = True
                result['message'] = "File written successfully."
                if write_target_path != resolved_path: # If we used a corrected_path
                    result['message'] += f" (Used existing path '{PurePosixPath(write_target_path).name}')"
            else:
                result['message'] = f"Failed to write file: {write_res.get('stderr', 'Unknown error')}"
                stderr_lower = (write_res.get('stderr') or '').lower()
                # Reasons are often set by _execute_real_command based on context's interpretation
                if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found' # Likely parent path issue
                elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
                elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state:
                result['message'] = "Simulation not initialized."
                result['reason'] = 'internal_error'
                return result
            fs = self._simulated_state['filesystem']
            item_info_at_target = fs.get(write_target_path)

            if item_info_at_target and item_info_at_target.get('type') == 'directory':
                result['message'] = "Cannot write: Is a directory."
                result['reason'] = 'is_directory'
            else:
                # Check parent directory exists for writing a new file
                parent_of_target = str(PurePosixPath(write_target_path).parent)
                actual_parent_of_target = parent_of_target
                if parent_of_target not in fs: # Try case-insensitive find for parent
                    grandparent = str(PurePosixPath(parent_of_target).parent)
                    parent_name = PurePosixPath(parent_of_target).name
                    matched_parent = self._find_case_insensitive_match(grandparent, parent_name)
                    if matched_parent and matched_parent in fs and fs[matched_parent].get('type') == 'directory':
                        actual_parent_of_target = matched_parent
                    else:
                        result['message'] = f"Cannot write: Parent directory '{parent_of_target}' does not exist."
                        result['reason'] = 'file_not_found' # For parent
                        return result
                elif fs[actual_parent_of_target].get('type') != 'directory':
                    result['message'] = f"Cannot write: Parent '{actual_parent_of_target}' is not a directory."
                    result['reason'] = 'invalid_path' # Parent not a dir
                    return result


                fs[write_target_path] = {'type': 'file', 'content': content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(content.encode('utf-8'))}
                result['success'] = True
                result['message'] = "File written successfully (simulation)."
                if write_target_path != resolved_path: # If we used a corrected_path
                    result['message'] += f" (Used existing path '{PurePosixPath(write_target_path).name}')"
        return result

    def get_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        if self.use_real_system:
             return self._get_real_system_state(target_path_hint)
        else: # Simulation
            if self._simulated_state:
                state_copy = copy.deepcopy(self._simulated_state)
                state_copy['target_path_hint'] = target_path_hint
                state_copy['mode'] = 'simulation'
                # CWD is already updated in _simulated_state by _set_current_cwd
                # state_copy['cwd'] = self._get_current_cwd() # Ensure it's the latest

                # If a target_path_hint is provided, ensure its status is reflected in the copied filesystem snapshot
                if target_path_hint:
                    sim_fs_snapshot = state_copy['filesystem'] # Work on the copy
                    abs_hint_path = self._resolve_path(target_path_hint)
                    if abs_hint_path:
                        # If path (case-sensitively) not in snapshot, try case-insensitive
                        actual_path_for_hint_status = abs_hint_path
                        if abs_hint_path not in sim_fs_snapshot:
                            parent = str(PurePosixPath(abs_hint_path).parent)
                            filename = PurePosixPath(abs_hint_path).name
                            match = self._find_case_insensitive_match(parent, filename)
                            if match:
                                actual_path_for_hint_status = match

                        if actual_path_for_hint_status not in sim_fs_snapshot:
                             # Add a placeholder if even case-insensitive match not found
                             sim_fs_snapshot[abs_hint_path] = {'type': None, 'exists': False, 'error': 'No such file or directory (Simulated)'}
                        else:
                             # Mark as existing (already in snapshot, just ensure 'exists' is True)
                             sim_fs_snapshot[actual_path_for_hint_status]['exists'] = True
                    else: # Path resolution failed
                         # Add placeholder for the original hint if resolution failed
                         sim_fs_snapshot[target_path_hint] = {'type': None, 'exists': False, 'error': 'Invalid path resolution (Simulated)'}
                return state_copy
            else:
                logger.error("Cannot get state: Simulation state not initialized.")
                return {"error": "Simulation state not initialized."}

    def _get_real_system_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        current_cwd = self._get_current_cwd()
        mode_str = 'docker' if self.docker_container else 'subprocess'
        logger.debug(f"Probing real system state (Mode: {mode_str}, CWD: {current_cwd})...")
        state: Dict[str, Any] = {
            'timestamp': time.time(),
            'filesystem': {}, # Store info about specific paths (CWD, target_path_hint)
            'resources': {},
            'mode': mode_str,
            'cwd': current_cwd,
            'target_path_hint': target_path_hint,
            'probe_errors': []
        }
        probe_results: Dict[str, Dict] = {} # Store results of actual probe commands
        exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context

        # Define probes
        # Using standard shell commands. Ensure they are in allowed_real_commands or magic value is set.
        probes_to_run: Dict[str, str] = {
            'cpu': "top -bn1 | grep '^%Cpu' | head -n1", # For Linux-like systems
            'mem': "grep -E 'MemTotal|MemAvailable|MemFree' /proc/meminfo", # For Linux
            'disk_cwd': f"df -k {shlex.quote(current_cwd)}", # Disk usage for CWD's filesystem
            'ls_cwd': f"ls -lA --full-time {shlex.quote(current_cwd)}" # Listing of CWD
        }

        abs_target_hint_path: Optional[str] = None
        if target_path_hint:
            abs_target_hint_path = self._resolve_path(target_path_hint)
            if abs_target_hint_path:
                probes_to_run['stat_target'] = f'stat {shlex.quote(abs_target_hint_path)}'
                state['filesystem'][abs_target_hint_path] = {'type':None,'exists':False,'error':'Probe pending'} # Initialize target hint entry
            else:
                state['probe_errors'].append(f"Invalid target path hint resolution: {target_path_hint}")
                if target_path_hint not in state['filesystem']: # Ensure entry exists if hint was provided
                    state['filesystem'][target_path_hint] = {'type':None,'exists':False,'error':'Invalid path resolution (Real Mode)'}


        # Filter probes based on allowed commands (unless allow_all is active)
        allowed_probes_to_run: Dict[str, str] = {}
        if VM_SERVICE_ALLOWED_REAL_COMMANDS == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
            allowed_probes_to_run = probes_to_run
        else:
            for key, cmd_str in probes_to_run.items():
                cmd_base = cmd_str.split()[0] # Get the base command (e.g., 'top', 'grep')
                if cmd_base in self._effective_allowed_commands_list:
                    allowed_probes_to_run[key] = cmd_str
                else:
                    state['probe_errors'].append(f"Probe '{key}' (command '{cmd_base}') not allowed.")
                    if key == 'stat_target' and abs_target_hint_path: # Handle missing stat for target
                         state['filesystem'][abs_target_hint_path]['error'] = 'Stat command disabled or not allowed.'


        # Execute allowed probes
        for key, cmd in allowed_probes_to_run.items():
            res = self._execute_real_command(cmd, exec_func) # Use internal exec to respect CWD
            probe_results[key] = res
            if not res.get('success'):
                err_msg = f"Probe '{key}' failed (Code {res.get('exit_code','?')}). Stderr: {res.get('stderr', 'N/A')}"
                state['probe_errors'].append(err_msg)
                logger.warning(f"State probe '{key}' command failed: {cmd} -> {res.get('stderr', 'No stderr')}")

        # Parse probe results
        try:
            # CPU
            cpu_res = probe_results.get('cpu')
            if cpu_res and cpu_res.get('success'):
                idle_match = re.search(r"ni(?:,)?\s*([\d\.]+)\s*id", cpu_res['stdout']) # More flexible regex for %Cpu line
                if idle_match:
                    try: idle_perc = float(idle_match.group(1)); state['resources']['cpu_load_percent'] = round(100.0 - idle_perc, 1)
                    except ValueError: state['probe_errors'].append("Failed to parse CPU idle percentage.")
                else: state['probe_errors'].append("Could not find CPU idle percentage in 'top' output.")

            # Memory
            mem_res = probe_results.get('mem')
            if mem_res and mem_res.get('success'):
                total_kb_match = re.search(r"MemTotal:\s+(\d+)\s*kB", mem_res['stdout'])
                avail_kb_match = re.search(r"MemAvailable:\s+(\d+)\s*kB", mem_res['stdout'])
                free_kb_match = re.search(r"MemFree:\s+(\d+)\s*kB", mem_res['stdout']) # Fallback if MemAvailable not present

                if total_kb_match:
                    try:
                        total = int(total_kb_match.group(1))
                        used_mem = 0
                        if avail_kb_match:
                            avail = int(avail_kb_match.group(1))
                            used_mem = total - avail
                        elif free_kb_match: # Fallback to MemFree
                            free = int(free_kb_match.group(1))
                            # This is a rougher estimate, MemAvailable is preferred
                            used_mem = total - free # Buffers/cache are counted as "used" here

                        if total > 0: state['resources']['memory_usage_percent'] = round((used_mem / total) * 100.0, 1)
                        else: state['resources']['memory_usage_percent'] = 0.0
                    except (ValueError, ZeroDivisionError): state['probe_errors'].append("Failed to parse memory usage.")
                else: state['probe_errors'].append("Could not find MemTotal in /proc/meminfo output.")

            # Disk CWD
            disk_res = probe_results.get('disk_cwd')
            if disk_res and disk_res.get('success'):
                lines = disk_res['stdout'].strip().split('\n')
                if len(lines) > 1: # Expect header + data line
                    # Regex to find percentage used, typically the second to last field before mount point
                    match_df = re.search(r'\s+(\d+)%\s+(?:/[^\s]*)?$', lines[-1])
                    if match_df:
                        try: state['resources']['disk_usage_percent'] = float(match_df.group(1))
                        except ValueError: state['probe_errors'].append("Failed to parse disk usage percentage.")
                    else: state['probe_errors'].append("Could not parse 'df -k' output for disk usage.")
                else: state['probe_errors'].append("'df -k' output format unexpected for disk usage.")

            # Filesystem CWD listing
            state['filesystem'][current_cwd]={'type':'directory','content_listing':None,'error':None, 'exists': True}
            ls_res = probe_results.get('ls_cwd')
            if ls_res:
                state['filesystem'][current_cwd]['content_listing'] = ls_res['stdout'] if ls_res.get('success') else None
                state['filesystem'][current_cwd]['error'] = None if ls_res.get('success') else ls_res.get('stderr', 'ls command failed')

            # Filesystem Target Hint (if probed)
            stat_res = probe_results.get('stat_target')
            if stat_res and abs_target_hint_path and abs_target_hint_path in state['filesystem']: # Ensure entry exists
                stat_entry = state['filesystem'][abs_target_hint_path] # Modify existing entry
                stat_out = stat_res['stdout'] if stat_res.get('success') else None
                stat_entry['stat_output'] = stat_out # Store raw stat output
                if stat_out:
                    stat_entry['exists'] = True
                    if 'directory' in stat_out: stat_entry['type'] = 'directory'
                    elif 'regular empty file' in stat_out: stat_entry['type'] = 'file'; stat_entry['size_bytes'] = 0
                    elif 'regular file' in stat_out: stat_entry['type'] = 'file'
                    elif 'symbolic link' in stat_out: stat_entry['type'] = 'symlink' # Handle symlinks
                    else: stat_entry['type'] = 'other'

                    size_m = re.search(r"Size:\s*(\d+)", stat_out)
                    mtime_m = re.search(r"Modify:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", stat_out) # Standard stat format
                    # Fallback mtime for systems with different stat output (e.g. busybox)
                    if not mtime_m: mtime_m = re.search(r"Change:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", stat_out) # ctime
                    perms_m = re.search(r"Access:\s*\((\d+)/([a-zA-Z-]+)\)", stat_out) # e.g. (0644/-rw-r--r--)
                    owner_m = re.search(r"Uid:\s*\(\s*\d+/\s*([\w-]+)\)", stat_out) # e.g. ( 1000/   user)

                    if size_m: stat_entry['size_bytes'] = int(size_m.group(1))
                    if mtime_m: stat_entry['mtime'] = mtime_m.group(1).strip() # Timestamp string
                    if perms_m: stat_entry['perms_octal'] = perms_m.group(1); stat_entry['perms_symbolic'] = perms_m.group(2)
                    if owner_m: stat_entry['owner'] = owner_m.group(1).strip()
                    stat_entry['error'] = None # Clear pending error if stat successful
                else: # Stat command failed
                    stat_entry['exists'] = False
                    stat_entry['error'] = stat_res.get('stderr', 'Stat command failed with no stderr')
                    # Refine reason based on stderr if possible
                    stderr_stat_lower = (stat_entry['error'] or "").lower()
                    if "no such file or directory" in stderr_stat_lower:
                         stat_entry['reason_stat_fail'] = 'file_not_found'
                    elif "permission denied" in stderr_stat_lower:
                         stat_entry['reason_stat_fail'] = 'permission_denied'


        except Exception as parse_err:
            logger.error(f"Error parsing real system state from probe results: {parse_err}", exc_info=True)
            state['parsing_error'] = f"State parsing failed: {parse_err}"

        if not state.get('probe_errors'): state.pop('probe_errors', None) # Clean up if no errors
        return state

    def disconnect(self):
        if self.docker_client:
            try:
                logger.info("Closing Docker client connection...")
                self.docker_client.close()
                logger.info("Docker client closed.")
            except Exception as e:
                logger.error(f"Error closing Docker client: {e}", exc_info=True)
            finally:
                self.docker_client = None
                self.docker_container = None
        logger.info("Seed VMService disconnected/cleaned up.")

    def get_current_allowed_commands(self) -> List[str]:
        """ Returns the list of currently allowed commands based on mode. """
        if self.use_real_system:
            if self.allowed_real_commands_config_value == VM_SERVICE_ALLOW_ALL_COMMANDS_MAGIC_VALUE:
                return ["__ALLOW_ALL_COMMANDS__"] # Indicate all are allowed
            return self._effective_allowed_commands_list
        else: # Simulation mode
            return self._sim_available_commands if hasattr(self, '_sim_available_commands') else []