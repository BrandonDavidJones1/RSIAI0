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

logger = logging.getLogger(__name__)

# --- Configuration ---
# Use relative import now
from .config import (
    VM_SERVICE_USE_REAL, VM_SERVICE_DOCKER_CONTAINER,
    VM_SERVICE_COMMAND_TIMEOUT_SEC, VM_SERVICE_ALLOWED_REAL_COMMANDS,
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
    logger.info("Docker library not found, Docker interaction disabled for VMService.")

# Renamed class
class Seed_VMService:
    """ Simulates or interacts with an external system (VM/Docker/Subprocess). """
    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 use_real_system: bool = VM_SERVICE_USE_REAL,
                 docker_container_name: Optional[str] = VM_SERVICE_DOCKER_CONTAINER):
        """ Initializes the VMService, choosing mode and setting up connections/state. """
        self.config = config if config else {}
        # Core commands needed for probing and basic file ops in real mode
        self._core_real_probes = ['sh', 'pwd', 'ls', 'df', 'stat', 'grep', 'head', 'tail', 'top', 'cat', 'echo', 'rm', 'touch', 'mkdir', 'mv', 'cp', 'printf']
        base_allowed = VM_SERVICE_ALLOWED_REAL_COMMANDS if isinstance(VM_SERVICE_ALLOWED_REAL_COMMANDS, list) else []
        # Ensure core probes are always allowed if real mode is possible
        self.allowed_real_commands: List[str] = list(set(base_allowed + self._core_real_probes))

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

            logger.info(f"Initializing Seed VMService (Real System Mode: {mode_name}). Allowed commands: {sorted(self.allowed_real_commands)}")
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
                logger.info(f"Docker container initial CWD detected as: '{container_cwd}'. Setting internal CWD.")
                self._real_system_cwd = container_cwd # Set internal CWD to match container's default
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
        self._sim_available_commands = ['ls', 'pwd', 'cat', 'echo', 'touch', 'mkdir', 'rm', 'cd', 'cp', 'mv']

        # --- Placeholder content for test files ---
        # <<< IMPORTANT: Replace placeholders below with your actual test file content >>>
        # Example: Use multiline strings or read from actual files if needed
        test_core_content = """
# RSIAI0/seed/tests/test_core.py
# Placeholder - Replace with actual test code provided previously
import pytest
def test_placeholder_core():
    assert True
"""
        test_memory_content = """
# RSIAI0/seed/tests/test_memory.py
# Placeholder - Replace with actual test code provided previously
import pytest
def test_placeholder_memory():
    assert True
"""
        # --- End Placeholder content ---

        # Using standard read/write permissions for user now for simplicity
        self._simulated_state = {
            'timestamp': time.time(), 'cwd': self._simulated_system_cwd,
            'filesystem': {
                # Standard directories and files
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

                # --- ADDED Test Directory and Files ---
                '/app/seed/tests': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/seed/tests/__init__.py': {'type': 'file', 'content': '', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0},
                '/app/seed/tests/test_core.py': {'type': 'file', 'content': test_core_content, 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(test_core_content.encode())},
                '/app/seed/tests/test_memory.py': {'type': 'file', 'content': test_memory_content, 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(test_memory_content.encode())},
                # --- END OF ADDED ---
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
                 # Combine and normalize using PurePosixPath logic
                 # Use resolve() for potentially better handling of '..'
                 # Note: resolve() might fail on non-existent paths in some Path impls,
                 # but PurePosixPath should handle normalization logically.
                 # Let's stick to the simpler join unless resolve proves necessary and reliable here.
                 # combined_path = (current_path / posix_path).resolve() # More complex
                 combined_path = current_path / posix_path # Simpler join
                 # Normalize manually for robustness across Path versions/contexts
                 parts = []
                 for part in combined_path.parts:
                     if part == '.': continue
                     if part == '..':
                         if parts and parts[-1] != '/': parts.pop()
                         elif not parts or parts == ['/']: continue # Ignore .. on root
                     else:
                         parts.append(part)
                 if not parts or parts == ['/']: posix_path = PurePosixPath('/')
                 else: posix_path = PurePosixPath('/' + '/'.join(p for p in parts if p != '/'))

                 logger.debug(f"Resolved relative CWD '{new_cwd}' to '{posix_path}'")

            normalized_cwd = str(posix_path)

            # Add check for simulation mode to prevent setting CWD to non-existent dir
            if not self.use_real_system and self._simulated_state:
                fs = self._simulated_state['filesystem']
                # Check if the target exists AND is a directory in the simulation
                if normalized_cwd not in fs or fs[normalized_cwd].get('type') != 'directory':
                     logger.warning(f"Attempted to set CWD to non-existent/non-directory path in simulation: '{normalized_cwd}'. CWD unchanged.")
                     return # Exit without changing CWD if invalid in simulation

            if self.use_real_system:
                self._real_system_cwd = normalized_cwd
            else:
                self._simulated_system_cwd = normalized_cwd
        except (TypeError, ValueError) as e:
            logger.error(f"Attempted to set invalid CWD '{new_cwd}': {e}")

    # --- Path/Permission Helpers ---
    def _resolve_path(self, path_str: str) -> Optional[str]:
        """ Resolves a given path string relative to the current CWD using PurePosixPath. Returns absolute POSIX path string or None. """
        current_dir = self._get_current_cwd()
        if not path_str:
            return current_dir
        try:
            cwd_path = PurePosixPath(current_dir)
            target_path: PurePosixPath

            if path_str == '~': target_path = PurePosixPath('/app') # Simple home assumption
            elif path_str.startswith('~/'): target_path = PurePosixPath('/app') / path_str[2:]
            else: input_path = PurePosixPath(path_str); target_path = cwd_path / input_path

            # Normalize using os.path.normpath after converting to string, ensure POSIX separators
            resolved_path_str = os.path.normpath(str(target_path)).replace('\\', '/')

            # Ensure it starts with / if it's absolute after normalization
            # Check if the normalized path still represents an absolute path concept
            # A simple check is if it starts with '/' or drive letter (though less relevant for POSIX focus)
            is_absolute_after_norm = resolved_path_str.startswith('/') # Sufficient for POSIX

            if not is_absolute_after_norm:
                 # Re-join with current dir if normalization resulted in relative path (e.g., '.', 'some_dir')
                 resolved_path_str = str(PurePosixPath(current_dir) / resolved_path_str)
                 # Re-normalize after re-joining
                 resolved_path_str = os.path.normpath(resolved_path_str).replace('\\', '/')

            # Final check to ensure it starts with '/' if it's not just '/'
            if not resolved_path_str.startswith('/') and resolved_path_str != '/':
                 logger.warning(f"Path resolution resulted in non-absolute POSIX path '{resolved_path_str}'. Prepending '/'.")
                 resolved_path_str = '/' + resolved_path_str

            logger.debug(f"Resolved path '{path_str}' from '{current_dir}' to '{resolved_path_str}'")
            return resolved_path_str

        except Exception as e:
            logger.error(f"Path resolution error for '{path_str}' from '{current_dir}': {e}", exc_info=True)
            return None


    def _sim_check_permissions(self, path: str, action: str = 'read') -> Tuple[bool, str]:
        """ Checks permissions (currently bypassed - returns True). """
        logger.debug(f"Sim permission check skipped for action '{action}' on '{path}' (Always returning True).")
        return (True, "")

    # --- NEW: Case-Insensitive Lookup Helpers ---
    def _find_case_insensitive_match(self, parent_path_str: str, filename_str: str) -> Optional[str]:
        """
        Looks for a unique case-insensitive filename match within a parent directory.

        Args:
            parent_path_str: The absolute path of the parent directory.
            filename_str: The target filename (potentially with incorrect casing).

        Returns:
            The correctly cased absolute path if a unique match is found, otherwise None.
        """
        logger.debug(f"Attempting case-insensitive search for '{filename_str}' in '{parent_path_str}'")
        matches = []
        target_filename_lower = filename_str.lower()

        if not self.use_real_system:
            # Simulation Mode
            if not self._simulated_state: return None
            fs = self._simulated_state['filesystem']
            norm_parent_path = str(PurePosixPath(parent_path_str)) # Normalize for comparison
            for path_key in fs.keys():
                try:
                    entry_path = PurePosixPath(path_key)
                    # Check if entry is directly within the parent directory
                    # Handle root directory case correctly
                    if str(entry_path.parent) == norm_parent_path:
                         entry_filename = entry_path.name
                         if entry_filename.lower() == target_filename_lower:
                             matches.append(path_key) # Store the correctly cased full path
                except Exception as e:
                    logger.warning(f"Error processing path '{path_key}' during case-insensitive search: {e}")
                    continue # Skip problematic paths
        else:
            # Real Mode
            if 'ls' not in self.allowed_real_commands:
                logger.warning("Cannot perform case-insensitive search: 'ls' command not allowed.")
                return None
            # List directory contents including hidden files, one item per line
            ls_cmd = f"ls -1a {shlex.quote(parent_path_str)}"
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            ls_res = self._execute_real_command(ls_cmd, exec_func)

            if ls_res.get('success'):
                # Split output into lines and filter empty lines and '.' '..'
                potential_files = [line for line in ls_res.get('stdout', '').splitlines() if line and line not in ['.', '..']]
                for potential_file in potential_files:
                    if potential_file.lower() == target_filename_lower:
                        # Construct the full path using PurePosixPath for correctness
                        try:
                           full_match_path = str(PurePosixPath(parent_path_str) / potential_file)
                           matches.append(full_match_path)
                        except Exception as path_e:
                            logger.error(f"Error constructing path for matched file '{potential_file}' in '{parent_path_str}': {path_e}")
            else:
                # Log if ls failed, but don't necessarily stop (maybe dir was empty or permissions issue)
                logger.warning(f"Case-insensitive search: 'ls' command failed in '{parent_path_str}'. Stderr: {ls_res.get('stderr')}")

        # Evaluate Matches
        if len(matches) == 1:
            logger.info(f"Found unique case-insensitive match for '{filename_str}' in '{parent_path_str}': '{matches[0]}'")
            return matches[0]
        elif len(matches) > 1:
            logger.warning(f"Found multiple ({len(matches)}) case-insensitive matches for '{filename_str}' in '{parent_path_str}': {matches}. Ambiguous.")
            return None # Ambiguous
        else:
            logger.debug(f"Found no case-insensitive match for '{filename_str}' in '{parent_path_str}'.")
            return None

    def _get_potential_matches(self, parent_path_str: str, filename_str: str) -> List[str]:
        """ Helper to get list of case-insensitive matches (internal use). """
        matches = []
        target_filename_lower = filename_str.lower()
        if not self.use_real_system:
            # --- Simulation Mode ---
            if not self._simulated_state: return []
            fs = self._simulated_state['filesystem']
            norm_parent_path = str(PurePosixPath(parent_path_str))
            for path_key in fs.keys():
                if path_key == parent_path_str or path_key == '/': continue
                try:
                    entry_path = PurePosixPath(path_key)
                    if str(entry_path.parent) == norm_parent_path:
                         entry_filename = entry_path.name
                         if entry_filename.lower() == target_filename_lower:
                             matches.append(path_key)
                except Exception: continue
        else:
            # --- Real Mode ---
            if 'ls' not in self.allowed_real_commands: return []
            ls_cmd = f"ls -1a {shlex.quote(parent_path_str)}"
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            ls_res = self._execute_real_command(ls_cmd, exec_func)
            if ls_res.get('success'):
                potential_files = [line for line in ls_res.get('stdout', '').splitlines() if line and line not in ['.', '..']]
                for potential_file in potential_files:
                    if potential_file.lower() == target_filename_lower:
                        try:
                           full_match_path = str(PurePosixPath(parent_path_str) / potential_file)
                           matches.append(full_match_path)
                        except Exception: continue
        return matches
    # --- End Case-Insensitive Helpers ---

    # --- End Path/Permission Helpers --- (Original Comment Location)


    # --- Command Execution ---
    def execute_command(self, command_str: str) -> Dict[str, Any]:
        """ Executes command in simulation OR real system, handling CWD internally. """
        if self.use_real_system:
            exec_context = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            return self._execute_real_command(command_str, exec_context)
        else:
            return self._execute_simulated_command(command_str)

    def _execute_simulated_command(self, command_str: str) -> Dict[str, Any]:
        """ Executes command within the simulated filesystem (with case-insensitive lookups). """
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
            # --- Simulated Command Logic (with case-insensitive checks integrated) ---
            if cmd == 'pwd':
                result['stdout'], result['success'], result['exit_code'] = current_cwd, True, 0
            elif cmd == 'cd':
                target_dir_str = args[0] if args else '~'; resolved_path = resolve_path_func(target_dir_str); target_path = resolved_path
                if resolved_path and resolved_path not in fs:
                    parent = str(PurePosixPath(resolved_path).parent); filename = PurePosixPath(resolved_path).name; found_match = find_match_func(parent, filename)
                    if found_match: target_path = found_match; logger.debug(f"cd: Using corrected path '{target_path}' for '{target_dir_str}'")
                    else: result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found'; return result # Exit early if not found after check
                # Check the potentially corrected path
                if target_path and target_path in fs and fs[target_path].get('type') == 'directory': self._set_current_cwd(target_path); self._simulated_state['cwd'] = self._get_current_cwd(); result['success'], result['exit_code'] = True, 0
                elif target_path and target_path in fs: result['stderr'], result['reason'] = f"cd: Not a directory: {target_dir_str}", 'is_not_directory'
                elif not target_path: result['stderr'], result['reason'] = f"cd: Invalid path resolution for: {target_dir_str}", 'invalid_path'
                else: result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found' # Should be caught earlier
            elif cmd == 'ls':
                target_path_str = args[0] if args else '.'; resolved_path = resolve_path_func(target_path_str); target_path = resolved_path
                if resolved_path and resolved_path not in fs:
                    parent = str(PurePosixPath(resolved_path).parent); filename = PurePosixPath(resolved_path).name; found_match = find_match_func(parent, filename)
                    if found_match: target_path = found_match; logger.debug(f"ls: Using corrected path '{target_path}' for '{target_path_str}'")
                    else: result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found'; return result
                # Check the potentially corrected path
                if target_path and target_path in fs:
                    item_info = fs[target_path]
                    if item_info.get('type') == 'directory': parent_path_str = target_path if target_path != '/' else target_path; contents = [PurePosixPath(n).name for n, f in fs.items() if str(PurePosixPath(n).parent) == parent_path_str and n != '/']; result['stdout'], result['success'], result['exit_code'] = "\n".join(sorted(contents)), True, 0
                    else: result['stdout'], result['success'], result['exit_code'] = PurePosixPath(target_path).name, True, 0
                elif not target_path: result['stderr'], result['reason'] = f"ls: invalid path resolution for '{target_path_str}'", 'invalid_path'
                else: result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found' # Should be caught earlier
            elif cmd == 'cat':
                if not args: result['stderr'], result['reason'] = "cat: missing file operand", 'missing_args'; return result
                p = resolve_path_func(args[0]); target_p = p
                if p and p not in fs: parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name; found_match = find_match_func(parent, filename);
                if found_match: target_p = found_match
                else: result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found'; return result
                # Use target_p for checks
                if target_p and target_p in fs:
                    info = fs[target_p]
                    if info.get('type') == 'file': result['stdout'], result['success'], result['exit_code'] = info.get('content',''), True, 0
                    else: result['stderr'], result['reason'] = f"cat: {args[0]}: Is a directory", 'is_directory'
                elif not target_p: result['stderr'], result['reason'] = f"cat: invalid path resolution for '{args[0]}'", 'invalid_path'
                else: result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found' # Should have been caught
            elif cmd == 'touch':
                if not args: result['stderr'], result['reason'] = "touch: missing file operand", 'missing_args'; return result
                p = resolve_path_func(args[0]); target_p = p # target_p will be the path used, possibly corrected
                if p and p not in fs:
                    parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name
                    found_match = find_match_func(parent, filename);
                    if found_match:
                        target_p = found_match # Use existing cased path
                        logger.debug(f"touch: Targeting existing file/dir '{target_p}' (case-insensitive match)")
                # Operate on target_p (which is original resolved p if no match found or p was already in fs)
                if target_p:
                    if target_p in fs: # Update existing file/dir mtime
                        fs[target_p]['mtime'] = time.time()
                    else: # Create new file with casing from target_p (which is from resolve_path_func)
                        fs[target_p] = {'type': 'file', 'content': '', 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0}
                    result['success'], result['exit_code'] = True, 0
                else: result['stderr'], result['reason'] = f"touch: Invalid path resolution for: {args[0]}", 'invalid_path'
            elif cmd == 'mkdir':
                 if not args: result['stderr'], result['reason'] = "mkdir: missing operand", 'missing_args'; return result
                 p = resolve_path_func(args[0])
                 if p:
                     # mkdir should not use case-insensitive match to decide it "exists" if casing is different. It should create with given casing or fail if exact path exists.
                     if p in fs: result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': File exists", 'file_exists'
                     else:
                         parent_p_str = str(PurePosixPath(p).parent)
                         # Parent check: must exist with exact case or case-insensitive match
                         actual_parent_p = parent_p_str
                         if parent_p_str not in fs:
                             grandparent = str(PurePosixPath(parent_p_str).parent)
                             parent_name = PurePosixPath(parent_p_str).name
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
                 if p and p not in fs: parent = str(PurePosixPath(p).parent); filename = PurePosixPath(p).name; found_match = find_match_func(parent, filename);
                 if found_match: target_p = found_match; logger.debug(f"rm: Targeting existing file/dir '{target_p}' (case-insensitive match)")
                 else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found'; return result
                 # Operate on target_p
                 if target_p and target_p in fs:
                     info = fs[target_p]
                     if info.get('type') == 'directory':
                         # Check if directory is empty BEFORE deleting
                         is_empty = not any(str(PurePosixPath(n).parent)==target_p for n in fs if n != target_p)
                         if is_empty: del fs[target_p]; result['success'], result['exit_code'] = True, 0
                         else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': Directory not empty", 'directory_not_empty'
                     else: # It's a file
                          del fs[target_p]; result['success'], result['exit_code'] = True, 0
                 elif target_p == '/': result['stderr'], result['reason'] = "rm: cannot remove root directory", 'permission_denied'
                 elif not target_p: result['stderr'], result['reason'] = f"rm: invalid path resolution for '{args[0]}'", 'invalid_path'
                 else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found' # Should have been caught
            
            # --- START OF NEW/COMPLETED LOGIC ---
            elif cmd == 'echo':
                content_to_echo = ""
                target_file_str = None # The string path given by user for redirection
                redirect_mode = None # '>', '>>'
                
                # Parse arguments for echo
                idx = 0
                while idx < len(args):
                    if args[idx] == '>' or args[idx] == '>>':
                        redirect_mode = args[idx]
                        if idx + 1 < len(args):
                            target_file_str = args[idx+1]
                            # Content is everything before the redirect operator
                            content_to_echo = " ".join(args[:idx])
                            idx = len(args) # Break loop
                        else:
                            result['stderr'],result['reason']="echo: missing target file for redirection",'missing_args'; return result
                        break # Found redirect, break from arg parsing
                    idx += 1
                
                if not redirect_mode: # No redirection found, all args are content
                    content_to_echo = " ".join(args)

                # Strip common outer quotes (simple approach)
                if content_to_echo.startswith('"') and content_to_echo.endswith('"'):
                    content_to_echo = content_to_echo[1:-1]
                elif content_to_echo.startswith("'") and content_to_echo.endswith("'"):
                    content_to_echo = content_to_echo[1:-1]

                if not redirect_mode: # Echo to stdout
                    result['stdout'], result['success'], result['exit_code'] = content_to_echo, True, 0
                else: # Redirect to file
                    if not target_file_str: # Should be caught by parser, but defensive
                        result['stderr'],result['reason']="echo: missing target file for redirection",'missing_args'; return result
                    
                    resolved_target_path = resolve_path_func(target_file_str)
                    if not resolved_target_path:
                        result['stderr'], result['reason'] = f"echo: Invalid path resolution for redirection target: {target_file_str}", 'invalid_path'; return result

                    final_write_path = resolved_target_path
                    parent_dir = str(PurePosixPath(resolved_target_path).parent)
                    filename = PurePosixPath(resolved_target_path).name
                    existing_match_path = find_match_func(parent_dir, filename)

                    if existing_match_path:
                        if existing_match_path in fs and fs[existing_match_path].get('type') == 'directory':
                            result['stderr'], result['reason'] = f"echo: {target_file_str}: Is a directory", 'is_directory'; return result
                        final_write_path = existing_match_path
                    
                    current_file_content = ""
                    if redirect_mode == '>>':
                        if final_write_path in fs and fs[final_write_path].get('type') == 'file':
                            current_file_content = fs[final_write_path].get('content', '')
                            if current_file_content and not current_file_content.endswith('\n') and content_to_echo:
                                current_file_content += '\n'
                        # If appending to a non-existent file, it's like creating it.
                    
                    new_content = current_file_content + content_to_echo
                    
                    fs[final_write_path] = {
                        'type': 'file', 'content': new_content, 'owner': 'user',
                        'perms': 'rw-r--r--', 'mtime': time.time(),
                        'size_bytes': len(new_content.encode('utf-8'))
                    }
                    result['success'], result['exit_code'] = True, 0

            elif cmd == 'cp':
                if len(args) != 2:
                    result['stderr'], result['reason'] = "cp: missing file operand or too many arguments", 'missing_args'; return result
                src_arg, dest_arg = args[0], args[1]

                resolved_src = resolve_path_func(src_arg)
                if not resolved_src: result['stderr'], result['reason'] = f"cp: Invalid source path: {src_arg}", 'invalid_path'; return result
                
                actual_src_path = resolved_src
                if resolved_src not in fs:
                    match = find_match_func(str(PurePosixPath(resolved_src).parent), PurePosixPath(resolved_src).name)
                    if match: actual_src_path = match
                    else: result['stderr'], result['reason'] = f"cp: cannot stat '{src_arg}': No such file or directory", 'file_not_found'; return result
                
                if actual_src_path not in fs: result['stderr'],result['reason']=f"cp: source '{actual_src_path}' disappeared after check",'internal_error'; return result # Should not happen

                src_info = fs[actual_src_path]
                if src_info.get('type') == 'directory': result['stderr'], result['reason'] = f"cp: omitting directory '{src_arg}' (no -r)", 'is_directory'; return result
                if src_info.get('type') != 'file': result['stderr'], result['reason'] = f"cp: '{src_arg}' is not a regular file", 'invalid_type'; return result

                resolved_dest_arg = resolve_path_func(dest_arg)
                if not resolved_dest_arg: result['stderr'], result['reason'] = f"cp: Invalid destination path: {dest_arg}", 'invalid_path'; return result

                final_dest_path = resolved_dest_arg
                # Check if destination is an existing directory (case-insensitive)
                dest_parent_for_match = str(PurePosixPath(resolved_dest_arg).parent)
                dest_filename_for_match = PurePosixPath(resolved_dest_arg).name
                existing_dest_dir_match = find_match_func(dest_parent_for_match, dest_filename_for_match)

                if existing_dest_dir_match and existing_dest_dir_match in fs and fs[existing_dest_dir_match].get('type') == 'directory':
                    src_basename = PurePosixPath(actual_src_path).name
                    path_in_dir = str(PurePosixPath(existing_dest_dir_match) / src_basename)
                    # Check if file with src_basename (case-insensitively) exists in this target dir
                    file_match_in_target_dir = find_match_func(existing_dest_dir_match, src_basename)
                    if file_match_in_target_dir:
                        final_dest_path = file_match_in_target_dir # Overwrite this existing file
                    else:
                        final_dest_path = path_in_dir # Create new file in dir
                elif existing_dest_dir_match and existing_dest_dir_match in fs : # Is an existing file
                    final_dest_path = existing_dest_dir_match # Overwrite this file
                # Else: final_dest_path remains resolved_dest_arg (create new file with this casing)
                
                if actual_src_path == final_dest_path:
                    result['stderr'], result['reason'] = f"cp: '{src_arg}' and '{dest_arg}' are the same file", 'invalid_argument'; return result

                fs[final_dest_path] = copy.deepcopy(src_info)
                fs[final_dest_path]['owner'] = 'user'
                fs[final_dest_path]['mtime'] = time.time()
                result['success'], result['exit_code'] = True, 0
            
            elif cmd == 'mv':
                if len(args) != 2:
                    result['stderr'], result['reason'] = "mv: missing file operand or too many arguments", 'missing_args'; return result
                src_arg, dest_arg = args[0], args[1]

                resolved_src = resolve_path_func(src_arg)
                if not resolved_src: result['stderr'], result['reason'] = f"mv: Invalid source path: {src_arg}", 'invalid_path'; return result

                actual_src_path = resolved_src
                if resolved_src not in fs:
                    match = find_match_func(str(PurePosixPath(resolved_src).parent), PurePosixPath(resolved_src).name)
                    if match: actual_src_path = match
                    else: result['stderr'], result['reason'] = f"mv: cannot stat '{src_arg}': No such file or directory", 'file_not_found'; return result
                
                if actual_src_path not in fs: result['stderr'],result['reason']=f"mv: source '{actual_src_path}' disappeared after check",'internal_error'; return result
                src_info_copy = copy.deepcopy(fs[actual_src_path]) # Copy before potential pop

                resolved_dest_arg = resolve_path_func(dest_arg)
                if not resolved_dest_arg: result['stderr'], result['reason'] = f"mv: Invalid destination path: {dest_arg}", 'invalid_path'; return result

                final_dest_path = resolved_dest_arg
                existing_dest_dir_match = find_match_func(str(PurePosixPath(resolved_dest_arg).parent), PurePosixPath(resolved_dest_arg).name)

                if existing_dest_dir_match and existing_dest_dir_match in fs and fs[existing_dest_dir_match].get('type') == 'directory':
                    src_basename = PurePosixPath(actual_src_path).name
                    path_in_dir = str(PurePosixPath(existing_dest_dir_match) / src_basename)
                    file_match_in_target_dir = find_match_func(existing_dest_dir_match, src_basename)
                    if file_match_in_target_dir: final_dest_path = file_match_in_target_dir
                    else: final_dest_path = path_in_dir
                elif existing_dest_dir_match and existing_dest_dir_match in fs:
                    final_dest_path = existing_dest_dir_match
                
                if actual_src_path == final_dest_path:
                    result['stderr'], result['reason'] = f"mv: '{src_arg}' and '{dest_arg}' are the same file", 'invalid_argument'; return result

                # Check for moving a directory into itself or a subdirectory
                if src_info_copy.get('type') == 'directory' and final_dest_path.startswith(actual_src_path + '/'):
                    result['stderr'], result['reason'] = f"mv: cannot move '{src_arg}' to a subdirectory of itself, '{dest_arg}'", 'invalid_argument'; return result

                # Handle overwriting existing destination based on types
                if final_dest_path in fs:
                    dest_info_at_final = fs[final_dest_path]
                    if src_info_copy.get('type') == 'directory': # Moving a directory
                        if dest_info_at_final.get('type') == 'file':
                            result['stderr'], result['reason'] = f"mv: cannot overwrite non-directory '{final_dest_path}' with directory '{src_arg}'", 'overwrite_file_with_dir'; return result
                        elif dest_info_at_final.get('type') == 'directory': # Moving dir onto existing dir
                            is_dest_empty = not any(str(PurePosixPath(k).parent) == final_dest_path for k in fs if k != final_dest_path and k != actual_src_path)
                            if not is_dest_empty:
                                result['stderr'], result['reason'] = f"mv: failed to move '{src_arg}' to '{dest_arg}': Directory not empty ('{final_dest_path}')", 'directory_not_empty'; return result
                    elif src_info_copy.get('type') == 'file': # Moving a file
                        if dest_info_at_final.get('type') == 'directory': # Moving file onto dir (final_dest_path constructed to be /dir/filename)
                             # This case implies final_dest_path = /dir/filename is itself a directory, which is an error.
                             result['stderr'], result['reason'] = f"mv: cannot overwrite directory '{final_dest_path}' with non-directory '{src_arg}'", 'overwrite_dir_with_file'; return result
                
                # Perform move
                data_to_move = fs.pop(actual_src_path)
                fs[final_dest_path] = data_to_move
                fs[final_dest_path]['mtime'] = time.time()
                result['success'], result['exit_code'] = True, 0
            # --- END OF NEW/COMPLETED LOGIC ---
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
        """ Prepares and executes a command in the real system context (Docker/Subprocess), handling CWD. """
        # This method remains unchanged from the block you provided
        result: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'command': command_str, 'reason': '', 'exit_code': -1}
        try: parts = shlex.split(command_str)
        except ValueError as e: result['stderr'], result['reason'] = f"Command parse error: {e}", 'parse_error'; return result
        command = parts[0] if parts else ''
        if command not in self.allowed_real_commands: result['stderr'], result['reason'] = f"Command '{command}' not allowed for real execution.", 'safety_violation'; logger.warning(f"Safety Violation: Blocked real system execution: '{command_str}'"); return result
        try:
             if command == 'cd':
                 if len(parts) == 1: target_dir_str = '~'
                 elif len(parts) == 2: target_dir_str = parts[1]
                 else: result['stderr'], result['reason'] = "cd: too many arguments", 'invalid_argument'; return result
                 if target_dir_str == '~': target_dir = '/app' # Default assumption
                 else: target_dir = self._resolve_path(target_dir_str)
                 if not target_dir: result['stderr'], result['reason'] = f"cd: Invalid path resolution for: {target_dir_str}", 'invalid_path'; return result
                 check_cmd = f"ls -ld {shlex.quote(target_dir)}"; check_res = execution_context(check_cmd)
                 if check_res.get('success') and check_res.get('stdout','').strip().startswith('d'): self._set_current_cwd(target_dir); result['success'], result['exit_code'] = True, 0; logger.info(f"Real system CWD updated to: {self._get_current_cwd()}")
                 elif check_res.get('exit_code') != 0: stderr = check_res.get('stderr', '').lower(); reason = 'execution_error';
                 if 'no such file or directory' in stderr: reason = 'file_not_found'
                 elif 'permission denied' in stderr: reason = 'permission_denied'; result.update({'success':False, 'exit_code':check_res.get('exit_code', 1), 'stderr':f"cd: cannot access '{target_dir_str}': {check_res.get('stderr','Unknown error')}", 'reason':reason})
                 else: result.update({'success':False, 'exit_code':1, 'stderr':f"cd: not a directory: {target_dir_str}", 'reason':'is_not_directory'})
                 return result
             else: return execution_context(command_str)
        except Exception as e: logger.error(f"Unexpected error during real command prep/exec '{command_str}': {e}", exc_info=True); result['stderr'], result['reason'] = f"Unexpected internal error: {e}", 'internal_error'; return result

    def _docker_exec_context(self, command_str: str) -> Dict[str, Any]:
        """ Executes a command inside the configured Docker container. """
        # This method remains unchanged from the block you provided
        res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
        if not self.docker_container: res['stderr']='Docker container unavailable'; res['reason']='docker_error'; return res
        try:
            full_cmd = f"sh -c {shlex.quote(command_str)}"; logger.debug(f"Docker Exec Run: cmd='{full_cmd}', workdir='{self._get_current_cwd()}'")
            exit_code, output = self.docker_container.exec_run(cmd=full_cmd, workdir=self._get_current_cwd(), stream=False, demux=False, user='root')
            output_bytes: bytes = output if isinstance(output, bytes) else b''; output_str = output_bytes.decode('utf-8', errors='replace').strip()
            res['exit_code'] = exit_code; res['success'] = (exit_code == 0); res['stdout'] = output_str if res['success'] else ''; res['stderr'] = '' if res['success'] else output_str
            if not res['success']: res['reason'] = 'execution_error';
            # <<< FIX: Ensure stderr is populated even if output_str is empty on failure >>>
            if not res['success'] and not res['stderr']: res['stderr'] = f"Command failed (Code {res['exit_code']}) with no output."
            # <<< END FIX >>>
            return res
        except DockerAPIError as api_err: logger.error(f"Docker API error executing '{command_str}': {api_err}", exc_info=True); res.update({'stderr':f"Docker API error: {api_err}", 'reason':'docker_api_error'}); return res
        except Exception as e: logger.error(f"Docker exec_run unexpected error '{command_str}': {e}", exc_info=True); res.update({'stderr':f"Docker exec_run error: {e}", 'reason':'docker_error'}); return res

    def _subprocess_exec_context(self, command_str: str) -> Dict[str, Any]:
         """ Executes a command using the local subprocess module. """
         # This method remains unchanged from the block you provided
         res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
         current_cwd = self._get_current_cwd(); logger.debug(f"Subprocess Exec Run: cmd='{command_str}', cwd='{current_cwd}'")
         try:
            proc = subprocess.run(command_str, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=self.command_timeout_sec, check=False, cwd=current_cwd)
            res['exit_code'] = proc.returncode; res['stdout'] = proc.stdout.strip() if proc.stdout else ''; res['stderr'] = proc.stderr.strip() if proc.stderr else ''; res['success'] = (proc.returncode == 0)
            if not res['success']: res['reason'] = 'execution_error';
            # <<< FIX: Ensure stderr is populated even if proc.stderr is empty on failure >>>
            if not res['success'] and not res['stderr']: res['stderr'] = f"Command failed (Code {proc.returncode}) with no stderr."
             # <<< END FIX >>>
            return res
         except FileNotFoundError: res['stderr'], res['reason'] = f"Command or shell not found: {command_str.split()[0]}", 'command_not_found'; res['exit_code'] = 127; return res
         except subprocess.TimeoutExpired: res['stderr'], res['reason'] = f"Timeout ({self.command_timeout_sec}s)", 'timeout'; res['exit_code'] = -9; return res
         except Exception as e: logger.error(f"Subprocess error executing '{command_str}': {e}", exc_info=True); res['stderr']=f"Subprocess exec error: {e}"; res['reason']='internal_error'; return res
    # --- End Command Execution ---

    # --- Filesystem Operations (Updated to use case-insensitive logic) ---
    def read_file(self, path: str) -> Dict[str, Any]:
        """ Reads file content from simulation OR real system, attempting case-insensitive lookup on failure. """
        logger.info(f"Seed VMService: Reading file '{path}'")
        result = {'success': False, 'content': None, 'message': '', 'details': {'path': path}, 'reason': ''}
        original_abs_path = self._resolve_path(path)
        if not original_abs_path: result['message'] = "Invalid path resolution."; result['reason'] = 'invalid_path'; return result
        result['details']['absolute_path'] = original_abs_path
        abs_path = original_abs_path # Start with the exact path

        def _perform_read(read_path: str) -> Dict[str, Any]:
            read_result = {'success': False, 'content': None, 'message': '', 'reason': '', 'details': {}} # Add details dict
            if self.use_real_system:
                if 'cat' not in self.allowed_real_commands: read_result['message'] = "Cannot read file: 'cat' command not allowed."; read_result['reason'] = 'safety_violation'; return read_result
                exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context; cat_res = self._execute_real_command(f"cat {shlex.quote(read_path)}", exec_func)
                read_result['reason'] = cat_res.get('reason', 'execution_error'); read_result['details'] = {'exit_code': cat_res.get('exit_code', -1), 'stderr': cat_res.get('stderr')}
                if cat_res.get('success'): read_result['success'] = True; read_result['content'] = cat_res.get('stdout', ''); read_result['message'] = "File read successfully."
                else:
                    read_result['message'] = f"Failed to read file: {cat_res.get('stderr', 'Unknown error')}"; stderr_lower = (cat_res.get('stderr') or '').lower()
                    if 'no such file or directory' in stderr_lower: read_result['reason'] = 'file_not_found'
                    elif 'is a directory' in stderr_lower: read_result['reason'] = 'is_directory'
                    elif 'permission denied' in stderr_lower: read_result['reason'] = 'permission_denied'
            else: # Simulation mode
                if not self._simulated_state: read_result['message'] = "Simulation not initialized."; read_result['reason'] = 'internal_error'; return read_result
                fs = self._simulated_state['filesystem']
                if read_path in fs:
                    item_info = fs[read_path]
                    # Permissions check bypassed
                    if item_info.get('type') == 'file': read_result['success'] = True; read_result['content'] = item_info.get('content', ''); read_result['message'] = "File read successfully (simulation)."
                    elif item_info.get('type') == 'directory': read_result['message'] = "Cannot read: Is a directory."; read_result['reason'] = 'is_directory'
                    else: read_result['message'] = f"Cannot read: Not a file (Type: {item_info.get('type')})."; read_result['reason'] = 'invalid_type'
                else: read_result['message'] = "File not found (simulation)."; read_result['reason'] = 'file_not_found'
            return read_result

        # Attempt 1: Read exact path
        attempt1_result = _perform_read(abs_path)

        # Attempt 2: Case-insensitive lookup if Attempt 1 failed with 'file_not_found'
        if not attempt1_result['success'] and attempt1_result['reason'] == 'file_not_found':
            logger.info(f"Read failed for exact path '{abs_path}', attempting case-insensitive lookup.")
            try:
                parent_path = str(PurePosixPath(abs_path).parent); filename = PurePosixPath(abs_path).name; corrected_path = self._find_case_insensitive_match(parent_path, filename)
                if corrected_path:
                    logger.info(f"Found corrected path '{corrected_path}', retrying read.")
                    result['details']['corrected_path_used'] = corrected_path
                    attempt2_result = _perform_read(corrected_path); result.update(attempt2_result)
                    result['message'] += f" (used corrected path '{PurePosixPath(corrected_path).name}')"
                else: result.update(attempt1_result) # Use original failure if no match
            except Exception as lookup_e:
                logger.error(f"Error during case-insensitive lookup for read: {lookup_e}", exc_info=True)
                result.update(attempt1_result); result['message'] += " (Case-insensitive lookup failed)"; result['reason'] = 'internal_error'
        else: result.update(attempt1_result) # Use result from first attempt

        # Ensure original path details are kept
        result['details']['path'] = path
        result['details']['absolute_path'] = original_abs_path
        return result

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """ Writes content to a file, attempting case-insensitive path resolution first. """
        logger.info(f"Seed VMService: Writing to file '{path}' (Content length: {len(content)})")
        result = {'success': False, 'message': '', 'details': {'path': path, 'content_length': len(content)}, 'reason': ''}
        resolved_path = self._resolve_path(path)
        if not resolved_path: result['message'] = "Invalid path resolution."; result['reason'] = 'invalid_path'; return result
        result['details']['absolute_path'] = resolved_path
        write_target_path = resolved_path

        try: # Check for existing case-insensitive match BEFORE writing
            parent_path = str(PurePosixPath(resolved_path).parent); filename = PurePosixPath(resolved_path).name; corrected_path = self._find_case_insensitive_match(parent_path, filename)
            if corrected_path: logger.info(f"Found existing case-insensitive match '{corrected_path}'. Writing will target this path."); write_target_path = corrected_path; result['details']['corrected_path_used'] = corrected_path
            elif len(self._get_potential_matches(parent_path, filename)) > 1: result['message'] = f"Write failed: Ambiguous path, multiple case-insensitive matches for '{filename}' in '{parent_path}'."; result['reason'] = 'ambiguous_path'; return result
        except Exception as lookup_e: logger.error(f"Error during case-insensitive lookup for write: {lookup_e}", exc_info=True); result['message'] = f"Write failed: Error during path lookup: {lookup_e}"; result['reason'] = 'internal_error'; return result
        result['details']['final_write_path'] = write_target_path

        # Perform the write using write_target_path
        if self.use_real_system:
            if 'sh' not in self.allowed_real_commands or 'printf' not in self.allowed_real_commands: result['message'] = "Cannot write file: 'sh' or 'printf' command not allowed."; result['reason'] = 'safety_violation'; return result
            write_cmd = f"printf %s {shlex.quote(content)} > {shlex.quote(write_target_path)}"; exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context; write_res = self._execute_real_command(write_cmd, exec_func)
            result['details']['exit_code'] = write_res.get('exit_code', -1); result['details']['stderr'] = write_res.get('stderr'); result['reason'] = write_res.get('reason', 'execution_error')
            if write_res.get('success'):
                result['success'] = True; result['message'] = "File written successfully."
                if write_target_path != resolved_path: result['message'] += f" (Used existing path '{PurePosixPath(write_target_path).name}')"
            else:
                result['message'] = f"Failed to write file: {write_res.get('stderr', 'Unknown error')}"; stderr_lower = (write_res.get('stderr') or '').lower()
                if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found'
                elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
                elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state: result['message'] = "Simulation not initialized."; result['reason'] = 'internal_error'; return result
            fs = self._simulated_state['filesystem']; item_info = fs.get(write_target_path)
            if item_info and item_info.get('type') == 'directory': result['message'] = "Cannot write: Is a directory."; result['reason'] = 'is_directory'
            else: # Permission check bypassed
                fs[write_target_path] = {'type': 'file', 'content': content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(content.encode())}; result['success'] = True; result['message'] = "File written successfully (simulation)."
                if write_target_path != resolved_path: result['message'] += f" (Used existing path '{PurePosixPath(write_target_path).name}')"
        return result
    # --- End Filesystem Operations ---

    # --- State Retrieval ---
    def get_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        """ Retrieves state snapshot from simulation OR real system. """
        # This method remains unchanged from the block you provided
        if self.use_real_system:
             return self._get_real_system_state(target_path_hint)
        else: # Simulation Mode
            if self._simulated_state:
                state_copy = copy.deepcopy(self._simulated_state)
                state_copy['target_path_hint'] = target_path_hint
                state_copy['mode'] = 'simulation'
                state_copy['cwd'] = self._get_current_cwd()
                # Simulate probing the target path hint
                if target_path_hint:
                    sim_fs = state_copy['filesystem']
                    abs_hint_path = self._resolve_path(target_path_hint) # Use fixed resolver
                    if abs_hint_path:
                        if abs_hint_path not in sim_fs:
                            # Add an entry indicating it doesn't exist
                             sim_fs[abs_hint_path] = {'type': None, 'exists': False, 'error': 'No such file or directory (Simulated)'}
                        else:
                             # Mark existing entry with 'exists: True' for clarity if needed by SensoryRefiner
                             sim_fs[abs_hint_path]['exists'] = True
                    else:
                         # Add an entry indicating invalid path if resolution failed
                         sim_fs[target_path_hint] = {'type': None, 'exists': False, 'error': 'Invalid path resolution (Simulated)'}

                return state_copy
            else:
                logger.error("Cannot get state: Simulation state not initialized.")
                return {"error": "Simulation state not initialized."}

    def _get_real_system_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        """ Probes the real system (Docker/Subprocess) for state information. """
        # This method remains unchanged from the block you provided
        current_cwd = self._get_current_cwd(); logger.debug(f"Probing real system state (Mode: {'Docker' if self.docker_container else 'Subprocess'}, CWD: {current_cwd})...")
        state: Dict[str, Any] = {'timestamp': time.time(), 'filesystem': {}, 'resources': {}, 'mode': 'docker' if self.docker_container else 'subprocess', 'cwd': current_cwd, 'target_path_hint': target_path_hint, 'probe_errors': []}
        probe_results = {}; exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
        probes = {'cpu': "top -bn1 | grep '^%Cpu' | head -n1", 'mem': "grep -E 'MemTotal|MemAvailable' /proc/meminfo", 'disk_cwd': f"df -k {shlex.quote(current_cwd)}", 'ls_cwd': f"ls -lA --full-time {shlex.quote(current_cwd)}"}
        allowed_probes = {k: cmd for k, cmd in probes.items() if cmd.split()[0] in self.allowed_real_commands}
        abs_target_hint: Optional[str] = None
        if target_path_hint:
            abs_target_hint = self._resolve_path(target_path_hint)
            if abs_target_hint:
                 if 'stat' in self.allowed_real_commands: allowed_probes['stat_target'] = f'stat {shlex.quote(abs_target_hint)}'
                 else: state['probe_errors'].append("'stat' command not allowed..."); state['filesystem'][target_path_hint] = {'type':None,'exists':None,'error':'Stat command disabled'}
            else: state['probe_errors'].append(f"Invalid target path hint: {target_path_hint}"); state['filesystem'][target_path_hint] = {'type':None,'exists':False,'error':'Invalid path resolution (Real Mode)'}
        for key, cmd in allowed_probes.items():
            res = self._execute_real_command(cmd, exec_func); probe_results[key] = res
            if not res.get('success'): state['probe_errors'].append(f"Probe '{key}' failed (Code {res.get('exit_code','?')})"); logger.warning(f"State probe '{key}' cmd failed: {cmd} -> {res.get('stderr', 'No stderr')}")
        try: # Parse Probe Results
            cpu_res = probe_results.get('cpu');
            idle_match = re.search(r"ni,\s*([\d\.]+)\s*id,", cpu_res['stdout']) if cpu_res and cpu_res.get('success') else None;
            if idle_match:
                try:
                    idle_perc = float(idle_match.group(1))
                    state['resources']['cpu_load_percent'] = round(100.0 - idle_perc, 1)
                except ValueError:
                    pass # Ignore if conversion fails
            mem_res = probe_results.get('mem'); total_kb=None; avail_kb=None; free_kb=None;
            if mem_res and mem_res.get('success'): total_kb = re.search(r"MemTotal:\s+(\d+)\s*kB", mem_res['stdout']); avail_kb = re.search(r"MemAvailable:\s+(\d+)\s*kB", mem_res['stdout']); free_kb = re.search(r"MemFree:\s+(\d+)\s*kB", mem_res['stdout'])
            if total_kb and avail_kb:
                try:
                    total = int(total_kb.group(1))
                    avail = int(avail_kb.group(1))
                    state['resources']['memory_usage_percent'] = round(((total - avail) / total) * 100.0, 1) if total > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    pass # Ignore calculation errors
            elif total_kb and free_kb:
                try:
                    total = int(total_kb.group(1))
                    free = int(free_kb.group(1))
                    state['resources']['memory_usage_percent'] = round(((total - free) / total) * 100.0, 1) if total > 0 else 0.0
                except (ValueError, ZeroDivisionError):
                    pass # Ignore calculation errors
            disk_res = probe_results.get('disk_cwd'); lines = disk_res['stdout'].strip().split('\n') if disk_res and disk_res.get('success') else [];
            match = None # Ensure match is defined before the if block
            if len(lines) > 1: match = re.search(r'\s+(\d+)%\s+(?:/[^\s]*)?$', lines[-1]);
            # <<< FIX: Corrected SyntaxError Here (was an indentation issue in your provided block) >>>
            if match: # This 'if' was previously misaligned in the thought process snippet
                try:
                    state['resources']['disk_usage_percent'] = float(match.group(1))
                except ValueError:
                    pass # Ignore conversion errors
            # <<< END FIX >>>
            state['filesystem'][current_cwd]={'type':'directory','content_listing':None,'error':None, 'exists': True}; ls_res = probe_results.get('ls_cwd')
            if ls_res: state['filesystem'][current_cwd]['content_listing'] = ls_res['stdout'] if ls_res.get('success') else None; state['filesystem'][current_cwd]['error'] = None if ls_res.get('success') else ls_res.get('stderr', 'ls failed')
            stat_res = probe_results.get('stat_target');
            if stat_res and abs_target_hint:
                stat_entry = {'type':None,'exists':False,'error':None, 'size_bytes': None, 'mtime': None, 'perms_octal': None, 'perms_symbolic': None, 'owner': None, 'stat_output': None}; stat_out = stat_res['stdout'] if stat_res.get('success') else None
                if stat_out:
                    stat_entry['exists'] = True; stat_entry['stat_output'] = stat_out;
                    if 'directory' in stat_out: stat_entry['type'] = 'directory'
                    elif 'regular empty file' in stat_out: stat_entry['type'] = 'file'; stat_entry['size_bytes'] = 0
                    elif 'regular file' in stat_out: stat_entry['type'] = 'file'
                    else: stat_entry['type'] = 'other'
                    size_match = re.search(r"Size:\s*(\d+)", stat_out); mtime_match = re.search(r"Modify:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)", stat_out); perms_match = re.search(r"Access:\s*\((\d+)/([a-zA-Z-]+)\)", stat_out); owner_match = re.search(r"Uid:\s*\(\s*\d+/\s*([\w-]+)\)", stat_out)
                    if size_match: stat_entry['size_bytes'] = int(size_match.group(1));
                    if mtime_match: stat_entry['mtime'] = mtime_match.group(1)
                    if perms_match: stat_entry['perms_octal'] = perms_match.group(1); stat_entry['perms_symbolic'] = perms_match.group(2)
                    if owner_match: stat_entry['owner'] = owner_match.group(1)
                else: stat_entry['exists'] = False; stat_entry['error'] = stat_res.get('stderr', 'Stat failed')
                state['filesystem'][abs_target_hint] = stat_entry
        except Exception as parse_err: logger.error(f"Error parsing real system state: {parse_err}", exc_info=True); state['parsing_error'] = f"State parsing failed: {parse_err}"
        if not state.get('probe_errors'): state.pop('probe_errors', None)
        return state
    # --- End State Retrieval ---

    def disconnect(self):
        """ Closes Docker client connection if open. """
        # This method remains unchanged from the block you provided
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