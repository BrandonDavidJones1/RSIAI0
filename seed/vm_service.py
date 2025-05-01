# --- START OF FILE seed/vm_service.py ---

# RSIAI/seed/vm_service.py
"""
Defines the Seed_VMService class for simulating or interacting with
an external system (VM, container, OS). Executes commands, reads/writes files,
and probes state to provide snapshots for Seed core reasoning.
Uses configured command whitelist and timeouts for safety in real mode.
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
from typing import Dict, Any, Optional, Tuple, Callable, List

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
        """ Sets up the initial simulated environment state. """
        logger.info("Setting up simulated filesystem and state...")
        self._simulated_system_cwd = '/app'
        # Simplified base commands for simulation mode (core file ops)
        self._sim_available_commands = ['ls', 'pwd', 'cat', 'echo', 'touch', 'mkdir', 'rm', 'cd', 'cp', 'mv']
        # Renamed internal var
        self._simulated_state = {
            'timestamp': time.time(), 'cwd': self._simulated_system_cwd,
            'filesystem': { # Basic FS - ** UPDATED TO INCLUDE /app/seed **
                '/': {'type': 'directory', 'owner': 'root', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/tmp': {'type': 'directory', 'owner': 'user', 'perms': 'rwxrwxrwx', 'mtime': time.time(), 'size_bytes': 4096},
                # Add the missing seed directory and files
                '/app/seed': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/seed/__init__.py': {'type': 'file', 'content': '# Seed package init\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 20},
                '/app/seed/core.py': {'type': 'file', 'content': '# RSIAI/seed/core.py\nprint("Simulated core.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 60},
                '/app/seed/memory_system.py': {'type': 'file', 'content': '# RSIAI/seed/memory_system.py\nprint("Simulated memory_system.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 70},
                '/app/seed/config.py': {'type': 'file', 'content': '# RSIAI/seed/config.py\nprint("Simulated config.py content")\n', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 64},
                # Add other essential files here if needed for testing
                # '/app/placeholder.txt': {'type': 'file', 'content': 'Initial content.', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 16}, # Removed old placeholder
            },
            'resources': {'cpu_load_percent': 1.0, 'memory_usage_percent': 5.0, 'disk_usage_percent': 10.0}, # Lower defaults
            'last_command_result': None,
            'available_commands': list(self._sim_available_commands) # Use simplified list
        }
        logger.info(f"Simulation Initialized. CWD: {self._simulated_system_cwd}. Sim commands: {self._sim_available_commands}")

    def _get_current_cwd(self) -> str:
        return self._real_system_cwd if self.use_real_system else self._simulated_system_cwd

    def _set_current_cwd(self, new_cwd: str):
        """ Sets the internal current working directory, ensuring it's absolute. """
        try:
            # Use PurePosixPath for internal representation, even if host OS is Windows
            # This ensures consistency when interacting with Docker or simulations.
            posix_path = PurePosixPath(new_cwd)
            if not posix_path.is_absolute():
                 # If not absolute, resolve it relative to the *current* CWD
                 current_path = PurePosixPath(self._get_current_cwd())
                 posix_path = (current_path / posix_path).resolve()
                 logger.debug(f"Resolved relative CWD '{new_cwd}' to '{posix_path}'")

            normalized_cwd = str(posix_path) # Use resolved path

            if self.use_real_system:
                self._real_system_cwd = normalized_cwd
            else:
                self._simulated_system_cwd = normalized_cwd
        except (TypeError, ValueError) as e:
            logger.error(f"Attempted to set invalid CWD '{new_cwd}': {e}")

    # --- Path/Permission Helpers ---
    def _resolve_path(self, path_str: str) -> Optional[str]:
        """ Resolves a given path string relative to the current CWD. Returns absolute POSIX path string or None. """
        current_dir = self._get_current_cwd()
        if not path_str: return current_dir
        try:
            cwd_path = PurePosixPath(current_dir)
            # Handle ~ home directory expansion (assuming /app as home for simplicity)
            if path_str == '~' or path_str.startswith('~/'):
                home_dir = '/app' # Define a consistent "home" for resolution
                path_part = path_str[2:] if path_str.startswith('~/') else ''
                target_path = (PurePosixPath(home_dir) / path_part)
            else:
                input_path = PurePosixPath(path_str)
                target_path = (cwd_path / input_path) if not input_path.is_absolute() else input_path

            # Use resolve() to handle '..' and '.' components
            # Note: resolve() might fail if intermediate components don't exist in real FS,
            # but it's generally safe for string manipulation with PurePosixPath.
            resolved_path = target_path.resolve()
            resolved_path_str = str(resolved_path)

            # Final check if it's absolute (should always be after resolve)
            if not resolved_path.is_absolute():
                logger.warning(f"Path resolution resulted in non-absolute path '{resolved_path_str}' for input '{path_str}'. Using original target: {target_path}")
                return str(target_path)

            return resolved_path_str
        except Exception as e:
            logger.warning(f"Path resolution error for '{path_str}' from '{current_dir}': {e}")
            return None

    def _sim_check_permissions(self, path: str, action: str = 'read') -> Tuple[bool, str]:
        """ Checks permissions for an action on a path in the simulated filesystem. """
        if not self._simulated_state: return False, "Simulation state not initialized"
        fs = self._simulated_state['filesystem']
        norm_path = str(PurePosixPath(path)) # Ensure normalized path for lookup
        parent = str(PurePosixPath(norm_path).parent) if norm_path != '/' else None
        parent_info = fs.get(parent) if parent else None
        item_info = fs.get(norm_path)

        # Check parent permissions for write/create/delete actions
        if action in ['write', 'delete', 'create']:
            if norm_path == '/': return False, "Permission denied: cannot modify root directory"
            if not parent: return False, "Internal error: Cannot determine parent for non-root path."
            if not parent_info: return False, f"Parent directory '{parent}' does not exist"
            if parent_info.get('type') != 'directory': return False, f"Parent '{parent}' is not a directory"
            # Simplified permissions: Allow write if parent owner is 'user' or if parent is /tmp
            if not (parent_info.get('owner') == 'user' or parent == '/tmp'):
                 return False, f"Permission denied writing to parent '{parent}'"

        # Check item permissions/existence
        if item_info:
            owner = item_info.get('owner', 'system')
            perms = item_info.get('perms', '---------') # e.g., 'rw-r--r--'
            # Simplified: Assume 'user' owns files it creates.
            # Read allowed if user owns and has 'r', or if other has 'r'
            can_read = (owner == 'user' and len(perms) > 1 and perms[1] == 'r') or \
                       (len(perms) > 7 and perms[7] == 'r')
            # Write allowed if user owns and has 'w'
            can_write = (owner == 'user' and len(perms) > 2 and perms[2] == 'w')

            if action == 'read': return (True, "") if can_read else (False, "Permission denied (read)")
            if action == 'write': return (True, "") if can_write else (False, "Permission denied (write)")
            if action == 'delete': return (True, "") if can_write else (False, "Permission denied (delete)") # Use write perm for delete simplicity

        elif action == 'read': return False, "No such file or directory"
        elif action == 'create': return True, "" # Parent check already done
        elif action in ['write', 'delete']: return False, "No such file or directory"

        # Fallback
        return False, f"Permission check fallback denied for action '{action}' on '{path}'"
    # --- End Path/Permission Helpers ---

    # --- Command Execution ---
    def execute_command(self, command_str: str) -> Dict[str, Any]:
        """ Executes command in simulation OR real system, handling CWD internally. """
        if self.use_real_system:
            exec_context = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            return self._execute_real_command(command_str, exec_context)
        else:
            return self._execute_simulated_command(command_str)

    def _execute_simulated_command(self, command_str: str) -> Dict[str, Any]:
        """ Executes command within the simulated filesystem and updates internal state. """
        current_cwd = self._get_current_cwd()
        logger.debug(f"VM Sim Exec (CWD: {current_cwd}): '{command_str}'")
        result={'success':False,'stdout':'','stderr':'','command':command_str,'reason':'','exit_code':1}
        if not self._simulated_state:
            result['stderr'],result['reason']="Sim not init.",'internal_error'
            return result
        try:
            parts = shlex.split(command_str)
        except ValueError as e:
            result['stderr'],result['reason']=f"Command parse error: {e}",'parse_error'
            return result

        cmd=parts[0] if parts else ''
        args=parts[1:]

        # Use internal list of allowed sim commands
        if not cmd or cmd not in self._sim_available_commands:
            result['stderr'],result['reason']=f"Command not found/allowed in sim: {cmd}",'illegal_command'
            return result

        try:
            fs = self._simulated_state['filesystem']
            resolve_path_func = self._resolve_path # Alias for readability

            # --- Simulated Command Logic ---
            if cmd == 'pwd':
                result['stdout'], result['success'], result['exit_code'] = current_cwd, True, 0
            elif cmd == 'cd':
                target_dir_str = args[0] if args else '~' # Default to home if no arg
                resolved_path = resolve_path_func(target_dir_str)
                if resolved_path and resolved_path in fs and fs[resolved_path].get('type') == 'directory':
                    # Check permissions only if it's a valid directory
                    allowed, msg = self._sim_check_permissions(resolved_path, 'read') # Can we 'enter' it?
                    if allowed:
                        self._set_current_cwd(resolved_path)
                        self._simulated_state['cwd'] = self._get_current_cwd() # Update state's CWD too
                        result['success'], result['exit_code'] = True, 0
                    else:
                        result['stderr'], result['reason'] = msg, 'permission_denied'
                elif resolved_path and resolved_path not in fs:
                    result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found'
                elif resolved_path: # Exists but not a directory
                    result['stderr'], result['reason'] = f"cd: Not a directory: {target_dir_str}", 'is_not_directory'
                else: # Invalid path resolution
                    result['stderr'], result['reason'] = f"cd: Invalid path: {target_dir_str}", 'invalid_path'

            elif cmd == 'ls':
                target_path_str = args[0] if args else '.' # Default to current dir
                resolved_path = resolve_path_func(target_path_str)
                if resolved_path and resolved_path in fs:
                    item_info = fs[resolved_path]
                    allowed, msg = self._sim_check_permissions(resolved_path, 'read')
                    if not allowed:
                        result['stderr'], result['reason'] = msg, 'permission_denied'
                    elif item_info.get('type') == 'directory':
                        # Find items whose parent is the resolved path
                        contents = [PurePosixPath(n).name for n, f in fs.items() if str(PurePosixPath(n).parent) == resolved_path and n != resolved_path]
                        result['stdout'], result['success'], result['exit_code'] = "\n".join(sorted(contents)), True, 0
                    else: # It's a file, just list its name
                        result['stdout'], result['success'], result['exit_code'] = PurePosixPath(resolved_path).name, True, 0
                else:
                    result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found'

            elif cmd == 'cat':
                if not args:
                    result['stderr'], result['reason'] = "cat: missing file operand", 'missing_args'
                else:
                    p = resolve_path_func(args[0])
                    if p and p in fs:
                        info = fs[p]
                        allowed, msg = self._sim_check_permissions(p, 'read')
                        if not allowed:
                             result['stderr'], result['reason'] = msg, 'permission_denied'
                        elif info.get('type') == 'file':
                             result['stdout'], result['success'], result['exit_code'] = info.get('content',''), True, 0
                        else:
                             result['stderr'], result['reason'] = f"cat: {args[0]}: Is a directory", 'is_directory'
                    else:
                        result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found'

            elif cmd == 'echo':
                content_to_echo = ""; target_file = None; redirect_mode=None; append=False
                # Simple parsing for > or >> redirection
                if len(args) >= 3 and args[-2] in ['>', '>>']:
                    redirect_mode = args[-2]
                    append = (redirect_mode == '>>')
                    target_file = args[-1]
                    content_to_echo = " ".join(args[:-2]).strip("'\"")
                else:
                    content_to_echo = " ".join(args).strip("'\"")

                if redirect_mode and target_file:
                    p = resolve_path_func(target_file)
                    if p:
                        allowed, msg = self._sim_check_permissions(p, 'write') if p in fs else self._sim_check_permissions(p, 'create')
                        if allowed:
                            existing_content = fs.get(p,{}).get('content','') if append and p in fs and fs[p].get('type')=='file' else ''
                            # Ensure newline separation if appending
                            separator = '\n' if existing_content and not existing_content.endswith('\n') else ''
                            new_content = existing_content + separator + content_to_echo # Echo usually adds newline implicitly in shell, simulate that? Maybe not here.
                            fs[p] = {'type': 'file', 'content': new_content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(new_content.encode())}
                            result['success'], result['exit_code'] = True, 0
                        else:
                            result['stderr'], result['reason'] = msg, 'permission_denied'
                    else:
                        result['stderr'], result['reason'] = f"echo: Invalid path: {target_file}", 'invalid_path'
                elif redirect_mode and not target_file:
                     result['stderr'], result['reason'] = f"echo: missing target for redirection '{redirect_mode}'", 'parse_error'
                else: # Echo to stdout
                    result['stdout'], result['success'], result['exit_code'] = content_to_echo, True, 0

            elif cmd == 'touch':
                if not args:
                    result['stderr'], result['reason'] = "touch: missing file operand", 'missing_args'
                else:
                    p = resolve_path_func(args[0])
                    if p:
                        allowed, msg = self._sim_check_permissions(p, 'write') if p in fs else self._sim_check_permissions(p, 'create')
                        if allowed:
                            if p in fs and fs[p].get('type') == 'file':
                                fs[p]['mtime'] = time.time() # Update timestamp
                            elif p in fs and fs[p].get('type') == 'directory':
                                fs[p]['mtime'] = time.time() # Can touch directories to update time
                            else: # Create empty file
                                fs[p] = {'type': 'file', 'content': '', 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0}
                            result['success'], result['exit_code'] = True, 0
                        else:
                            result['stderr'], result['reason'] = msg, 'permission_denied'
                    else:
                        result['stderr'], result['reason'] = f"touch: Invalid path: {args[0]}", 'invalid_path'

            elif cmd == 'mkdir':
                 if not args:
                     result['stderr'], result['reason'] = "mkdir: missing operand", 'missing_args'
                 else:
                     p = resolve_path_func(args[0])
                     if p:
                         if p in fs:
                             result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': File exists", 'file_exists'
                         else:
                             allowed, msg = self._sim_check_permissions(p, 'create')
                             if allowed:
                                 fs[p] = {'type': 'directory', 'owner': 'user', 'perms':'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096}
                                 result['success'], result['exit_code'] = True, 0
                             else:
                                 result['stderr'], result['reason'] = msg, 'permission_denied'
                     else:
                         result['stderr'], result['reason'] = f"mkdir: Invalid path: {args[0]}", 'invalid_path'

            elif cmd == 'rm':
                 if not args:
                     result['stderr'], result['reason'] = "rm: missing operand", 'missing_args'
                 else:
                     p = resolve_path_func(args[0])
                     if p and p in fs:
                         info = fs[p]
                         allowed, msg = self._sim_check_permissions(p, 'delete')
                         if not allowed:
                             result['stderr'], result['reason'] = msg, 'permission_denied'
                         elif info.get('type') == 'directory':
                             # Check if directory is empty (excluding itself)
                             is_empty = not any(str(PurePosixPath(n).parent)==p for n in fs if n != p)
                             if is_empty:
                                 del fs[p]
                                 result['success'], result['exit_code'] = True, 0
                             else:
                                 result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': Directory not empty", 'directory_not_empty'
                         else: # It's a file
                             del fs[p]
                             result['success'], result['exit_code'] = True, 0
                     elif p == '/':
                          result['stderr'], result['reason'] = "rm: cannot remove root directory", 'permission_denied'
                     else:
                          result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found'

            elif cmd == 'cp':
                 if len(args) != 2:
                     result['stderr'], result['reason'] = "cp: missing destination file operand", 'missing_args'
                 else:
                     src_p = resolve_path_func(args[0]); dest_p = resolve_path_func(args[1])
                     if not src_p or not dest_p:
                         result['stderr'], result['reason'] = "cp: Invalid path", 'invalid_path'
                     elif src_p == dest_p:
                         result['stderr'], result['reason'] = f"cp: '{args[0]}' and '{args[1]}' are the same file", 'invalid_argument'
                     elif src_p not in fs:
                         result['stderr'], result['reason'] = f"cp: cannot stat '{args[0]}': No such file or directory", 'file_not_found'
                     elif fs[src_p].get('type') == 'directory':
                         result['stderr'], result['reason'] = f"cp: omitting directory '{args[0]}'", 'is_directory' # Basic cp doesn't copy dirs
                     else:
                         src_allowed, src_msg = self._sim_check_permissions(src_p, 'read')
                         dest_allowed, dest_msg = self._sim_check_permissions(dest_p, 'write') if dest_p in fs else self._sim_check_permissions(dest_p, 'create')
                         if not src_allowed:
                             result['stderr'], result['reason'] = f"cp: {src_msg}", 'permission_denied'
                         elif not dest_allowed:
                              result['stderr'], result['reason'] = f"cp: {dest_msg}", 'permission_denied'
                         else:
                             fs[dest_p] = copy.deepcopy(fs[src_p]) # Copy data
                             fs[dest_p]['owner'] = 'user' # Assume new owner is user
                             fs[dest_p]['mtime'] = time.time()
                             result['success'], result['exit_code'] = True, 0

            elif cmd == 'mv':
                 if len(args) != 2:
                     result['stderr'], result['reason'] = "mv: missing destination file operand", 'missing_args'
                 else:
                     src_p = resolve_path_func(args[0]); dest_p = resolve_path_func(args[1])
                     if not src_p or not dest_p:
                         result['stderr'], result['reason'] = "mv: Invalid path", 'invalid_path'
                     elif src_p == dest_p:
                         result['stderr'], result['reason'] = f"mv: '{args[0]}' and '{args[1]}' are the same file", 'invalid_argument'
                     elif src_p not in fs:
                          result['stderr'], result['reason'] = f"mv: cannot stat '{args[0]}': No such file or directory", 'file_not_found'
                     else:
                         # Need delete permission on source's parent, write permission on dest parent
                         src_parent_allowed, src_parent_msg = self._sim_check_permissions(str(PurePosixPath(src_p).parent), 'write')
                         dest_allowed, dest_msg = self._sim_check_permissions(dest_p, 'write') if dest_p in fs else self._sim_check_permissions(dest_p, 'create')
                         if not src_parent_allowed:
                              result['stderr'], result['reason'] = f"mv: cannot move '{args[0]}' (check source parent permissions): {src_parent_msg}", 'permission_denied'
                         elif not dest_allowed:
                               result['stderr'], result['reason'] = f"mv: cannot move to '{args[1]}': {dest_msg}", 'permission_denied'
                         else:
                              fs[dest_p] = fs.pop(src_p) # Move the entry
                              fs[dest_p]['mtime'] = time.time() # Update timestamp
                              result['success'], result['exit_code'] = True, 0

            else:
                result['stderr'], result['reason'] = f"Sim command '{cmd}' logic not implemented.", 'not_implemented'

        except Exception as e:
            logger.error(f"Sim Internal Error exec '{command_str}': {e}",exc_info=True)
            result.update({'stderr':f"Internal sim error: {e}",'reason':'internal_error'})

        # Update simulation state timestamp and last command result
        self._simulated_state['timestamp']=time.time()
        self._simulated_state['last_command_result']=copy.deepcopy(result)
        logger.debug(f"VM Sim Res: Succ={result['success']}, Code={result['exit_code']}, Out='{result['stdout'][:30]}...', Err='{result['stderr'][:30]}...'")
        return result

    def _execute_real_command(self, command_str: str, execution_context: Callable[[str], Dict]) -> Dict[str, Any]:
        """ Prepares and executes a command in the real system context (Docker/Subprocess), handling CWD. """
        result: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'command': command_str, 'reason': '', 'exit_code': -1}
        try:
            parts = shlex.split(command_str)
        except ValueError as e:
            result['stderr'], result['reason'] = f"Command parse error: {e}", 'parse_error'
            return result

        command = parts[0] if parts else ''

        # --- Whitelist Check ---
        if command not in self.allowed_real_commands:
            result['stderr'], result['reason'] = f"Command '{command}' not allowed for real execution.", 'safety_violation'
            logger.warning(f"Safety Violation: Blocked real system execution: '{command_str}'")
            return result

        try:
             # --- Handle 'cd' Internally ---
             if command == 'cd':
                 if len(parts) == 1: target_dir_str = '~' # cd with no args goes home
                 elif len(parts) == 2: target_dir_str = parts[1]
                 else: result['stderr'], result['reason'] = "cd: too many arguments", 'invalid_argument'; return result

                 # Resolve target directory path string
                 if target_dir_str == '~':
                     # Determine home directory (use pwd in home context if possible, else default)
                     home_res = execution_context("cd ~ && pwd")
                     if home_res.get('success') and home_res.get('stdout','').strip():
                         target_dir = home_res['stdout'].strip()
                     else:
                         # Fallback if getting home dir failed
                         logger.warning(f"Could not determine home directory via 'cd ~ && pwd' (Stderr: {home_res.get('stderr','')}). Assuming '/app'.")
                         target_dir = '/app'
                 else:
                     # Resolve potentially relative path
                     target_dir = self._resolve_path(target_dir_str)

                 if not target_dir:
                     result['stderr'], result['reason'] = f"cd: Invalid path: {target_dir_str}", 'invalid_path'
                     return result

                 # Verify target exists and is a directory using 'ls -ld' before changing internal CWD
                 check_cmd = f"ls -ld {shlex.quote(target_dir)}"
                 check_res = execution_context(check_cmd)

                 if check_res.get('success') and check_res.get('stdout','').strip().startswith('d'):
                     # Target is a directory, update internal CWD
                     self._set_current_cwd(target_dir)
                     result['success'], result['exit_code'] = True, 0
                     logger.info(f"Real system CWD updated to: {self._get_current_cwd()}")
                 elif check_res.get('exit_code') != 0:
                     # ls failed, check stderr for common reasons
                     stderr = check_res.get('stderr', '').lower()
                     reason = 'execution_error' # Default reason
                     if 'no such file or directory' in stderr: reason = 'file_not_found'
                     elif 'permission denied' in stderr: reason = 'permission_denied'
                     result.update({'success':False, 'exit_code':check_res.get('exit_code', 1), 'stderr':f"cd: cannot access '{target_dir_str}': {check_res.get('stderr','Unknown error')}", 'reason':reason})
                 else:
                     # ls succeeded but didn't start with 'd', so not a directory
                     result.update({'success':False, 'exit_code':1, 'stderr':f"cd: not a directory: {target_dir_str}", 'reason':'is_not_directory'})
                 return result

             # --- Execute other commands directly ---
             exec_res_dict = execution_context(command_str)
             return exec_res_dict

        except Exception as e:
            logger.error(f"Unexpected error during real command prep/exec '{command_str}': {e}", exc_info=True)
            result['stderr'], result['reason'] = f"Unexpected internal error: {e}", 'internal_error'
            return result

    def _docker_exec_context(self, command_str: str) -> Dict[str, Any]:
        """ Executes a command inside the configured Docker container. """
        res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
        if not self.docker_container:
            res['stderr']='Docker container unavailable'
            res['reason']='docker_error'
            return res
        try:
            # Use sh -c to handle complex commands, pipes, etc. safely via quoting
            full_cmd = f"sh -c {shlex.quote(command_str)}"
            logger.debug(f"Docker Exec Run: cmd='{full_cmd}', workdir='{self._get_current_cwd()}'")
            # Ensure workdir exists in container? Docker exec might handle this.
            exit_code, output = self.docker_container.exec_run(
                cmd=full_cmd,
                workdir=self._get_current_cwd(), # Execute in the tracked CWD
                stream=False,
                demux=False,
                user='root' # Or configure a specific user if needed
            )

            output_bytes: bytes = output if isinstance(output, bytes) else b''
            output_str = output_bytes.decode(errors='replace').strip()

            res['exit_code'] = exit_code
            res['success'] = (exit_code == 0)
            # Assign output to stdout/stderr based on success
            res['stdout'] = output_str if res['success'] else ''
            res['stderr'] = '' if res['success'] else output_str

            if not res['success']:
                res['reason'] = 'execution_error' # Generic reason for non-zero exit
                if not res['stderr']: # Provide stderr if command failed silently
                     res['stderr'] = f"Command failed (Code {res['exit_code']}) with no output."
            return res

        except DockerAPIError as api_err:
            logger.error(f"Docker API error executing '{command_str}': {api_err}", exc_info=True)
            res.update({'stderr':f"Docker API error: {api_err}", 'reason':'docker_api_error'})
            return res
        except Exception as e:
            logger.error(f"Docker exec_run unexpected error '{command_str}': {e}", exc_info=True)
            res.update({'stderr':f"Docker exec_run error: {e}", 'reason':'docker_error'})
            return res

    def _subprocess_exec_context(self, command_str: str) -> Dict[str, Any]:
         """ Executes a command using the local subprocess module. """
         res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1, 'command': command_str}
         current_cwd = self._get_current_cwd()
         logger.debug(f"Subprocess Exec Run: cmd='{command_str}', cwd='{current_cwd}'")
         try:
            # Use shell=True cautiously, relies on whitelist for safety.
            # Pass explicit CWD.
            proc = subprocess.run(
                command_str,
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=self.command_timeout_sec,
                check=False, # Don't raise exception on non-zero exit code
                cwd=current_cwd # Execute in the tracked CWD
            )
            res['exit_code'] = proc.returncode
            res['stdout'] = proc.stdout.strip() if proc.stdout else ''
            res['stderr'] = proc.stderr.strip() if proc.stderr else ''
            res['success'] = (proc.returncode == 0)

            if not res['success']:
                res['reason'] = 'execution_error'
                if not res['stderr']: # Provide stderr if command failed silently
                     res['stderr'] = f"Command failed (Code {proc.returncode}) with no stderr."
            return res
         except FileNotFoundError:
            # This can happen if the shell itself or the command isn't found
            res['stderr'], res['reason'] = f"Command or shell not found: {command_str.split()[0]}", 'command_not_found'
            res['exit_code'] = 127
            return res
         except subprocess.TimeoutExpired:
            res['stderr'], res['reason'] = f"Timeout ({self.command_timeout_sec}s)", 'timeout'
            res['exit_code'] = -9 # Standard timeout signal code
            return res
         except Exception as e:
            logger.error(f"Subprocess error executing '{command_str}': {e}", exc_info=True)
            res['stderr']=f"Subprocess exec error: {e}"
            res['reason']='internal_error'
            return res
    # --- End Command Execution ---

    # --- Filesystem Operations ---
    def read_file(self, path: str) -> Dict[str, Any]:
        """ Reads file content from simulation OR real system. """
        logger.info(f"Seed VMService: Reading file '{path}'")
        result = {'success': False, 'content': None, 'message': '', 'details': {'path': path}, 'reason': ''}
        abs_path = self._resolve_path(path)
        if not abs_path:
            result['message'] = "Invalid path provided."
            result['reason'] = 'invalid_path'
            return result
        result['details']['absolute_path'] = abs_path

        if self.use_real_system:
            if 'cat' not in self.allowed_real_commands:
                result['message'] = "Cannot read file: 'cat' command not allowed."
                result['reason'] = 'safety_violation'
                return result
            # Determine execution context
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            # Execute cat command
            cat_res = self._execute_real_command(f"cat {shlex.quote(abs_path)}", exec_func)
            result['details']['exit_code'] = cat_res.get('exit_code', -1)
            result['details']['stderr'] = cat_res.get('stderr')
            result['reason'] = cat_res.get('reason', 'execution_error')

            if cat_res.get('success'):
                result['success'] = True
                result['content'] = cat_res.get('stdout', '')
                result['message'] = "File read successfully."
            else:
                result['message'] = f"Failed to read file: {cat_res.get('stderr', 'Unknown error')}"
                stderr_lower = (cat_res.get('stderr') or '').lower()
                if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found'
                elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
                elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state:
                result['message'] = "Simulation not initialized."
                result['reason'] = 'internal_error'
                return result
            fs = self._simulated_state['filesystem']
            if abs_path in fs:
                item_info = fs[abs_path]
                allowed, msg = self._sim_check_permissions(abs_path, 'read')
                if not allowed:
                    result['message'] = msg
                    result['reason'] = 'permission_denied'
                elif item_info.get('type') == 'file':
                    result['success'] = True
                    result['content'] = item_info.get('content', '')
                    result['message'] = "File read successfully (simulation)."
                elif item_info.get('type') == 'directory':
                    result['message'] = "Cannot read: Is a directory."
                    result['reason'] = 'is_directory'
                else:
                    result['message'] = f"Cannot read: Not a file (Type: {item_info.get('type')})."
                    result['reason'] = 'invalid_type'
            else:
                result['message'] = "File not found (simulation)."
                result['reason'] = 'file_not_found'
        return result

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """ Writes content to a file in simulation OR real system. """
        logger.info(f"Seed VMService: Writing to file '{path}' (Content length: {len(content)})")
        result = {'success': False, 'message': '', 'details': {'path': path, 'content_length': len(content)}, 'reason': ''}
        abs_path = self._resolve_path(path)
        if not abs_path:
            result['message'] = "Invalid path provided."
            result['reason'] = 'invalid_path'
            return result
        result['details']['absolute_path'] = abs_path

        if self.use_real_system:
            # Using printf is generally safer than echo for arbitrary content
            if 'sh' not in self.allowed_real_commands or 'printf' not in self.allowed_real_commands:
                result['message'] = "Cannot write file: 'sh' or 'printf' command not allowed."
                result['reason'] = 'safety_violation'
                return result
            # Ensure content is properly quoted for the shell command
            write_cmd = f"printf %s {shlex.quote(content)} > {shlex.quote(abs_path)}"
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            write_res = self._execute_real_command(write_cmd, exec_func)
            result['details']['exit_code'] = write_res.get('exit_code', -1)
            result['details']['stderr'] = write_res.get('stderr')
            result['reason'] = write_res.get('reason', 'execution_error')

            if write_res.get('success'):
                result['success'] = True
                result['message'] = "File written successfully."
            else:
                result['message'] = f"Failed to write file: {write_res.get('stderr', 'Unknown error')}"
                stderr_lower = (write_res.get('stderr') or '').lower()
                if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found' # Parent dir likely missing
                elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
                elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state:
                result['message'] = "Simulation not initialized."
                result['reason'] = 'internal_error'
                return result
            fs = self._simulated_state['filesystem']
            # Check write permission on file if exists, or create permission on parent if not
            allowed, msg = self._sim_check_permissions(abs_path, 'write') if abs_path in fs else self._sim_check_permissions(abs_path, 'create')
            if allowed:
                item_info = fs.get(abs_path)
                # Prevent overwriting a directory
                if item_info and item_info.get('type') == 'directory':
                    result['message'] = "Cannot write: Is a directory."
                    result['reason'] = 'is_directory'
                else:
                    fs[abs_path] = {'type': 'file', 'content': content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(content.encode())}
                    result['success'] = True
                    result['message'] = "File written successfully (simulation)."
            else:
                result['message'] = msg
                result['reason'] = 'permission_denied'
        return result
    # --- End Filesystem Operations ---

    # --- State Retrieval ---
    def get_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        """ Retrieves state snapshot from simulation OR real system. """
        if self.use_real_system:
             return self._get_real_system_state(target_path_hint)
        else: # Simulation Mode
            if self._simulated_state:
                state_copy = copy.deepcopy(self._simulated_state)
                state_copy['target_path_hint'] = target_path_hint
                state_copy['mode'] = 'simulation'
                state_copy['cwd'] = self._get_current_cwd()
                # Add simulated target stat info if hint provided
                if target_path_hint:
                    sim_fs = state_copy['filesystem']
                    abs_hint_path = self._resolve_path(target_path_hint)
                    if abs_hint_path:
                        # Ensure the entry exists for the hint path, marking if it doesn't
                        if abs_hint_path not in sim_fs:
                            # Add placeholder info indicating non-existence
                            sim_fs[abs_hint_path] = {'type': None, 'exists': False, 'error': 'No such file or directory (Simulated)'}
                        else:
                             # Ensure 'exists' flag is present and correct
                             sim_fs[abs_hint_path]['exists'] = True
                return state_copy
            else:
                logger.error("Cannot get state: Simulation state not initialized.")
                return {"error": "Simulation state not initialized."}

    def _get_real_system_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        """ Probes the real system (Docker/Subprocess) for state information. """
        current_cwd = self._get_current_cwd()
        logger.debug(f"Probing real system state (Mode: {'Docker' if self.docker_container else 'Subprocess'}, CWD: {current_cwd})...")
        state: Dict[str, Any] = {
            'timestamp': time.time(),
            'filesystem': {},
            'resources': {},
            'mode': 'docker' if self.docker_container else 'subprocess',
            'cwd': current_cwd,
            'target_path_hint': target_path_hint,
            'probe_errors': []
        }
        probe_results = {}
        exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context

        # Define probe commands
        probes = {
            'cpu': "top -bn1 | grep '^%Cpu' | head -n1", # Standard Linux top command
            'mem': "grep -E 'MemTotal|MemAvailable' /proc/meminfo", # Standard Linux proc file
            'disk_cwd': f"df -k {shlex.quote(current_cwd)}", # Disk usage for current dir mount point
            'ls_cwd': f"ls -lA --full-time {shlex.quote(current_cwd)}", # Detailed listing of CWD
        }

        # Add probe for target path hint if provided and valid
        abs_target_hint: Optional[str] = None
        if target_path_hint:
            abs_target_hint = self._resolve_path(target_path_hint)
            if abs_target_hint:
                probes['stat_target'] = f'stat {shlex.quote(abs_target_hint)}'
            else:
                 state['probe_errors'].append(f"Invalid target path hint for probing: {target_path_hint}")

        # Execute probes
        for key, cmd in probes.items():
            res = self._execute_real_command(cmd, exec_func)
            probe_results[key] = res
            if not res.get('success'):
                state['probe_errors'].append(f"Probe '{key}' failed (Code {res.get('exit_code','?')})")
                logger.warning(f"State probe '{key}' cmd failed: {cmd} -> {res.get('stderr', 'No stderr')}")

        # Parse probe results
        try:
            # Resources
            cpu_res = probe_results.get('cpu')
            mem_res = probe_results.get('mem')
            disk_res = probe_results.get('disk_cwd')
            if cpu_res and cpu_res.get('success'):
                # Try various regex patterns for CPU idle % then calculate usage
                idle_match = re.search(r"ni,\s*([\d\.]+)\s*id,", cpu_res['stdout']) # Standard Linux top
                if idle_match:
                     try: idle_perc = float(idle_match.group(1)); state['resources']['cpu_load_percent'] = round(100.0 - idle_perc, 1)
                     except ValueError: pass
                else: # Fallback patterns if needed
                     pass # Add other regex if needed for different 'top' versions
            if mem_res and mem_res.get('success'):
                total_kb = re.search(r"MemTotal:\s+(\d+)\s*kB", mem_res['stdout'])
                avail_kb = re.search(r"MemAvailable:\s+(\d+)\s*kB", mem_res['stdout'])
                if total_kb and avail_kb:
                    try: total = int(total_kb.group(1)); avail = int(avail_kb.group(1)); state['resources']['memory_usage_percent'] = round(((total - avail) / total) * 100.0, 1) if total > 0 else 0.0
                    except (ValueError, ZeroDivisionError): pass
            if disk_res and disk_res.get('success'):
                # More robust df parsing: find line with CWD, get % use
                lines = disk_res['stdout'].strip().split('\n')
                if len(lines) > 1:
                     match = re.search(r'\s+(\d+)%\s+(?:/[^\s]*)?$', lines[-1]) # Check last line for % usage
                     if match:
                         try: state['resources']['disk_usage_percent'] = float(match.group(1))
                         except ValueError: pass

            # Filesystem - CWD Listing
            state['filesystem'][current_cwd]={'type':'directory','content_listing':None,'error':None, 'exists': True} # Assume CWD exists if we are 'in' it
            ls_res = probe_results.get('ls_cwd')
            if ls_res:
                if ls_res.get('success'): state['filesystem'][current_cwd]['content_listing'] = ls_res['stdout']
                else: state['filesystem'][current_cwd]['error'] = ls_res.get('stderr', 'ls failed')

            # Filesystem - Target Path Stat
            stat_res = probe_results.get('stat_target')
            if stat_res and abs_target_hint:
                stat_entry = {'type':None,'exists':False,'error':None, 'size_bytes': None, 'mtime': None, 'perms_octal': None, 'perms_symbolic': None, 'owner': None, 'stat_output': None}
                if stat_res.get('success'):
                    stat_out = stat_res['stdout']
                    stat_entry['exists'] = True
                    stat_entry['stat_output'] = stat_out
                    # Determine type
                    if 'directory' in stat_out: stat_entry['type'] = 'directory'
                    elif 'regular empty file' in stat_out: stat_entry['type'] = 'file'; stat_entry['size_bytes'] = 0
                    elif 'regular file' in stat_out: stat_entry['type'] = 'file'
                    else: stat_entry['type'] = 'other' # Symlink, pipe, etc.
                    # Extract other details using regex
                    size_match = re.search(r"Size:\s*(\d+)", stat_out)
                    mtime_match = re.search(r"Modify:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+[+-]\d{4}", stat_out) # More specific timestamp match
                    perms_match = re.search(r"Access:\s*\((\d+)/([a-zA-Z-]+)\)", stat_out) # Octal and symbolic
                    owner_match = re.search(r"Uid:\s*\(\s*\d+/\s*([\w-]+)\)", stat_out) # User name
                    if size_match: stat_entry['size_bytes'] = int(size_match.group(1))
                    if mtime_match:
                        try: # Attempt to parse timestamp for consistency (optional)
                            # Example: time.mktime(time.strptime(mtime_match.group(1), '%Y-%m-%d %H:%M:%S.%f'))
                            stat_entry['mtime'] = mtime_match.group(1) # Store raw string for now
                        except: pass # Ignore parsing errors
                    if perms_match: stat_entry['perms_octal'] = perms_match.group(1); stat_entry['perms_symbolic'] = perms_match.group(2)
                    if owner_match: stat_entry['owner'] = owner_match.group(1)
                else: # Stat failed
                    stat_entry['exists'] = False
                    stat_entry['error'] = stat_res.get('stderr', 'Stat failed')
                state['filesystem'][abs_target_hint] = stat_entry

        except Exception as parse_err:
            logger.error(f"Error parsing real system state: {parse_err}", exc_info=True)
            state['parsing_error'] = f"State parsing failed: {parse_err}"

        # Clean up empty error list
        if not state.get('probe_errors'):
            state.pop('probe_errors', None)
        return state
    # --- End State Retrieval ---

    def disconnect(self):
        """ Closes Docker client connection if open. """
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

# --- END OF FILE seed/vm_service.py ---