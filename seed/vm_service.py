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
from pathlib import PurePosixPath
from typing import Dict, Any, Optional, Tuple, Callable, List

logger = logging.getLogger(__name__)

# --- Configuration ---
from ..config import ( # Adjusted relative import
    VM_SERVICE_USE_REAL, VM_SERVICE_DOCKER_CONTAINER,
    VM_SERVICE_COMMAND_TIMEOUT_SEC, VM_SERVICE_ALLOWED_REAL_COMMANDS,
    # LOWER_LEVEL_SIM_BASE_COMMANDS removed
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
        self._core_real_probes = ['sh', 'pwd', 'ls', 'df', 'stat', 'grep', 'head', 'tail', 'top', 'cat', 'echo', 'rm', 'touch', 'mkdir', 'mv', 'cp', 'printf']
        base_allowed = VM_SERVICE_ALLOWED_REAL_COMMANDS if isinstance(VM_SERVICE_ALLOWED_REAL_COMMANDS, list) else []
        self.allowed_real_commands: List[str] = list(set(base_allowed + self._core_real_probes))

        self.use_real_system: bool = use_real_system and (DOCKER_AVAILABLE or not docker_container_name)
        self.docker_container_name: Optional[str] = docker_container_name if DOCKER_AVAILABLE and use_real_system else None
        self.docker_client = None
        self.docker_container = None
        self._simulated_state: Optional[Dict] = None # Renamed internal var

        self.command_timeout_sec: int = VM_SERVICE_COMMAND_TIMEOUT_SEC
        self._real_system_cwd: str = '/app'
        self._simulated_system_cwd: str = '/app'

        if self.use_real_system:
            mode_name = "Subprocess"
            if self.docker_container_name:
                if self._connect_docker(): mode_name = "Docker"
                else: logger.warning(f"Failed to connect to Docker '{self.docker_container_name}'. Falling back to Subprocess."); self.docker_container_name = None; mode_name = "Subprocess (Docker Fallback)"
            logger.info(f"Initializing Seed VMService (Real System Mode: {mode_name}). Allowed commands: {sorted(self.allowed_real_commands)}") # Renamed log
        else:
            logger.info("Initializing Seed VMService (Simulation Mode)...") # Renamed log
            self._initialize_simulation()

    def _connect_docker(self) -> bool:
        if not self.docker_container_name or not DOCKER_AVAILABLE: return False
        logger.info(f"Attempting connection to Docker container: {self.docker_container_name}")
        try:
            self.docker_client = docker.from_env(timeout=20)
            if not self.docker_client.ping(): logger.error("Docker daemon is not responding."); self.docker_client = None; return False
            self.docker_container = self.docker_client.containers.get(self.docker_container_name)
            if self.docker_container.status.lower() != 'running': logger.error(f"Docker container '{self.docker_container_name}' not running (Status: {self.docker_container.status})."); self.docker_container = None; return False
            test_result_dict = self._docker_exec_context("pwd")
            if test_result_dict.get('success'):
                cwd_test = test_result_dict.get('stdout','').strip()
                if cwd_test != self._real_system_cwd: logger.warning(f"Docker test CWD ('{cwd_test}') differs from VMService internal CWD ('{self._real_system_cwd}'). VMService will use internal CWD.")
                logger.info(f"Successfully connected to Docker container '{self.docker_container_name}'.")
                return True
            else: logger.error(f"Docker test command 'pwd' failed. Exit Code: {test_result_dict.get('exit_code')}. Stderr: {test_result_dict.get('stderr','')}"); self.docker_container = None; return False
        except DockerNotFound: logger.error(f"Docker container '{self.docker_container_name}' not found."); self.docker_container = None; self.docker_client = None; return False
        except DockerAPIError as api_err: logger.error(f"Docker API error connecting: {api_err}"); self.docker_container = None; self.docker_client = None; return False
        except Exception as e: logger.error(f"Unexpected error connecting to Docker: {e}", exc_info=True); self.docker_container = None; self.docker_client = None; return False

    def _initialize_simulation(self):
        """ Sets up the initial simulated environment state. """
        logger.info("Setting up simulated filesystem and state...")
        self._simulated_system_cwd = '/app'
        # Simplified base commands for simulation mode (core file ops)
        self._sim_available_commands = ['ls', 'pwd', 'cat', 'echo', 'touch', 'mkdir', 'rm', 'cd', 'cp', 'mv']
        # Renamed internal var
        self._simulated_state = {
            'timestamp': time.time(), 'cwd': self._simulated_system_cwd,
            'filesystem': { # Basic FS
                '/': {'type': 'directory', 'owner': 'root', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/app': {'type': 'directory', 'owner': 'user', 'perms': 'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096},
                '/tmp': {'type': 'directory', 'owner': 'user', 'perms': 'rwxrwxrwx', 'mtime': time.time(), 'size_bytes': 4096},
                '/app/placeholder.txt': {'type': 'file', 'content': 'Initial content.', 'owner': 'user', 'perms': 'rw-r--r--', 'mtime': time.time(), 'size_bytes': 16},
            },
            'resources': {'cpu_load_percent': 1.0, 'memory_usage_percent': 5.0, 'disk_usage_percent': 10.0}, # Lower defaults
            'last_command_result': None,
            'available_commands': list(self._sim_available_commands) # Use simplified list
        }
        logger.info(f"Simulation Initialized. CWD: {self._simulated_system_cwd}. Sim commands: {self._sim_available_commands}")

    def _get_current_cwd(self) -> str:
        return self._real_system_cwd if self.use_real_system else self._simulated_system_cwd

    def _set_current_cwd(self, new_cwd: str):
        try:
            posix_path = PurePosixPath(new_cwd); normalized_cwd = str(posix_path.resolve())
            if not posix_path.is_absolute(): raise ValueError("CWD must be absolute")
            if self.use_real_system: self._real_system_cwd = normalized_cwd
            else: self._simulated_system_cwd = normalized_cwd
        except (TypeError, ValueError) as e: logger.error(f"Attempted to set invalid CWD '{new_cwd}': {e}")

    # --- Path/Permission Helpers ---
    def _resolve_path(self, path_str: str) -> Optional[str]:
        current_dir = self._get_current_cwd()
        if not path_str: return current_dir
        try:
            cwd_path = PurePosixPath(current_dir)
            if not cwd_path.is_absolute(): logger.error(f"Internal CWD '{current_dir}' is not absolute! Falling back to '/'."); cwd_path = PurePosixPath('/')
            if path_str == '~' or path_str.startswith('~/'): home_dir = '/app'; path_part = path_str[2:] if path_str.startswith('~/') else ''; target_path = PurePosixPath(home_dir) / path_part
            else: input_path = PurePosixPath(path_str); target_path = cwd_path / input_path if not input_path.is_absolute() else input_path
            resolved_path_str = str(target_path.resolve())
            if not PurePosixPath(resolved_path_str).is_absolute(): logger.warning(f"Path resolution resulted in non-absolute path '{resolved_path_str}' for input '{path_str}'. Using original target."); return str(target_path)
            return resolved_path_str
        except Exception as e: logger.warning(f"Path resolution error for '{path_str}' from '{current_dir}': {e}"); return None

    def _sim_check_permissions(self, path: str, action: str = 'read') -> Tuple[bool, str]:
        if not self._simulated_state: return False, "Simulation state not initialized"
        fs = self._simulated_state['filesystem']; parent = str(PurePosixPath(path).parent) if path != '/' else None; parent_info = fs.get(parent) if parent else None; item_info = fs.get(path)
        if action in ['write', 'delete', 'create']:
            if path == '/': return False, "Permission denied: cannot modify root directory"
            if not parent: return False, "Internal error: Cannot determine parent for non-root path."
            if not parent_info: return False, f"Parent directory '{parent}' does not exist"
            if parent_info.get('type') != 'directory': return False, f"Parent '{parent}' is not a directory"
            if parent_info.get('owner') != 'user' and parent != '/tmp': return False, f"Permission denied writing to parent '{parent}'"
        if item_info:
            owner = item_info.get('owner', 'system'); perms = item_info.get('perms', '---------'); can_read = (owner == 'user' and perms[1] == 'r') or (owner != 'user' and perms[7] == 'r'); can_write = (owner == 'user' and perms[2] == 'w')
            if action == 'read': return (True, "") if can_read else (False, "Permission denied (read)")
            if action == 'write': return (True, "") if can_write else (False, "Permission denied (write)")
            if action == 'delete': return (True, "") if can_write else (False, "Permission denied (delete)")
        elif action == 'read': return False, "No such file or directory"
        elif action == 'create': return True, ""
        elif action in ['write', 'delete']: return False, "No such file or directory"
        return False, "Permission check fallback (denied)"
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
        if not self._simulated_state: result['stderr'],result['reason']="Sim not init.",'internal_error'; return result
        try: parts = shlex.split(command_str)
        except ValueError as e: result['stderr'],result['reason']=f"Command parse error: {e}",'parse_error'; return result
        cmd=parts[0] if parts else ''; args=parts[1:]
        # Use internal list of allowed sim commands
        if not cmd or cmd not in self._sim_available_commands:
            result['stderr'],result['reason']=f"Command not found/allowed in sim: {cmd}",'illegal_command'; return result
        try:
            fs = self._simulated_state['filesystem']; resolve_path_func = self._resolve_path
            # Simplified command logic (core file ops)
            if cmd == 'pwd': result['stdout'], result['success'], result['exit_code'] = current_cwd, True, 0
            elif cmd == 'cd':
                target_dir_str = args[0] if args else '/app'; resolved_path = resolve_path_func(target_dir_str)
                if resolved_path and resolved_path in fs and fs[resolved_path].get('type') == 'directory': allowed, msg = self._sim_check_permissions(resolved_path, 'read');
                if allowed: self._set_current_cwd(resolved_path); self._simulated_state['cwd'] = self._get_current_cwd(); result['success'], result['exit_code'] = True, 0
                else: result['stderr'], result['reason'] = msg, 'permission_denied'
                elif resolved_path and resolved_path not in fs: result['stderr'], result['reason'] = f"cd: No such file or directory: {target_dir_str}", 'file_not_found'
                elif resolved_path: result['stderr'], result['reason'] = f"cd: Not a directory: {target_dir_str}", 'is_not_directory'
                else: result['stderr'], result['reason'] = f"cd: Invalid path: {target_dir_str}", 'invalid_path'
            elif cmd == 'ls':
                target_path_str = args[0] if args else '.'; resolved_path = resolve_path_func(target_path_str)
                if resolved_path and resolved_path in fs: item_info = fs[resolved_path]; allowed, msg = self._sim_check_permissions(resolved_path, 'read');
                if not allowed: result['stderr'], result['reason'] = msg, 'permission_denied'
                elif item_info.get('type') == 'directory': contents = [PurePosixPath(n).name for n,f in fs.items() if str(PurePosixPath(n).parent) == resolved_path and n != resolved_path]; result['stdout'], result['success'], result['exit_code'] = "\n".join(sorted(contents)), True, 0
                else: result['stdout'], result['success'], result['exit_code'] = PurePosixPath(resolved_path).name, True, 0
                else: result['stderr'], result['reason'] = f"ls: cannot access '{target_path_str}': No such file or directory", 'file_not_found'
            elif cmd == 'cat':
                if not args: result['stderr'], result['reason'] = "cat: missing file operand", 'missing_args'
                else: p = resolve_path_func(args[0]);
                if p and p in fs: info = fs[p]; allowed, msg = self._sim_check_permissions(p, 'read');
                if not allowed: result['stderr'], result['reason'] = msg, 'permission_denied'
                elif info.get('type') == 'file': result['stdout'], result['success'], result['exit_code'] = info.get('content',''), True, 0
                else: result['stderr'], result['reason'] = f"cat: {args[0]}: Is a directory", 'is_directory'
                else: result['stderr'], result['reason'] = f"cat: {args[0]}: No such file or directory", 'file_not_found'
            elif cmd == 'echo':
                content_to_echo = ""; target_file = None; redirect_mode='>'; append=False
                try:
                    if '>' in args: idx = args.index('>');
                    if idx > 0 and args[idx-1] == '>': idx -= 1; append=True; redirect_mode='>>'
                    content_to_echo = " ".join(args[:idx]).strip("'\""); target_file = args[idx+len(redirect_mode)] if idx+len(redirect_mode)<len(args) else None
                    else: content_to_echo = " ".join(args).strip("'\"")
                except ValueError: redirect_mode = None
                if redirect_mode and target_file: p = resolve_path_func(target_file);
                if p: allowed, msg = self._sim_check_permissions(p, 'write') if p in fs else self._sim_check_permissions(p, 'create');
                if allowed: existing_content = fs.get(p,{}).get('content','') if append else ''; new_content = existing_content + content_to_echo + '\n'; fs[p] = {'type': 'file', 'content': new_content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(new_content.encode())}; result['success'], result['exit_code'] = True, 0
                else: result['stderr'], result['reason'] = msg, 'permission_denied'
                else: result['stderr'], result['reason'] = f"echo: Invalid path: {target_file}", 'invalid_path'
                elif redirect_mode and not target_file: result['stderr'], result['reason'] = f"echo: missing target for redirection '{redirect_mode}'", 'parse_error'
                else: result['stdout'], result['success'], result['exit_code'] = content_to_echo, True, 0 # Echo to stdout
            elif cmd == 'touch':
                if not args: result['stderr'], result['reason'] = "touch: missing file operand", 'missing_args'
                else: p = resolve_path_func(args[0]);
                if p: allowed, msg = self._sim_check_permissions(p, 'write') if p in fs else self._sim_check_permissions(p, 'create');
                if allowed:
                    if p in fs and fs[p].get('type') == 'file': fs[p]['mtime'] = time.time()
                    elif p in fs and fs[p].get('type') == 'directory': result['stderr'], result['reason'] = f"touch: cannot touch '{args[0]}': Is a directory", 'is_directory'
                    else: fs[p] = {'type': 'file', 'content': '', 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': 0}
                    result['success'], result['exit_code'] = True, 0
                else: result['stderr'], result['reason'] = msg, 'permission_denied'
                else: result['stderr'], result['reason'] = f"touch: Invalid path: {args[0]}", 'invalid_path'
            elif cmd == 'mkdir':
                if not args: result['stderr'], result['reason'] = "mkdir: missing operand", 'missing_args'
                else: p = resolve_path_func(args[0]);
                if p:
                    if p in fs: result['stderr'], result['reason'] = f"mkdir: cannot create directory '{args[0]}': File exists", 'file_exists'
                    else: allowed, msg = self._sim_check_permissions(p, 'create');
                    if allowed: fs[p] = {'type': 'directory', 'owner': 'user', 'perms':'rwxr-xr-x', 'mtime': time.time(), 'size_bytes': 4096}; result['success'], result['exit_code'] = True, 0
                    else: result['stderr'], result['reason'] = msg, 'permission_denied'
                else: result['stderr'], result['reason'] = f"mkdir: Invalid path: {args[0]}", 'invalid_path'
            elif cmd == 'rm':
                 if not args: result['stderr'], result['reason'] = "rm: missing operand", 'missing_args'
                 else: p = resolve_path_func(args[0]);
                 if p and p in fs: info = fs[p]; allowed, msg = self._sim_check_permissions(p, 'delete');
                 if not allowed: result['stderr'], result['reason'] = msg, 'permission_denied'
                 elif info.get('type') == 'directory': is_empty = not any(str(PurePosixPath(n).parent)==p for n in fs if n != p);
                 if is_empty: del fs[p]; result['success'], result['exit_code'] = True, 0
                 else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': Directory not empty", 'directory_not_empty'
                 else: del fs[p]; result['success'], result['exit_code'] = True, 0
                 elif p == '/': result['stderr'], result['reason'] = "rm: cannot remove root directory", 'permission_denied'
                 else: result['stderr'], result['reason'] = f"rm: cannot remove '{args[0]}': No such file or directory", 'file_not_found'
            elif cmd == 'cp':
                 if len(args) != 2: result['stderr'], result['reason'] = "cp: missing destination file operand", 'missing_args'
                 else: src_p = resolve_path_func(args[0]); dest_p = resolve_path_func(args[1]);
                 if not src_p or not dest_p: result['stderr'], result['reason'] = "cp: Invalid path", 'invalid_path'
                 elif src_p == dest_p: result['stderr'], result['reason'] = f"cp: '{args[0]}' and '{args[1]}' are the same file", 'invalid_argument'
                 elif src_p not in fs: result['stderr'], result['reason'] = f"cp: cannot stat '{args[0]}': No such file or directory", 'file_not_found'
                 elif fs[src_p].get('type') == 'directory': result['stderr'], result['reason'] = f"cp: omitting directory '{args[0]}'", 'is_directory'
                 else: src_allowed, src_msg = self._sim_check_permissions(src_p, 'read'); dest_allowed, dest_msg = self._sim_check_permissions(dest_p, 'write') if dest_p in fs else self._sim_check_permissions(dest_p, 'create');
                 if not src_allowed: result['stderr'], result['reason'] = f"cp: {src_msg}", 'permission_denied'
                 elif not dest_allowed: result['stderr'], result['reason'] = f"cp: {dest_msg}", 'permission_denied'
                 else: fs[dest_p] = copy.deepcopy(fs[src_p]); fs[dest_p]['owner'] = 'user'; fs[dest_p]['mtime'] = time.time(); result['success'], result['exit_code'] = True, 0
            elif cmd == 'mv':
                 if len(args) != 2: result['stderr'], result['reason'] = "mv: missing destination file operand", 'missing_args'
                 else: src_p = resolve_path_func(args[0]); dest_p = resolve_path_func(args[1]);
                 if not src_p or not dest_p: result['stderr'], result['reason'] = "mv: Invalid path", 'invalid_path'
                 elif src_p == dest_p: result['stderr'], result['reason'] = f"mv: '{args[0]}' and '{args[1]}' are the same file", 'invalid_argument'
                 elif src_p not in fs: result['stderr'], result['reason'] = f"mv: cannot stat '{args[0]}': No such file or directory", 'file_not_found'
                 else: src_allowed, src_msg = self._sim_check_permissions(src_p, 'delete'); dest_allowed, dest_msg = self._sim_check_permissions(dest_p, 'write') if dest_p in fs else self._sim_check_permissions(dest_p, 'create');
                 if not src_allowed: result['stderr'], result['reason'] = f"mv: {src_msg}", 'permission_denied'
                 elif not dest_allowed: result['stderr'], result['reason'] = f"mv: {dest_msg}", 'permission_denied'
                 else: fs[dest_p] = fs.pop(src_p); fs[dest_p]['owner'] = 'user'; fs[dest_p]['mtime'] = time.time(); result['success'], result['exit_code'] = True, 0
            # Removed sleep/idle as distinct commands from simulation
            else: result['stderr'], result['reason'] = f"Sim command '{cmd}' logic missing.", 'not_implemented'

        except Exception as e: logger.error(f"Sim Internal Error exec '{command_str}': {e}",exc_info=True); result.update({'stderr':f"Internal sim error: {e}",'reason':'internal_error'})

        self._simulated_state['timestamp']=time.time()
        self._simulated_state['last_command_result']=copy.deepcopy(result)
        logger.debug(f"VM Sim Res: Succ={result['success']}, Code={result['exit_code']}, Out='{result['stdout'][:30]}...', Err='{result['stderr'][:30]}...'")
        return result

    def _execute_real_command(self, command_str: str, execution_context: Callable[[str], Dict]) -> Dict[str, Any]:
        # ... (Implementation remains the same, relies on self.allowed_real_commands and internal CWD) ...
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
                if target_dir_str == '~': home_res = execution_context("cd ~ && pwd");
                if home_res.get('success'): target_dir = home_res['stdout'].strip()
                else: result.update({'success':False, 'exit_code':home_res.get('exit_code', 1), 'stderr':"cd: Could not determine home directory.", 'reason':'internal_error'}); return result
                else: target_dir = self._resolve_path(target_dir_str)
                if not target_dir: result['stderr'], result['reason'] = f"cd: Invalid path: {target_dir_str}", 'invalid_path'; return result
                check_cmd = f"ls -ld {shlex.quote(target_dir)}"; check_res = execution_context(check_cmd)
                if check_res.get('success') and check_res.get('stdout','').strip().startswith('d'): self._set_current_cwd(target_dir); result['success'], result['exit_code'] = True, 0; logger.info(f"Real system CWD updated to: {self._get_current_cwd()}")
                elif check_res.get('exit_code') != 0: stderr = check_res.get('stderr', '').lower(); reason = 'execution_error';
                if 'no such file or directory' in stderr: reason = 'file_not_found'
                elif 'permission denied' in stderr: reason = 'permission_denied'
                result.update({'success':False, 'exit_code':check_res.get('exit_code', 1), 'stderr':f"cd: cannot access '{target_dir_str}': {check_res.get('stderr','Unknown error')}", 'reason':reason})
                else: result.update({'success':False, 'exit_code':1, 'stderr':f"cd: not a directory: {target_dir_str}", 'reason':'is_not_directory'})
                return result
            exec_res_dict = execution_context(command_str)
            return exec_res_dict
        except Exception as e: logger.error(f"Unexpected error during real command prep/exec '{command_str}': {e}", exc_info=True); result['stderr'], result['reason'] = f"Unexpected internal error: {e}", 'internal_error'; return result

    def _docker_exec_context(self, command_str: str) -> Dict[str, Any]:
        # ... (Implementation remains the same) ...
        res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1}
        if not self.docker_container: res['stderr']='Docker container unavailable'; res['reason']='docker_error'; return res
        try:
            full_cmd = f"sh -c {shlex.quote(command_str)}"; logger.debug(f"Docker Exec Run: cmd='{full_cmd}', workdir='{self._get_current_cwd()}'")
            exit_code, output = self.docker_container.exec_run( cmd=full_cmd, workdir=self._get_current_cwd(), stream=False, demux=False, user='root' )
            output_bytes: bytes = output if isinstance(output, bytes) else b''; output_str = output_bytes.decode(errors='replace').strip()
            res['exit_code'] = exit_code; res['success'] = (exit_code == 0); res['stdout'] = output_str if res['success'] else ''; res['stderr'] = '' if res['success'] else output_str
            if not res['success']: res['reason'] = 'execution_error';
            if not res['success'] and not res['stderr']: res['stderr'] = f"Command failed (Code {res['exit_code']}) with no output."
            return res
        except DockerAPIError as api_err: logger.error(f"Docker API error executing '{command_str}': {api_err}", exc_info=True); res.update({'stderr':f"Docker API error: {api_err}", 'reason':'docker_api_error'}); return res
        except Exception as e: logger.error(f"Docker exec_run unexpected error '{command_str}': {e}", exc_info=True); res.update({'stderr':f"Docker exec_run error: {e}", 'reason':'docker_error'}); return res

    def _subprocess_exec_context(self, command_str: str) -> Dict[str, Any]:
         # ... (Implementation remains the same) ...
         res: Dict[str, Any] = {'success': False, 'stdout': '', 'stderr': '', 'exit_code': -1}
         try:
            proc = subprocess.run(command_str, shell=True, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=self.command_timeout_sec, check=False, cwd=self._get_current_cwd())
            res['exit_code'] = proc.returncode; res['stdout'] = proc.stdout.strip() if proc.stdout else ''; res['stderr'] = proc.stderr.strip() if proc.stderr else ''; res['success'] = (proc.returncode == 0)
            if not res['success']: res['reason'] = 'execution_error';
            if not res['success'] and not res['stderr']: res['stderr'] = f"Command failed (Code {proc.returncode}) with no stderr."
            return res
         except FileNotFoundError: res['stderr'], res['reason'] = f"Command not found: {command_str.split()[0]}", 'command_not_found'; res['exit_code'] = 127; return res
         except subprocess.TimeoutExpired: res['stderr'], res['reason'] = f"Timeout ({self.command_timeout_sec}s)", 'timeout'; res['exit_code'] = -9; return res
         except Exception as e: logger.error(f"Subprocess error executing '{command_str}': {e}", exc_info=True); res['stderr']=f"Subprocess exec error: {e}"; res['reason']='internal_error'; return res
    # --- End Command Execution ---

    # --- Filesystem Operations ---
    def read_file(self, path: str) -> Dict[str, Any]:
        logger.info(f"Seed VMService: Reading file '{path}'") # Renamed log
        result = {'success': False, 'content': None, 'message': '', 'details': {'path': path}, 'reason': ''}
        abs_path = self._resolve_path(path)
        if not abs_path: result['message'] = "Invalid path provided."; result['reason'] = 'invalid_path'; return result
        result['details']['absolute_path'] = abs_path
        if self.use_real_system:
            if 'cat' not in self.allowed_real_commands: result['message'] = "Cannot read file: 'cat' command not allowed."; result['reason'] = 'safety_violation'; return result
            exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            cat_res = self._execute_real_command(f"cat {shlex.quote(abs_path)}", exec_func); result['details']['exit_code'] = cat_res.get('exit_code', -1); result['details']['stderr'] = cat_res.get('stderr'); result['reason'] = cat_res.get('reason', 'execution_error')
            if cat_res.get('success'): result['success'] = True; result['content'] = cat_res.get('stdout', ''); result['message'] = "File read successfully."
            else: result['message'] = f"Failed to read file: {cat_res.get('stderr', 'Unknown error')}"; stderr_lower = (cat_res.get('stderr') or '').lower();
            if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found'
            elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
            elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state: result['message'] = "Simulation not initialized."; result['reason'] = 'internal_error'; return result
            fs = self._simulated_state['filesystem']
            if abs_path in fs: item_info = fs[abs_path]; allowed, msg = self._sim_check_permissions(abs_path, 'read');
            if not allowed: result['message'] = msg; result['reason'] = 'permission_denied'
            elif item_info.get('type') == 'file': result['success'] = True; result['content'] = item_info.get('content', ''); result['message'] = "File read successfully (simulation)."
            elif item_info.get('type') == 'directory': result['message'] = "Cannot read: Is a directory."; result['reason'] = 'is_directory'
            else: result['message'] = f"Cannot read: Not a file (Type: {item_info.get('type')})."; result['reason'] = 'invalid_type'
            else: result['message'] = "File not found (simulation)."; result['reason'] = 'file_not_found'
        return result

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        logger.info(f"Seed VMService: Writing to file '{path}' (Content length: {len(content)})") # Renamed log
        result = {'success': False, 'message': '', 'details': {'path': path, 'content_length': len(content)}, 'reason': ''}
        abs_path = self._resolve_path(path)
        if not abs_path: result['message'] = "Invalid path provided."; result['reason'] = 'invalid_path'; return result
        result['details']['absolute_path'] = abs_path
        if self.use_real_system:
            if 'sh' not in self.allowed_real_commands or 'printf' not in self.allowed_real_commands: result['message'] = "Cannot write file: 'sh' or 'printf' command not allowed."; result['reason'] = 'safety_violation'; return result
            write_cmd = f"printf %s {shlex.quote(content)} > {shlex.quote(abs_path)}"; exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
            write_res = self._execute_real_command(write_cmd, exec_func); result['details']['exit_code'] = write_res.get('exit_code', -1); result['details']['stderr'] = write_res.get('stderr'); result['reason'] = write_res.get('reason', 'execution_error')
            if write_res.get('success'): result['success'] = True; result['message'] = "File written successfully."
            else: result['message'] = f"Failed to write file: {write_res.get('stderr', 'Unknown error')}"; stderr_lower = (write_res.get('stderr') or '').lower();
            if 'no such file or directory' in stderr_lower: result['reason'] = 'file_not_found'
            elif 'is a directory' in stderr_lower: result['reason'] = 'is_directory'
            elif 'permission denied' in stderr_lower: result['reason'] = 'permission_denied'
        else: # Simulation mode
            if not self._simulated_state: result['message'] = "Simulation not initialized."; result['reason'] = 'internal_error'; return result
            fs = self._simulated_state['filesystem']; allowed, msg = self._sim_check_permissions(abs_path, 'write') if abs_path in fs else self._sim_check_permissions(abs_path, 'create')
            if allowed: item_info = fs.get(abs_path);
            if item_info and item_info.get('type') == 'directory': result['message'] = "Cannot write: Is a directory."; result['reason'] = 'is_directory'
            else: fs[abs_path] = {'type': 'file', 'content': content, 'owner': 'user', 'perms':'rw-r--r--', 'mtime': time.time(), 'size_bytes': len(content.encode())}; result['success'] = True; result['message'] = "File written successfully (simulation)."
            else: result['message'] = msg; result['reason'] = 'permission_denied'
        return result
    # --- End Filesystem Operations ---

    # --- State Retrieval ---
    def get_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        """ Retrieves state snapshot from simulation OR real system. """
        if self.use_real_system:
             return self._get_real_system_state(target_path_hint)
        else: # Simulation Mode
            if self._simulated_state:
                state_copy = copy.deepcopy(self._simulated_state); state_copy['target_path_hint'] = target_path_hint; state_copy['mode'] = 'simulation'; state_copy['cwd'] = self._get_current_cwd()
                # Add simulated target stat info if hint provided
                if target_path_hint:
                    sim_fs = state_copy['filesystem']; abs_hint_path = self._resolve_path(target_path_hint)
                    if abs_hint_path: sim_fs[abs_hint_path] = sim_fs.get(abs_hint_path, {'type': None, 'exists': False}); sim_fs[abs_hint_path]['exists'] = abs_hint_path in sim_fs # Ensure exists flag is correct
                return state_copy
            else: logger.error("Cannot get state: Simulation state not initialized."); return {"error": "Simulation state not initialized."}

    def _get_real_system_state(self, target_path_hint: Optional[str] = None) -> Dict[str, Any]:
        # ... (Implementation remains the same, uses internal CWD and allowed commands) ...
        current_cwd = self._get_current_cwd()
        logger.debug(f"Probing real system state (Mode: {'Docker' if self.docker_container else 'Subprocess'}, CWD: {current_cwd})...")
        state: Dict[str, Any] = { 'timestamp': time.time(), 'filesystem': {}, 'resources': {}, 'mode': 'docker' if self.docker_container else 'subprocess', 'cwd': current_cwd, 'target_path_hint': target_path_hint, 'probe_errors': [] }; probe_results = {}; exec_func = self._docker_exec_context if self.docker_container else self._subprocess_exec_context
        probes = { 'cpu': "top -bn1 | grep '^%Cpu' | head -n1", 'mem': "grep -E 'MemTotal|MemAvailable' /proc/meminfo", 'disk_cwd': f"df -k {shlex.quote(current_cwd)}", 'ls_cwd': f"ls -lA --full-time {shlex.quote(current_cwd)}", }
        abs_target_hint: Optional[str] = None
        if target_path_hint: abs_target_hint = self._resolve_path(target_path_hint);
        if abs_target_hint: probes['stat_target'] = f'stat {shlex.quote(abs_target_hint)}'
        else: state['probe_errors'].append(f"Invalid target path hint: {target_path_hint}")
        for key, cmd in probes.items(): res = self._execute_real_command(cmd, exec_func); probe_results[key] = res;
        if not res.get('success'): state['probe_errors'].append(f"Probe '{key}' failed (Code {res.get('exit_code','?')})"); logger.warning(f"State probe '{key}' cmd failed: {cmd} -> {res.get('stderr', 'No stderr')}")
        try:
            cpu_res = probe_results.get('cpu'); mem_res = probe_results.get('mem'); disk_res = probe_results.get('disk_cwd'); ls_res = probe_results.get('ls_cwd'); stat_res = probe_results.get('stat_target')
            if cpu_res and cpu_res.get('success'): match = re.search(r"([\d\.]+) us", cpu_res['stdout']); state['resources']['cpu_load_percent'] = float(match.group(1)) if match else None
            if mem_res and mem_res.get('success'): total=re.search(r"MemTotal:\s+(\d+)",mem_res['stdout']); avail=re.search(r"MemAvailable:\s+(\d+)",mem_res['stdout']); state['resources']['memory_usage_percent']=((int(total.group(1))-int(avail.group(1)))/int(total.group(1)))*100 if total and avail and int(total.group(1))>0 else 0.0
            if disk_res and disk_res.get('success'): match=re.search(r"\s+(\d+)%\s+" + re.escape(current_cwd), disk_res['stdout']) or re.search(r"\s+(\d+)%\s+/\s*$", disk_res['stdout'], re.MULTILINE); state['resources']['disk_usage_percent']=float(match.group(1)) if match else None
            state['filesystem'][current_cwd]={'type':'directory','content_listing':None,'error':None}
            if ls_res and ls_res.get('success'): state['filesystem'][current_cwd]['content_listing']=ls_res['stdout']
            elif ls_res: state['filesystem'][current_cwd]['error']=ls_res.get('stderr','ls failed')
            if stat_res and abs_target_hint:
                if stat_res.get('success'): stat_out=stat_res['stdout']; f_type='directory' if 'directory' in stat_out else ('file' if ('regular file' in stat_out or 'regular empty file' in stat_out) else 'other'); size_match=re.search(r"Size:\s*(\d+)", stat_out); mtime_match=re.search(r"Modify:\s*(\d+)\s*\(", stat_out); perms_match=re.search(r"Access:\s*\((\d+)/([a-zA-Z-]+)\)", stat_out); owner_match=re.search(r"Uid:\s*\(\s*\d+/\s*([\w-]+)\)", stat_out); mtime_ts = int(mtime_match.group(1)) if mtime_match else None; state['filesystem'][abs_target_hint]={'type':f_type,'exists':True,'size_bytes':int(size_match.group(1)) if size_match else None,'mtime':mtime_ts,'perms_octal':perms_match.group(1) if perms_match else None,'perms_symbolic':perms_match.group(2) if perms_match else None,'owner':owner_match.group(1) if owner_match else None,'stat_output':stat_out}
                else: state['filesystem'][abs_target_hint]={'type':None,'exists':False,'error':stat_res.get('stderr','Stat failed')}
        except Exception as parse_err: logger.error(f"Error parsing real system state: {parse_err}", exc_info=True); state['parsing_error'] = f"State parsing failed: {parse_err}"
        if not state.get('probe_errors'): state.pop('probe_errors', None)
        return state
    # --- End State Retrieval ---

    def disconnect(self):
        """ Closes Docker client connection if open. """
        if self.docker_client:
            try: logger.info("Closing Docker client connection..."); self.docker_client.close(); logger.info("Docker client closed.")
            except Exception as e: logger.error(f"Error closing Docker client: {e}", exc_info=True)
            finally: self.docker_client = None; self.docker_container = None

# --- END OF FILE seed/vm_service.py ---