# --- START OF FILE seed/verification.py ---

# RSIAI/seed/verification.py
"""
Handles the execution of verification tests for proposed core code modifications.
Applies modifications to temporary code copies, runs specified test suites,
parses results, and performs cleanup.
"""
import time
import logging # Import base logging module
import os
import shutil
import subprocess
import pathlib
import traceback
import re
import ast
import tempfile
from typing import Dict, Any, Tuple, Optional, List, Union

# --- Configuration ---
# Use relative import now that config.py is in the same package
from .config import (
    CORE_CODE_VERIFICATION_TIMEOUT_SEC,
    CORE_CODE_VERIFICATION_TEMP_DIR, # Used as prefix now
    CORE_CODE_VERIFICATION_SUITES
)

# --- AST Unparsing ---
# Use built-in ast.unparse if available (Python 3.9+)
# Otherwise, require 'astor' library as a fallback (needs pip install astor)
# Use the 'logging' module directly here as 'logger' instance isn't defined yet.
try:
    from ast import unparse as ast_unparse
    ASTOR_AVAILABLE = False
    logging.debug("Using built-in ast.unparse for verification.") # Use logging directly
except ImportError:
    try:
        import astor
        ast_unparse = astor.to_source
        ASTOR_AVAILABLE = True
        logging.info("Using 'astor' library for AST unparsing during verification. (pip install astor)") # Use logging directly
    except ImportError:
        ast_unparse = None
        ASTOR_AVAILABLE = False
        logging.error("AST unparsing library not available (Python < 3.9 and 'astor' not installed). REPLACE_FUNCTION/METHOD verification will fail.") # Use logging directly

# Define the logger for the rest of the module
logger = logging.getLogger(__name__)

# --- AST Node Transformer for Function/Method Replacement ---
class ReplaceFunctionTransformer(ast.NodeTransformer):
    """
    AST transformer to find and replace a specific function or method definition.
    """
    def __init__(self, target_name: str, new_func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        super().__init__()
        self.target_name = target_name
        self.new_func_node = new_func_node
        self.target_found_and_replaced = False
        self.is_method = False # We don't strictly need to know if it's method for replacement

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Optional[ast.AST]:
        if node.name == self.target_name:
            logger.debug(f"AST Transformer: Found target function '{self.target_name}'. Replacing.")
            self.target_found_and_replaced = True
            return self.new_func_node
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Optional[ast.AST]:
        if node.name == self.target_name:
            logger.debug(f"AST Transformer: Found target async function '{self.target_name}'. Replacing.")
            self.target_found_and_replaced = True
            if not isinstance(self.new_func_node, ast.AsyncFunctionDef):
                 logger.warning(f"AST Transformer: Replacing async function '{self.target_name}' with non-async node. This might be incorrect.")
            return self.new_func_node
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        return self.generic_visit(node)


# --- Helper Function to Apply Modification ---
def apply_modification_to_copy(
    temp_file_path: pathlib.Path,
    mod_type: str,
    target_id: str, # Can be function/method name or line content
    new_code: Optional[str]
) -> Tuple[bool, str]:
    """
    Applies the specified modification to the copied file at temp_path.
    Handles different modification types (line-based and AST-based).

    Args:
        temp_file_path: Path to the temporary file copy to modify.
        mod_type: Type of modification (REPLACE_LINE, INSERT_AFTER_LINE, DELETE_LINE,
                                      REPLACE_FUNCTION, REPLACE_METHOD).
        target_id: Identifier for the target (line content or function/method name).
        new_code: The new code string (for REPLACE*, INSERT* types). None for DELETE_LINE.

    Returns:
        Tuple[bool, str]: (success, message)
    """
    logger.debug(f"Applying modification '{mod_type}' to temp file '{temp_file_path}' (Target: '{target_id[:50]}...')")
    try:
        if not temp_file_path.is_file():
             return False, f"Temporary file missing at {temp_file_path}"

        if mod_type in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]:
            # --- Line-based Modifications ---
            with open(temp_file_path, 'r', encoding='utf-8') as f_read:
                original_lines = f_read.readlines()

            target_indices = [i for i, line in enumerate(original_lines) if target_id.strip() in line.strip()]

            if not target_indices:
                return False, f"Target line content not found in temp file: '{target_id[:100]}...'"
            if len(target_indices) > 1:
                logger.warning(f"Target line content ambiguous ({len(target_indices)} matches) in '{temp_file_path}'. Using first match for verification.")
            target_idx = target_indices[0]
            modified_lines = original_lines[:]

            if mod_type == "REPLACE_LINE":
                if new_code is None: return False, "Missing 'new_code' for REPLACE_LINE"
                modified_lines[target_idx] = new_code + ('\n' if not new_code.endswith('\n') else '')
            elif mod_type == "INSERT_AFTER_LINE":
                if new_code is None: return False, "Missing 'new_code' for INSERT_AFTER_LINE"
                indent = re.match(r"^\s*", original_lines[target_idx]).group(0) if target_idx < len(original_lines) else ""
                lines_to_insert = [(indent + line) for line in new_code.splitlines(True)]
                if lines_to_insert and not lines_to_insert[-1].endswith('\n'): lines_to_insert[-1] += '\n'
                modified_lines[target_idx+1:target_idx+1] = lines_to_insert
            elif mod_type == "DELETE_LINE":
                if target_idx < len(modified_lines):
                     del modified_lines[target_idx]
                else: return False, f"Target index {target_idx} out of bounds for DELETE_LINE."

            modified_code_str = "".join(modified_lines)

        elif mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]:
            # --- AST-based Modifications ---
            if new_code is None:
                return False, f"Missing 'new_code' for {mod_type}"
            if ast_unparse is None:
                 return False, "AST unparsing library (ast.unparse or astor) not available. Cannot perform function/method replacement."

            try:
                with open(temp_file_path, 'r', encoding='utf-8') as f_read:
                    original_code_str = f_read.read()
                original_ast = ast.parse(original_code_str, filename=str(temp_file_path))

                new_code_ast_module = ast.parse(new_code.strip(), filename="<new_code>")
                if not new_code_ast_module.body or not isinstance(new_code_ast_module.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return False, "Provided 'new_code' does not parse into a valid function/method definition."
                new_func_node = new_code_ast_module.body[0]

                transformer = ReplaceFunctionTransformer(target_name=target_id, new_func_node=new_func_node)
                modified_ast = transformer.visit(original_ast)

                if not transformer.target_found_and_replaced:
                    return False, f"Target function/method '{target_id}' not found in the AST of '{temp_file_path}'."

                modified_code_str = ast_unparse(modified_ast)
                logger.debug(f"AST modification successful for '{target_id}'.")

            except SyntaxError as syn_err:
                 return False, f"Syntax error parsing original or new code: {syn_err}"
            except Exception as ast_err:
                 return False, f"Error during AST manipulation for '{target_id}': {ast_err}"
        else:
            return False, f"Unknown modification_type: {mod_type}"

        # --- Validate Syntax of the final modified code ---
        try:
            ast.parse(modified_code_str)
        except SyntaxError as final_syn_err:
            logger.error(f"Modified code resulted in Syntax Error: {final_syn_err}")
            return False, f"Modification resulted in invalid Python syntax: {final_syn_err}"

        # --- Write the modified code back to the temporary file ---
        try:
            with open(temp_file_path, 'w', encoding='utf-8') as f_write:
                f_write.write(modified_code_str)
            logger.debug(f"Successfully wrote modified code back to '{temp_file_path}'.")
            return True, "Modification applied successfully to temporary file."
        except IOError as write_err:
            return False, f"Failed to write modified temporary file: {write_err}"

    except Exception as e:
        logger.error(f"Unexpected error applying modification to {temp_file_path}: {e}", exc_info=True)
        return False, f"Unexpected error during modification: {e}"


# --- Main Verification Function ---
def run_verification_suite(
    project_root: pathlib.Path,
    modification_params: Dict[str, Any],
    verification_level: str = "basic"
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Runs a verification test suite against a proposed core code modification
    by applying the change to a temporary copy of the project and executing tests.

    Args:
        project_root: The root directory of the AGI project.
        modification_params: Dictionary containing details of the proposed change
                             (file_path, modification_type, target_id, new_code, etc.).
        verification_level: String key referencing the test suite command list
                             in config.CORE_CODE_VERIFICATION_SUITES.

    Returns:
        Tuple[bool, str, Dict]: (success, message, details_dict)
                                details_dict contains stdout, stderr, exit_code, etc.
    """
    start_time = time.time()
    success = False
    message = "Verification failed during setup."
    details: Dict[str, Any] = {"output": "", "error": None, "exit_code": -1, "level": verification_level, "duration_sec": 0.0}
    temp_dir_path: Optional[pathlib.Path] = None # Initialize path variable

    try:
        # --- Parameter Extraction and Validation ---
        file_rel_path_str = modification_params.get("file_path")
        mod_type = modification_params.get("modification_type")
        if mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]:
             target_id = modification_params.get("target_name")
        elif mod_type in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]:
             target_id = modification_params.get("target_line_content")
        else: raise ValueError(f"Unknown modification_type for verification: {mod_type}")
        if mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]: new_code = modification_params.get("new_logic")
        elif mod_type == "REPLACE_LINE" or mod_type == "INSERT_AFTER_LINE": new_code = modification_params.get("new_content")
        else: new_code = None
        if not file_rel_path_str or not target_id: raise ValueError("Missing required modification parameters (file_path, target_id/target_line_content).")
        if new_code is None and mod_type != "DELETE_LINE": raise ValueError(f"Missing required 'new_code'/'new_content' for modification type '{mod_type}'.")

        file_rel_path = pathlib.Path(file_rel_path_str)
        original_file_path = project_root.joinpath(file_rel_path).resolve()
        if not original_file_path.is_file(): raise FileNotFoundError(f"Original file for verification not found at '{original_file_path}'")

        # --- Temporary Directory Setup ---
        temp_dir_path = pathlib.Path(tempfile.mkdtemp(prefix=f"{CORE_CODE_VERIFICATION_TEMP_DIR}_"))
        logger.info(f"Verification starting for {file_rel_path} (Level: {verification_level}) in temp dir: {temp_dir_path}")

        # --- Copy Project Structure ---
        ignore_patterns = shutil.ignore_patterns( '.git*', '*__pycache__*', '*.pyc', '*.pyo', '.DS_Store', CORE_CODE_VERIFICATION_TEMP_DIR + "*", CORE_CODE_MODIFICATION_BACKUP_DIR, '*.log', '*.pkl', '*.bak', '.env', 'venv', '.venv' )
        shutil.copytree(project_root, temp_dir_path, ignore=ignore_patterns, dirs_exist_ok=True)
        temp_file_path = temp_dir_path.joinpath(file_rel_path).resolve()
        logger.debug(f"Copied project structure to '{temp_dir_path}'. Target file temp path: '{temp_file_path}'")

        # --- Apply Modification ---
        applied_ok, apply_message = apply_modification_to_copy(temp_file_path, mod_type, target_id, new_code)
        if not applied_ok: message = f"Failed to apply modification to temporary file: {apply_message}"; details["error"] = message; raise RuntimeError(message)
        logger.info("Modification applied successfully to temporary code copy.")

        # --- Select and Run Test Suite ---
        test_command_list = CORE_CODE_VERIFICATION_SUITES.get(verification_level)
        if not test_command_list or not isinstance(test_command_list, list): raise ValueError(f"Verification level '{verification_level}' not found or invalid in config.")
        logger.info(f"Running verification command: {' '.join(test_command_list)} in temp dir '{temp_dir_path}'")

        # --- Execute Tests ---
        try:
            proc = subprocess.run( test_command_list, cwd=temp_dir_path, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=CORE_CODE_VERIFICATION_TIMEOUT_SEC, check=False, shell=False )
            exit_code = proc.returncode; stdout = proc.stdout; stderr = proc.stderr
            logger.info(f"Verification command finished with exit code {exit_code}.")
        except subprocess.TimeoutExpired: exit_code = -9; stdout = ""; stderr = f"Verification timed out after {CORE_CODE_VERIFICATION_TIMEOUT_SEC} seconds."; logger.error(stderr)
        except FileNotFoundError: exit_code = 127; stdout = ""; stderr = f"Verification command not found: '{test_command_list[0]}'. Ensure test runner (e.g., pytest) is installed and in PATH."; logger.error(stderr)
        except Exception as subp_err: exit_code = -1; stdout = ""; stderr = f"Error running verification command: {subp_err}\n{traceback.format_exc()}"; logger.error(f"Error running verification subprocess: {subp_err}")

        details['exit_code'] = exit_code; details['output'] = stdout.strip(); details['error'] = stderr.strip()

        # --- Parse Test Results ---
        if exit_code == 0:
            success = True; message = f"Verification PASSED (Level: {verification_level})."
            logger.info(message)
        else:
            success = False; stderr_snippet = f" Stderr: {stderr[:200].strip()}..." if stderr else ""
            message = f"Verification FAILED (Level: {verification_level}). Exit Code: {exit_code}.{stderr_snippet}"; logger.warning(message)
            if stderr: logger.debug(f"Full Verification Stderr:\n{stderr}")

    except Exception as e:
        success = False; message = f"Verification process failed: {e}"
        current_error = details.get("error", "")
        if not current_error or str(e) not in current_error: details["error"] = current_error + f"\nVerification Workflow Error:\n{traceback.format_exc()}"
        logger.error(message, exc_info=True)
    finally:
        # --- Cleanup Temporary Directory ---
        if temp_dir_path and temp_dir_path.exists():
            try:
                logger.debug(f"Cleaning up verification temp directory: {temp_dir_path}")
                shutil.rmtree(temp_dir_path, ignore_errors=True)
            except Exception as clean_err:
                logger.error(f"Failed to clean up verification temp directory {temp_dir_path}: {clean_err}")
                if "error" not in details: details["error"] = ""
                if "Cleanup Error" not in details["error"]: details["error"] += f"\nCleanup Error: {clean_err}"

    details["duration_sec"] = round(time.time() - start_time, 2)
    return success, message, details

# --- END OF FILE seed/verification.py ---