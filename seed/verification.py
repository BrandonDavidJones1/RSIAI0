# --- START OF FILE seed/verification.py ---

# RSIAI/seed/verification.py
"""
Handles the execution of verification tests for proposed core code modifications.
Applies modifications to temporary code copies, runs specified test suites,
parses results, and performs cleanup.
"""
import time
import logging
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
from ..config import ( # Adjusted relative import
    CORE_CODE_VERIFICATION_TIMEOUT_SEC,
    CORE_CODE_VERIFICATION_TEMP_DIR, # Used as prefix now
    CORE_CODE_VERIFICATION_SUITES
)

# --- AST Unparsing ---
# Use built-in ast.unparse if available (Python 3.9+)
# Otherwise, require 'astor' library as a fallback (needs pip install astor)
try:
    from ast import unparse as ast_unparse
    ASTOR_AVAILABLE = False
    logger.debug("Using built-in ast.unparse for verification.")
except ImportError:
    try:
        import astor
        ast_unparse = astor.to_source
        ASTOR_AVAILABLE = True
        logger.info("Using 'astor' library for AST unparsing during verification. (pip install astor)")
    except ImportError:
        ast_unparse = None
        ASTOR_AVAILABLE = False
        logger.error("AST unparsing library not available (Python < 3.9 and 'astor' not installed). REPLACE_FUNCTION/METHOD verification will fail.")

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
            # Return the new node, replacing the old one
            # Ensure the replacement node has the correct type hints/decorators if needed (complex)
            # For now, direct replacement assumes the new_func_node is complete.
            return self.new_func_node
        # Visit children nodes normally if name doesn't match
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Optional[ast.AST]:
        # Handle async functions similarly
        if node.name == self.target_name:
            logger.debug(f"AST Transformer: Found target async function '{self.target_name}'. Replacing.")
            self.target_found_and_replaced = True
            # Ensure the replacement is also an AsyncFunctionDef if needed
            if not isinstance(self.new_func_node, ast.AsyncFunctionDef):
                 logger.warning(f"AST Transformer: Replacing async function '{self.target_name}' with non-async node. This might be incorrect.")
            return self.new_func_node
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        # Visit methods within classes
        # We don't need to track entering/exiting class scope explicitly for simple name matching
        # unless target_name needs class qualification (e.g., "MyClass.method").
        # Current implementation assumes target_name is just the function/method name.
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

            # Find target line index (match stripped content for flexibility)
            target_indices = [i for i, line in enumerate(original_lines) if target_id.strip() in line.strip()]

            if not target_indices:
                return False, f"Target line content not found in temp file: '{target_id[:100]}...'"
            if len(target_indices) > 1:
                # Be stricter during verification than during apply? Maybe allow first match?
                logger.warning(f"Target line content ambiguous ({len(target_indices)} matches) in '{temp_file_path}'. Using first match for verification.")
                # return False, f"Target line content ambiguous ({len(target_indices)} matches)"
            target_idx = target_indices[0]
            modified_lines = original_lines[:] # Copy list

            if mod_type == "REPLACE_LINE":
                if new_code is None: return False, "Missing 'new_code' for REPLACE_LINE"
                modified_lines[target_idx] = new_code + ('\n' if not new_code.endswith('\n') else '')
            elif mod_type == "INSERT_AFTER_LINE":
                if new_code is None: return False, "Missing 'new_code' for INSERT_AFTER_LINE"
                indent = re.match(r"^\s*", original_lines[target_idx]).group(0) if target_idx < len(original_lines) else ""
                lines_to_insert = [(indent + line) for line in new_code.splitlines(True)]
                modified_lines[target_idx+1:target_idx+1] = lines_to_insert
            elif mod_type == "DELETE_LINE":
                del modified_lines[target_idx]

            modified_code_str = "".join(modified_lines)

        elif mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]:
            # --- AST-based Modifications ---
            if new_code is None:
                return False, f"Missing 'new_code' for {mod_type}"
            if ast_unparse is None:
                 return False, "AST unparsing library (ast.unparse or astor) not available. Cannot perform function/method replacement."

            try:
                # Parse the original code from the temp file
                with open(temp_file_path, 'r', encoding='utf-8') as f_read:
                    original_code_str = f_read.read()
                original_ast = ast.parse(original_code_str, filename=str(temp_file_path))

                # Parse the new code (should be a function definition)
                new_code_ast_module = ast.parse(new_code.strip(), filename="<new_code>")
                if not new_code_ast_module.body or not isinstance(new_code_ast_module.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
                    return False, "Provided 'new_code' does not parse into a valid function/method definition."
                new_func_node = new_code_ast_module.body[0]

                # Apply the transformation
                transformer = ReplaceFunctionTransformer(target_name=target_id, new_func_node=new_func_node)
                modified_ast = transformer.visit(original_ast)

                if not transformer.target_found_and_replaced:
                    return False, f"Target function/method '{target_id}' not found in the AST of '{temp_file_path}'."

                # Unparse the modified AST back to code
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
            # Optionally log the bad code snippet (carefully)
            # logger.debug(f"--- Bad Code Snippet ---\n{modified_code_str[:1000]}\n--- End Snippet ---")
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
        # Target ID depends on mod type
        if mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]: target_id = modification_params.get("target_name")
        elif mod_type in ["REPLACE_LINE", "INSERT_AFTER_LINE", "DELETE_LINE"]: target_id = modification_params.get("target_line_content")
        else: raise ValueError(f"Unknown modification_type for verification: {mod_type}")
        # New code depends on mod type
        if mod_type in ["REPLACE_FUNCTION", "REPLACE_METHOD"]: new_code = modification_params.get("new_logic")
        elif mod_type == "REPLACE_LINE" or mod_type == "INSERT_AFTER_LINE": new_code = modification_params.get("new_content")
        else: new_code = None # Not needed for DELETE_LINE

        if not file_rel_path_str or not target_id:
            raise ValueError("Missing required modification parameters (file_path, target_id).")

        file_rel_path = pathlib.Path(file_rel_path_str)
        original_file_path = project_root.joinpath(file_rel_path).resolve()

        # --- Temporary Directory Setup ---
        # Create a temporary directory within the project (or system temp)
        # Using system temp might be safer regarding accidental deletion
        temp_dir_path = pathlib.Path(tempfile.mkdtemp(prefix=f"{CORE_CODE_VERIFICATION_TEMP_DIR}_"))
        logger.info(f"Verification starting for {file_rel_path} (Level: {verification_level}) in temp dir: {temp_dir_path}")

        # --- Copy Project Structure (excluding noise) ---
        ignore_patterns = shutil.ignore_patterns(
            '.git*', '*__pycache__*', '*.pyc', '*.pyo',
            CORE_CODE_MODIFICATION_BACKUP_DIR, # Exclude backups
            f"{CORE_CODE_VERIFICATION_TEMP_DIR}*", # Exclude other temp verify dirs
            '*.log', '*.pkl', '*.bak', # Exclude common log/data/backup files
            '.env', 'venv', '.venv' # Exclude virtualenvs
        )
        # Copy the entire project root content to the temp dir
        shutil.copytree(project_root, temp_dir_path, ignore=ignore_patterns, dirs_exist_ok=True)
        temp_file_path = temp_dir_path.joinpath(file_rel_path).resolve()
        logger.debug(f"Copied project structure to '{temp_dir_path}'. Target file temp path: '{temp_file_path}'")

        # --- Apply Modification to Temporary Copy ---
        applied_ok, apply_message = apply_modification_to_copy(temp_file_path, mod_type, target_id, new_code)
        if not applied_ok:
            message = f"Failed to apply modification to temporary file: {apply_message}"
            details["error"] = message
            raise RuntimeError(message)
        logger.info("Modification applied successfully to temporary code copy.")

        # --- Select and Run Test Suite ---
        test_command_list = CORE_CODE_VERIFICATION_SUITES.get(verification_level)
        if not test_command_list or not isinstance(test_command_list, list):
             raise ValueError(f"Verification level '{verification_level}' not found or invalid in config.")

        logger.info(f"Running verification command: {' '.join(test_command_list)} in temp dir '{temp_dir_path}'")

        # --- Execute Tests using subprocess ---
        try:
            proc = subprocess.run(
                test_command_list,
                cwd=temp_dir_path, # Run tests from the temporary directory context!
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=CORE_CODE_VERIFICATION_TIMEOUT_SEC,
                check=False # Don't raise exception on non-zero exit code
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
            logger.info(f"Verification command finished with exit code {exit_code}.")

        except subprocess.TimeoutExpired:
            exit_code = -9 # Indicate timeout
            stdout = ""
            stderr = f"Verification timed out after {CORE_CODE_VERIFICATION_TIMEOUT_SEC} seconds."
            logger.error(stderr)
        except FileNotFoundError:
             exit_code = 127
             stdout = ""
             stderr = f"Verification command not found: '{test_command_list[0]}'. Ensure test runner (e.g., pytest) is installed and in PATH."
             logger.error(stderr)
        except Exception as subp_err:
            exit_code = -1 # Indicate other error
            stdout = ""
            stderr = f"Error running verification command: {subp_err}\n{traceback.format_exc()}"
            logger.error(f"Error running verification subprocess: {subp_err}")

        details['exit_code'] = exit_code
        details['output'] = stdout.strip() # Store stripped output
        details['error'] = stderr.strip() # Store stripped error

        # --- Parse Test Results ---
        if exit_code == 0:
            success = True
            message = f"Verification PASSED (Level: {verification_level})."
            logger.info(message)
        else:
            success = False
            message = f"Verification FAILED (Level: {verification_level}). Exit Code: {exit_code}."
            logger.warning(message)
            if stderr: logger.warning(f"Verification Stderr Snippet: {stderr[:500].strip()}...")

    except Exception as e:
        success = False
        # Ensure message reflects the actual error
        message = f"Verification process failed: {e}"
        details["error"] = details.get("error","") + f"\nVerification Workflow Error:\n{traceback.format_exc()}" # Append traceback
        logger.error(message, exc_info=True)
    finally:
        # --- Cleanup Temporary Directory ---
        if temp_dir_path and temp_dir_path.exists():
            try:
                logger.debug(f"Cleaning up verification temp directory: {temp_dir_path}")
                shutil.rmtree(temp_dir_path, ignore_errors=True) # Use ignore_errors for resilience
            except Exception as clean_err:
                # Log error but don't overwrite primary failure reason
                logger.error(f"Failed to clean up verification temp directory {temp_dir_path}: {clean_err}")
                if "error" not in details: details["error"] = ""
                details["error"] += f"\nCleanup Error: {clean_err}"

    details["duration_sec"] = round(time.time() - start_time, 2)
    return success, message, details

# --- END OF FILE seed/verification.py ---