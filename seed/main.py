# --- START OF FILE seed/main.py ---

# seed/main.py
"""
Main entry point for the RSIAI Seed Orchestrator.
Initializes components based on configuration and runs the main control loop,
driving the Seed core strategic cycle. Includes self-restart mechanism
and manages persistence of essential state.
"""
import time
import json
import os
import gc
import sys          # For restart logic, sys.stdout/stderr
import pickle       # For restart state serialization
import copy         # For deep copying state during serialization
import traceback    # For printing traceback if execv or Popen fails
import logging
import subprocess   # For launching new process in _trigger_restart

# Keep TF import if still potentially needed elsewhere (e.g. future internal models)
import tensorflow as tf
from typing import Optional, List, Dict, Any, Callable
from collections import deque # For loading memory state

# --- Configuration ---
# Use relative import now that config.py is in the same package
from .config import (
    ALIGNMENT_DIRECTIVE, ALIGNMENT_CORE_LOGIC, ALIGNMENT_MISSION,
    MEMORY_SAVE_FILE, RESTART_SIGNAL_EVENT_TYPE,
    SEED_INITIAL_GOAL
)

# --- Core Components ---
# Use relative imports now that these are in the same package
from .memory_system import MemorySystem
from .llm_service import Seed_LLMService
from .vm_service import Seed_VMService
from .evaluator import Seed_SuccessEvaluator
from .sensory import Seed_SensoryRefiner
from .core import Seed_Core

# Setup root logger
logger = logging.getLogger(__name__)

# --- Constants for Self-Restart Mechanism ---
RESTART_STATE_FILE = MEMORY_SAVE_FILE.replace(".pkl", "_restart_state.pkl")

# --- Main Orchestrator Class ---
class Main_Orchestrator:
    """
    Initializes and coordinates the RSIAI Seed system. Runs the main execution loop,
    driving Seed Core strategic cycles, including checks for self-restart.
    Handles persistence of core state and memory.
    """
    def __init__(self, load_restart_state: bool = False):
        """ Initializes all core components and services based on config. """
        logger.info("--- Initializing Main Orchestrator ---")
        start_time = time.time()
        self.is_restarting = load_restart_state

        self.memory: MemorySystem = MemorySystem()

        loaded_state_components = None
        if self.is_restarting and os.path.exists(RESTART_STATE_FILE):
            logger.warning(f"Restart flag set. Attempting load component state from: {RESTART_STATE_FILE}")
            try:
                with open(RESTART_STATE_FILE, 'rb') as f:
                    loaded_state_components = pickle.load(f)
                logger.info("Successfully loaded component state from restart file.")
            except Exception as e:
                logger.error(f"Failed load state from restart file '{RESTART_STATE_FILE}': {e}. Proceeding with standard init.", exc_info=True)
                loaded_state_components = None

        logger.info("Initializing Seed Services/Shared Components...")
        self.llm_service: Seed_LLMService = Seed_LLMService(memory_system=self.memory)
        self.vm_service: Seed_VMService = Seed_VMService()
        self.success_evaluator: Seed_SuccessEvaluator = Seed_SuccessEvaluator()
        self.sensory_refiner: Seed_SensoryRefiner = Seed_SensoryRefiner()

        logger.info("Initializing Seed Core...")
        self.seed_core: Seed_Core = Seed_Core(
            llm_service=self.llm_service,
            vm_service=self.vm_service,
            memory_system=self.memory,
            success_evaluator=self.success_evaluator,
            sensory_refiner=self.sensory_refiner,
        )

        self.total_cycles_run: int = 0
        self.is_running: bool = False
        self.start_time: float = 0.0

        if loaded_state_components and isinstance(loaded_state_components, dict):
            seed_core_state = loaded_state_components.get('seed_core_state')
            if seed_core_state and isinstance(seed_core_state, dict):
                 logger.info("Restoring Seed Core state (goal) from loaded data.")
                 initial_goal = seed_core_state.get('current_goal', {})
                 if initial_goal:
                     if hasattr(self.seed_core, 'set_initial_state'):
                          self.seed_core.set_initial_state(initial_goal)
                     elif hasattr(self.seed_core, 'current_goal'):
                          self.seed_core.current_goal = initial_goal
                          logger.warning("Seed Core lacks set_initial_state method, setting current_goal attribute directly.")
                     else:
                           logger.error("Could not restore goal: Seed Core lacks set_initial_state method and current_goal attribute.")
                 else:
                       logger.warning("Restart state file loaded, but missing 'current_goal' in 'seed_core_state'.")

            orchestrator_state = loaded_state_components.get('orchestrator_state')
            if orchestrator_state and isinstance(orchestrator_state, dict):
                 self.total_cycles_run = orchestrator_state.get('total_cycles_run', 0)
                 if hasattr(self.seed_core, 'cycle_count'):
                     self.seed_core.cycle_count = self.total_cycles_run
                 logger.info(f"Restored orchestrator cycle count to {self.total_cycles_run}.")

        init_duration = time.time() - start_time
        logger.info(f"--- Main Orchestrator Initialized ({init_duration:.2f}s) ---")


    def _check_for_restart_signal(self) -> bool:
        """ Checks memory for the seed restart signal event. """
        try:
            restart_signals = self.memory.find_lifelong_by_criteria(
                lambda e: e.get('key', '').startswith(RESTART_SIGNAL_EVENT_TYPE),
                limit=1, newest_first=True )
            if restart_signals:
                logger.warning(f"RESTART SIGNAL DETECTED in memory (Event: {restart_signals[0].get('key')}).")
                return True
        except Exception as e: logger.error(f"Error checking for restart signal: {e}", exc_info=True)
        return False

    def _serialize_state_for_restart(self) -> bool:
        """ Saves essential state needed for restart to the restart file. """
        logger.info(f"Serializing essential state to restart file: {RESTART_STATE_FILE}")
        state_to_save = {}
        try:
            state_to_save['seed_core_state'] = {
                'current_goal': copy.deepcopy(self.seed_core.current_goal),
            }
            state_to_save['orchestrator_state'] = {
                'total_cycles_run': self.total_cycles_run,
            }
            with open(RESTART_STATE_FILE, 'wb') as f:
                pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Minimal restart state successfully serialized to {RESTART_STATE_FILE}.")

            logger.info("Saving full memory state before triggering restart...")
            if hasattr(self, 'memory') and self.memory:
                 self.memory.save_memory()
            else:
                 logger.error("Memory object not available for pre-restart save!")
                 return False

            return True
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Failed to serialize state or save memory for restart: {e}", exc_info=True)
            if os.path.exists(RESTART_STATE_FILE):
                try: os.remove(RESTART_STATE_FILE)
                except OSError as cleanup_err: logger.error(f"Failed to remove corrupt restart file '{RESTART_STATE_FILE}': {cleanup_err}")
            return False

    def _trigger_restart(self):
        # 1. Log intent (logger should still be working here)
        logger.warning("--- TRIGGERING SELF-RESTART (via new console for Windows) ---")

        # 2. Perform all your pre-shutdown cleanups
        if hasattr(self, 'vm_service') and hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            try:
                self.vm_service.disconnect()
                logger.info("VM Service disconnected pre-restart.")
            except Exception as e:
                logger.error(f"Error disconnecting VM Service pre-restart: {e}")

        if 'tensorflow' in sys.modules and 'tf' in globals(): # Check if tf was successfully imported
            try:
                tf.keras.backend.clear_session()
                gc.collect()
                logger.info("TF Session cleared and garbage collected pre-restart.")
            except Exception as e:
                logger.error(f"Error during TF cleanup pre-restart: {e}")
        else:
            gc.collect()
            logger.info("Garbage collected pre-restart (TF not imported/used or 'tf' not in globals).")

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception as e:
            # Use print as logger might be affected if stderr is broken by now
            print(f"WARNING: Error flushing standard streams pre-restart: {e}", file=sys.stderr)

        # 3. Shutdown logging - CRITICAL
        print("INFO: Shutting down logging system prior to new console restart...", file=sys.stderr)
        logging.shutdown() # After this, logger calls won't work as expected

        # 4. Prepare command for the new process
        python_executable = sys.executable
        entry_module = 'seed.main'
        command_to_run = [python_executable, '-m', entry_module, '--restarted']

        project_root_dir = None
        try:
            # Assuming __file__ is .../RSIAI0/seed/main.py
            # os.path.dirname(__file__) is .../RSIAI0/seed
            # os.path.join(..., '..') is .../RSIAI0
            project_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if not os.path.isdir(project_root_dir):
                print(f"WARNING: Calculated project_root_dir '{project_root_dir}' does not seem to be a valid directory. Falling back to None for CWD.", file=sys.stderr)
                project_root_dir = None
        except Exception as e_path:
            print(f"WARNING: Could not determine project_root_dir for new process CWD: {e_path}. CWD will be inherited or default.", file=sys.stderr)
            project_root_dir = None


        print(f"INFO: Attempting to launch new console with: {' '.join(command_to_run)}", file=sys.stderr)
        if project_root_dir:
            print(f"INFO: Setting CWD for new process to: {project_root_dir}", file=sys.stderr)

        try:
            if sys.platform == "win32":
                subprocess.Popen(command_to_run,
                                 creationflags=subprocess.CREATE_NEW_CONSOLE,
                                 cwd=project_root_dir)
            else:
                subprocess.Popen(command_to_run, cwd=project_root_dir)

            print("INFO: New process launched. Current process will now exit.", file=sys.stderr)
            sys.exit(0)

        except Exception as e:
            print(f"FATAL: Failed to launch new process for restart: {e}", file=sys.stderr)
            print("Traceback for Popen failure:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


    def _main_loop(self, max_cycles: Optional[int] = None):
        logger.info("Starting main orchestrator loop...")
        while self.is_running:
            loop_start_time = time.monotonic()
            logger.debug(f"Orchestrator: Triggering Seed Core Cycle {self.total_cycles_run + 1}...")
            seed_core_cycle_completed_successfully = False
            try:
                if not self.seed_core or not hasattr(self.seed_core, 'run_strategic_cycle') or not callable(self.seed_core.run_strategic_cycle):
                    logger.critical("Seed Core object invalid or missing 'run_strategic_cycle'. Stopping.")
                    self.is_running = False; break
                self.seed_core.run_strategic_cycle()
                self.total_cycles_run += 1
                seed_core_cycle_completed_successfully = True
                logger.debug(f"Orchestrator: Completed Seed Core Cycle {self.total_cycles_run}")
            except Exception as cycle_err:
                 logger.error(f"Error during Seed Core strategic cycle {self.total_cycles_run + 1}: {cycle_err}", exc_info=True)
                 try:
                     self.memory.log("SEED_CycleCriticalError", {"cycle": self.total_cycles_run + 1, "error": str(cycle_err)}, tags=['Seed','Error','Cycle','Critical'])
                 except Exception as log_err: logger.error(f"Failed log cycle error: {log_err}")

            if seed_core_cycle_completed_successfully and self._check_for_restart_signal():
                if self._serialize_state_for_restart():
                    self._trigger_restart()
                else:
                    logger.critical("Failed serialize state/save memory for restart. Aborting restart and stopping.")
                    self.is_running = False
                break

            if max_cycles is not None and self.total_cycles_run >= max_cycles:
                logger.info(f"Orchestrator: Reached max cycles ({max_cycles}). Stopping.")
                self.is_running = False

            loop_elapsed = time.monotonic() - loop_start_time
            sleep_time = max(0.01, 0.02 - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        logger.info("Orchestrator main loop finished.")

    def run(self, max_cycles: Optional[int] = None):
        if self.is_running: logger.warning("Orchestrator already running."); return
        run_type = f'Max Cycles: {max_cycles}' if max_cycles else 'Indefinite'
        logger.info(f"\n*** Starting RSIAI Seed Run ({run_type}) ***")
        self.is_running = True; self.start_time = time.time()
        try:
             if self.is_running: self._main_loop(max_cycles=max_cycles)
        except KeyboardInterrupt: logger.info("\nOrchestrator: KeyboardInterrupt received. Initiating shutdown...")
        except Exception as e:
             logger.critical(f"Orchestrator: UNHANDLED EXCEPTION in main loop: {e}", exc_info=True)
             if hasattr(self, 'memory') and self.memory:
                 try: self.memory.log("OrchestratorError", {"error": str(e), "traceback": traceback.format_exc()}, tags=['Critical', 'Error', 'Orchestrator'])
                 except Exception as log_err: logger.error(f"Failed log critical orchestrator error: {log_err}")
        finally:
             if self.is_running: self.is_running = False
             logger.info("Orchestrator performing cleanup...")
             self.stop()

    def stop(self):
        if not self.is_running and self.start_time == 0 and self.total_cycles_run == 0:
            logger.info("Orchestrator already stopped or not fully run.")
            return
        run_duration = time.time() - self.start_time if self.start_time > 0 else 0
        logger.info(f"\n--- Stopping RSIAI Seed Run (Duration: {run_duration:.2f}s) ---")
        self.is_running = False

        if hasattr(self, 'vm_service') and hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            logger.info("Disconnecting VM Service...")
            try: self.vm_service.disconnect()
            except Exception as e: logger.error(f"Error disconnecting VM Service: {e}", exc_info=True)

        logger.info("Performing final memory save...")
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'save_memory'):
             try: self.memory.save_memory()
             except Exception as e: logger.error(f"Error during final memory save: {e}", exc_info=True)

        if 'tensorflow' in sys.modules and 'tf' in globals():
            logger.info("Collecting garbage and clearing TF session...")
            try:
                 tf.keras.backend.clear_session(); gc.collect();
                 logger.debug("TF session cleared and GC collected.")
            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt received during TensorFlow session cleanup. Skipping further TF cleanup.")
            except Exception as e: logger.error(f"Error during final GC/TF cleanup: {e}", exc_info=True)
        else:
             logger.info("Collecting garbage...")
             gc.collect()
        logger.info(f"RSIAI Seed Run Stopped. Total Seed Cycles: {self.total_cycles_run}")
        self.start_time = 0.0

# --- Script Execution Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    is_restarted = '--restarted' in sys.argv
    if is_restarted:
        print("--- Orchestrator performing restart ---", file=sys.stderr)
        time.sleep(2)

    script_logger = logging.getLogger(__name__)

    if 'tensorflow' in sys.modules and 'tf' in globals():
        script_logger.info("--- TensorFlow Configuration ---"); script_logger.info(f"TF Version: {tf.__version__}")
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                    script_logger.info(f"Found {len(gpus)} Physical GPUs, Configured {len(logical_gpus)} Logical GPUs (Mem Growth).")
                except RuntimeError as e: script_logger.error(f"GPU Memory Growth Error: {e}.")
            else: script_logger.info("No GPUs detected by TensorFlow.")
        except Exception as e: script_logger.error(f"GPU Config Error: {e}", exc_info=True)
        script_logger.info("-----------------------------")
    else:
        script_logger.info("TensorFlow not imported or 'tf' not in globals, skipping GPU configuration.")

    script_logger.info("\n--- AGI Alignment Reminder ---")
    script_logger.info(f"Mission: {ALIGNMENT_MISSION}")
    script_logger.info(f"Core Logic: {ALIGNMENT_CORE_LOGIC}")
    script_logger.info(f"Directive: {ALIGNMENT_DIRECTIVE}")
    script_logger.info("--------------------------------")

    orchestrator = None
    try:
        orchestrator = Main_Orchestrator(load_restart_state=is_restarted)

        if is_restarted and os.path.exists(RESTART_STATE_FILE):
             script_logger.info("Processing restart state file post-init (primarily for cleanup)...")
             try:
                 if not orchestrator.seed_core.current_goal:
                     script_logger.warning("Post-init check: Seed Core goal seems not restored from restart file or was empty.")
                 os.remove(RESTART_STATE_FILE)
                 script_logger.info(f"Removed restart state file: {RESTART_STATE_FILE}")
             except OSError as e:
                 script_logger.error(f"Failed remove restart state file post-load: {e}")
             except Exception as post_init_err:
                 script_logger.error(f"Error during post-init restart state processing: {post_init_err}", exc_info=True)

        if not orchestrator.seed_core.current_goal:
             script_logger.info("Setting initial Seed goal...")
             if hasattr(orchestrator.seed_core, 'set_initial_state'):
                 orchestrator.seed_core.set_initial_state(goal=SEED_INITIAL_GOAL)
             elif hasattr(orchestrator.seed_core, 'current_goal'):
                 orchestrator.seed_core.current_goal = SEED_INITIAL_GOAL
                 script_logger.warning("Seed Core lacks set_initial_state, setting current_goal directly.")
             else:
                  script_logger.critical("Cannot set initial goal: Seed Core missing goal attributes/methods.")

        orchestrator.run(max_cycles=None)

    except Exception as main_exception:
         script_logger.critical(f"FATAL ERROR during Orchestrator setup or run: {main_exception}", exc_info=True)
         if orchestrator and hasattr(orchestrator, 'stop') and callable(orchestrator.stop):
             script_logger.info("Attempting emergency stop...")
             try: orchestrator.stop()
             except Exception as stop_err: script_logger.error(f"Emergency stop failed: {stop_err}")
    finally:
        script_logger.info("\n--- Post-Run Analysis ---")
        if orchestrator and hasattr(orchestrator, 'memory') and orchestrator.memory:
            try:
                def dump_mem_summary(label: str, filter_func: Callable[[Dict], bool], limit: int = 3, newest: bool = True):
                    try:
                        entries = orchestrator.memory.find_lifelong_by_criteria(filter_func, limit=limit, newest_first=newest)
                        summaries = []
                        for e in entries:
                            key = e.get('key', 'no_key')
                            data_part = e.get('data', {})
                            summary_content = data_part.get('message')
                            if summary_content is None:
                                try:
                                    summary_content_str = json.dumps(data_part, default=str, ensure_ascii=False)
                                    summary_content = summary_content_str[:100] + ("..." if len(summary_content_str) > 100 else "")
                                except Exception: summary_content = "[Data Summary Error - Non-JSON]"
                            else:
                                summary_content = str(summary_content)[:100] + ("..." if len(str(summary_content)) > 100 else "")
                            summaries.append(f"({key}: {summary_content})")
                        script_logger.info(f"\n{label} (Last {len(entries)}): {', '.join(summaries)}")
                    except Exception as e_dump:
                        script_logger.error(f"Error retrieving/summarizing '{label}': {e_dump}", exc_info=True)

                latest_epi = orchestrator.memory.get_latest_episodic(5);
                epi_summary = [f"({e.get('id', '?')}: {e.get('data',{}).get('event_type', '?')})" for e in latest_epi]
                script_logger.info(f"Last {len(latest_epi)} Episodic: {', '.join(epi_summary)}")

                dump_mem_summary("Last SEED Evals", lambda e: e.get('key','').startswith("SEED_Evaluation"))
                dump_mem_summary("Last SEED Decisions", lambda e: e.get('key','').startswith("SEED_Decision"))
                dump_mem_summary("Last SEED Goals", lambda e: e.get('key','').startswith("seed_goal_set"), limit=2)
                dump_mem_summary("Last CoreMod Writes", lambda e: e.get('key','').startswith("SEED_Action_MODIFY_CORE_CODE"), limit=2)
                dump_mem_summary("Last CoreMod Verifications", lambda e: e.get('key','').startswith("SEED_Action_VERIFY_CORE"), limit=2)
                dump_mem_summary("Last CoreMod Tests", lambda e: e.get('key','').startswith("SEED_Action_TEST_CORE"), limit=2)
                dump_mem_summary("Last Restarts Req", lambda e: e.get('key','').startswith(RESTART_SIGNAL_EVENT_TYPE), limit=5)
                dump_mem_summary("Recent Errors", lambda e: 'Error' in e.get('tags', []) or 'Critical' in e.get('tags',[]))

                current_params = orchestrator.memory.get_learning_parameter('')
                if current_params: # Check if params are not None
                    param_summary = json.dumps({
                        cat: {p: v.get('value') for p, v in params.items()} if isinstance(params, dict) and cat=="evaluation_weights"
                        else params.get('value') if isinstance(params, dict)
                        else params # Should not happen with current structure
                        for cat, params in current_params.items()
                    }, indent=2, default=str) # Added default=str for non-serializable
                    script_logger.info(f"\nFinal Learning Parameters:\n{param_summary}")
                else:
                    script_logger.warning("Could not retrieve final learning parameters for summary.")

                rules = orchestrator.memory.get_behavioral_rules()
                script_logger.info(f"\nFinal Behavioral Rules ({len(rules)}): {list(rules.keys())}")

            except Exception as post_run_error:
                script_logger.error(f"Post-run analysis error: {post_run_error}", exc_info=True)
        else:
            script_logger.warning("Orchestrator/memory not available for post-run analysis.")
        script_logger.info("\n--- RSIAI Seed Execution Finished ---")

# --- END OF FILE seed/main.py ---