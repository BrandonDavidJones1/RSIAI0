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
import sys          # For restart logic
import pickle       # For restart state serialization
import copy         # For deep copying state during serialization
import traceback
import logging
# Keep TF import if still potentially needed elsewhere (e.g. future internal models)
# If definitely not needed, it can be removed.
# >>> FIX #4 NOTE: TF import is kept as its presence is checked later, removal is safe if unused. <<<
import tensorflow as tf
from typing import Optional, List, Dict, Any, Callable
from collections import deque # For loading memory state

# --- Configuration ---
# Use relative import now that config.py is in the same package
from .config import (
    ALIGNMENT_DIRECTIVE, ALIGNMENT_CORE_LOGIC, ALIGNMENT_MISSION,
    MEMORY_SAVE_FILE, RESTART_SIGNAL_EVENT_TYPE,
    SEED_INITIAL_GOAL # Use the correct constant name from config.py
)

# --- Core Components ---
# Use relative imports now that these are in the same package
from .memory_system import MemorySystem
from .llm_service import Seed_LLMService     # Corrected path and class name
from .vm_service import Seed_VMService       # Corrected path and class name
from .evaluator import Seed_SuccessEvaluator # Corrected path and class name
from .sensory import Seed_SensoryRefiner     # Corrected path and class name
from .core import Seed_Core                  # Corrected path and class name

# Setup root logger
# Note: If you want logging config applied *before* imports,
# it might need to be handled differently, maybe in __init__.py or a setup script.
# For now, assuming basicConfig is sufficient here.
logger = logging.getLogger(__name__)

# --- Constants for Self-Restart Mechanism ---
RESTART_STATE_FILE = MEMORY_SAVE_FILE.replace(".pkl", "_restart_state.pkl")

# --- Main Orchestrator Class ---
# Renamed class
class Main_Orchestrator:
    """
    Initializes and coordinates the RSIAI Seed system. Runs the main execution loop,
    driving Seed Core strategic cycles, including checks for self-restart.
    Handles persistence of core state and memory.
    """
    def __init__(self, load_restart_state: bool = False):
        """ Initializes all core components and services based on config. """
        # Updated log message
        logger.info("--- Initializing Main Orchestrator ---")
        start_time = time.time()
        self.is_restarting = load_restart_state # Flag if this is a restart run

        # Initialize Memory System (loads standard memory file if it exists)
        # MemorySystem now handles its own loading internally via load_memory()
        self.memory: MemorySystem = MemorySystem()

        # --- Load Component State if restarting (Memory loaded separately) ---
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
        # --- End Component State Loading ---

        # Initialize Seed Services & Helper Components
        logger.info("Initializing Seed Services/Shared Components...")
        self.llm_service: Seed_LLMService = Seed_LLMService()
        self.vm_service: Seed_VMService = Seed_VMService()
        self.success_evaluator: Seed_SuccessEvaluator = Seed_SuccessEvaluator()
        self.sensory_refiner: Seed_SensoryRefiner = Seed_SensoryRefiner()

        # Initialize Seed Core (Strategic decision-maker)
        logger.info("Initializing Seed Core...")
        self.seed_core: Seed_Core = Seed_Core( # Renamed internal variable
            llm_service=self.llm_service,
            vm_service=self.vm_service,
            memory_system=self.memory,
            success_evaluator=self.success_evaluator,
            sensory_refiner=self.sensory_refiner,
        )

        # Orchestrator State
        self.total_cycles_run: int = 0
        self.is_running: bool = False
        self.start_time: float = 0.0

        # --- Restore Orchestrator/Seed Core State from Loaded Data (if restarting) ---
        if loaded_state_components and isinstance(loaded_state_components, dict):
            # Restore Seed Core state (Goal)
            seed_core_state = loaded_state_components.get('seed_core_state') # Updated state key name
            if seed_core_state and isinstance(seed_core_state, dict):
                 logger.info("Restoring Seed Core state (goal) from loaded data.")
                 initial_goal = seed_core_state.get('current_goal', {})
                 # Use renamed core and method
                 if initial_goal: # Only set if goal was found in loaded state
                     if hasattr(self.seed_core, 'set_initial_state'):
                          # Set only goal; config loading handled by components now
                          self.seed_core.set_initial_state(initial_goal)
                     elif hasattr(self.seed_core, 'current_goal'): # Fallback if method missing/renamed
                          self.seed_core.current_goal = initial_goal
                          logger.warning("Seed Core lacks set_initial_state method, setting current_goal attribute directly.")
                     else:
                           logger.error("Could not restore goal: Seed Core lacks set_initial_state method and current_goal attribute.")
                 else:
                       logger.warning("Restart state file loaded, but missing 'current_goal' in 'seed_core_state'.")


            # Restore Orchestrator state
            orchestrator_state = loaded_state_components.get('orchestrator_state')
            if orchestrator_state and isinstance(orchestrator_state, dict):
                 self.total_cycles_run = orchestrator_state.get('total_cycles_run', 0)
                 # Sync cycle count if Seed Core keeps its own internal counter
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
            # 1. Seed Core State
            state_to_save['seed_core_state'] = { # Renamed key
                'current_goal': copy.deepcopy(self.seed_core.current_goal),
            }

            # 2. Orchestrator State
            state_to_save['orchestrator_state'] = {
                'total_cycles_run': self.total_cycles_run,
            }

            # 3. Write minimal state to restart state file
            with open(RESTART_STATE_FILE, 'wb') as f:
                pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Minimal restart state (pointers) successfully serialized to {RESTART_STATE_FILE}.")

            # 4. Ensure Full Memory is Saved Before Restart Trigger
            logger.info("Saving full memory state before triggering restart...")
            if hasattr(self, 'memory') and self.memory:
                 self.memory.save_memory() # This saves episodic, lifelong, learning state
            else:
                 logger.error("Memory object not available for pre-restart save!")
                 return False # Abort restart if memory cannot be saved

            return True

        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Failed to serialize state or save memory for restart: {e}", exc_info=True)
            if os.path.exists(RESTART_STATE_FILE):
                try:
                    os.remove(RESTART_STATE_FILE)
                    logger.warning(f"Attempted cleanup: Removed potentially corrupt restart file: {RESTART_STATE_FILE}")
                except OSError as cleanup_err:
                     logger.error(f"Attempted cleanup: Failed to remove restart file '{RESTART_STATE_FILE}': {cleanup_err}")
            return False

    def _trigger_restart(self):
        """ Triggers the restart of the main script using os.execv. """
        logger.warning("--- TRIGGERING SELF-RESTART NOW ---")
        # Disconnect services
        if hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            try: self.vm_service.disconnect(); logger.info("VM Service disconnected pre-restart.")
            except Exception as e: logger.error(f"Error disconnecting VM Service pre-restart: {e}")

        # TF Session clear (if still used)
        if 'tensorflow' in sys.modules:
            try:
                tf.keras.backend.clear_session(); gc.collect();
                logger.info("TF Session cleared and garbage collected pre-restart.")
            except Exception as e: logger.error(f"Error during TF cleanup pre-restart: {e}")
        else:
            gc.collect()
            logger.info("Garbage collected pre-restart (TF not imported/used).")


        python_executable = sys.executable
        # Execute seed.main as the module
        entry_module = 'seed.main'
        restart_args = [python_executable, '-m', entry_module, '--restarted']

        logger.info(f"Executing restart command: {' '.join(restart_args)}")
        try: os.execv(python_executable, restart_args) # Replace current process
        except Exception as e:
            logger.critical(f"FATAL: os.execv failed to restart process: {e}", exc_info=True)
            sys.exit(1)

    # Renamed method
    def _main_loop(self, max_cycles: Optional[int] = None):
        """ The main orchestrator loop, driving seed core cycles. """
        logger.info("Starting main orchestrator loop...")
        while self.is_running:
            loop_start_time = time.monotonic()

            # --- Primary Action: Trigger Seed Core Strategic Cycle ---
            logger.debug(f"Orchestrator: Triggering Seed Core Cycle {self.total_cycles_run + 1}...")
            seed_core_cycle_completed_successfully = False # Renamed variable
            try:
                # Check if seed core and its method exist
                if not self.seed_core or not hasattr(self.seed_core, 'run_strategic_cycle') or not callable(self.seed_core.run_strategic_cycle):
                    logger.critical("Seed Core object invalid or missing 'run_strategic_cycle'. Stopping.")
                    self.is_running = False; break

                # --- Execute Seed Core Cycle ---
                self.seed_core.run_strategic_cycle()
                # --- End Execute Seed Core Cycle ---

                self.total_cycles_run += 1
                seed_core_cycle_completed_successfully = True # Renamed variable
                logger.debug(f"Orchestrator: Completed Seed Core Cycle {self.total_cycles_run}")

            except Exception as cycle_err:
                 logger.error(f"Error during Seed Core strategic cycle {self.total_cycles_run + 1}: {cycle_err}", exc_info=True)
                 try:
                     # Use updated log key/tags
                     self.memory.log("SEED_CycleCriticalError", {"cycle": self.total_cycles_run + 1, "error": str(cycle_err)}, tags=['Seed','Error','Cycle','Critical'])
                 except Exception as log_err: logger.error(f"Failed log cycle error: {log_err}")

            # --- Check for Restart Signal AFTER the cycle ---
            if seed_core_cycle_completed_successfully and self._check_for_restart_signal():
                if self._serialize_state_for_restart(): # This now includes saving memory
                    self._trigger_restart()
                else:
                    logger.critical("Failed serialize state/save memory for restart. Aborting restart and stopping.")
                    self.is_running = False
                break # Exit loop after restart trigger attempt

            # --- Check Termination Condition ---
            if max_cycles is not None and self.total_cycles_run >= max_cycles:
                logger.info(f"Orchestrator: Reached max cycles ({max_cycles}). Stopping.")
                self.is_running = False

            # --- Yield/Sleep ---
            loop_elapsed = time.monotonic() - loop_start_time
            # Reduced sleep time slightly, adjust as needed
            sleep_time = max(0.01, 0.02 - loop_elapsed)
            time.sleep(sleep_time)

        logger.info("Orchestrator main loop finished.")

    def run(self, max_cycles: Optional[int] = None):
        """ Starts the main orchestrator loop. """
        if self.is_running: logger.warning("Orchestrator already running."); return

        run_type = f'Max Cycles: {max_cycles}' if max_cycles else 'Indefinite'
        logger.info(f"\n*** Starting RSIAI Seed Run ({run_type}) ***") # Updated name
        self.is_running = True; self.start_time = time.time()

        # --- Main Execution Block ---
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
        """ Stops the main orchestrator loop and performs cleanup. """
        if not self.is_running and self.start_time == 0: logger.info("Orchestrator already stopped or not fully run."); return

        run_duration = time.time() - self.start_time if self.start_time > 0 else 0
        logger.info(f"\n--- Stopping RSIAI Seed Run (Duration: {run_duration:.2f}s) ---") # Updated name
        self.is_running = False

        # Disconnect VM Service
        if hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            logger.info("Disconnecting VM Service...")
            try: self.vm_service.disconnect()
            except Exception as e: logger.error(f"Error disconnecting VM Service: {e}", exc_info=True)

        # Final Memory Save (includes learning state)
        logger.info("Performing final memory save...")
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'save_memory'):
             try:
                 self.memory.save_memory()
             except Exception as e: logger.error(f"Error during final memory save: {e}", exc_info=True)

        # TF Session clear (if still used)
        if 'tensorflow' in sys.modules:
            logger.info("Collecting garbage and clearing TF session...")
            try:
                 tf.keras.backend.clear_session(); gc.collect();
                 logger.debug("TF session cleared and GC collected.")
            except Exception as e: logger.error(f"Error during final GC/TF cleanup: {e}", exc_info=True)
        else:
             logger.info("Collecting garbage...")
             gc.collect()

        logger.info(f"RSIAI Seed Run Stopped. Total Seed Cycles: {self.total_cycles_run}") # Updated name
        self.start_time = 0.0

# --- Script Execution Block ---
if __name__ == "__main__":
    # Setup logging first if possible
    # Example basic config, consider moving to a dedicated logging setup function
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    is_restarted = '--restarted' in sys.argv
    if is_restarted: logging.getLogger(__name__).warning("--- Orchestrator performing restart ---"); time.sleep(1) # Use logger after basicConfig

    # --- TensorFlow GPU Configuration ---
    # Optional: Check if TF is imported before attempting config
    if 'tensorflow' in sys.modules:
        logger.info("--- TensorFlow Configuration ---"); logger.info(f"TF Version: {tf.__version__}")
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]; logical_gpus = tf.config.experimental.list_logical_devices('GPU'); logger.info(f"Found {len(gpus)} Physical GPUs, Configured {len(logical_gpus)} Logical GPUs (Mem Growth).")
                except RuntimeError as e: logger.error(f"GPU Memory Growth Error: {e}.")
            else: logger.info("No GPUs detected by TensorFlow.")
        except Exception as e: logger.error(f"GPU Config Error: {e}", exc_info=True)
        logger.info("-----------------------------")
    else:
        logger.info("TensorFlow not imported, skipping GPU configuration.")

    # --- Print Alignment Reminder ---
    logger.info("\n--- AGI Alignment Reminder ---"); logger.info(f"Mission: {ALIGNMENT_MISSION}"); logger.info(f"Core Logic: {ALIGNMENT_CORE_LOGIC}"); logger.info(f"Directive: {ALIGNMENT_DIRECTIVE}"); logger.info("--------------------------------")

    # --- Initialize and Run Orchestrator ---
    orchestrator = None
    try:
        # Initialize components, potentially loading state if restarting
        orchestrator = Main_Orchestrator(load_restart_state=is_restarted)

        # --- Restore State from Restart File (if restarting) ---
        if is_restarted and os.path.exists(RESTART_STATE_FILE):
             logger.info("Attempting to apply state from restart file...")
             loaded_state_components = None
             try:
                 with open(RESTART_STATE_FILE, 'rb') as f: loaded_state_components = pickle.load(f)

                 # Restore Seed Core goal (Memory is handled by MemorySystem init/load)
                 seed_core_state = loaded_state_components.get('seed_core_state')
                 if seed_core_state and isinstance(seed_core_state, dict):
                     restored_goal = seed_core_state.get('current_goal')
                     if restored_goal:
                         logger.info(f"Restoring goal from restart state: {restored_goal.get('description')}")
                         if hasattr(orchestrator.seed_core, 'set_goal'):
                             orchestrator.seed_core.set_goal(restored_goal) # Use set_goal for consistency
                         elif hasattr(orchestrator.seed_core, 'current_goal'):
                              orchestrator.seed_core.current_goal = restored_goal
                              logger.warning("Restored goal via direct attribute setting (set_goal method missing).")
                         else:
                               logger.error("Could not restore goal: Seed Core lacks set_goal method and current_goal attribute.")
                     else: logger.warning("Restart state file missing 'current_goal' in 'seed_core_state'.")
                 else: logger.warning("Restart state file missing 'seed_core_state'.")

                 # Restore Orchestrator cycle count
                 orchestrator_state = loaded_state_components.get('orchestrator_state')
                 if orchestrator_state and isinstance(orchestrator_state, dict):
                     orchestrator.total_cycles_run = orchestrator_state.get('total_cycles_run', orchestrator.total_cycles_run)
                     if hasattr(orchestrator.seed_core, 'cycle_count'):
                         orchestrator.seed_core.cycle_count = orchestrator.total_cycles_run
                     logger.info(f"Restarted orchestrator cycle count restored to {orchestrator.total_cycles_run}.")

                 # Clean up restart file after successful processing
                 try: os.remove(RESTART_STATE_FILE); logger.info(f"Removed restart state file: {RESTART_STATE_FILE}")
                 except OSError as e: logger.error(f"Failed remove restart state file post-load: {e}")

             except Exception as load_err:
                 logger.error(f"Error processing restart state file after init: {load_err}", exc_info=True)

        # Set initial goal ONLY if not restarting OR if state wasn't successfully loaded
        if not orchestrator.seed_core.current_goal: # Check if goal is still unset
             logger.info("Setting initial Seed goal...")
             # Use renamed core and method
             if hasattr(orchestrator.seed_core, 'set_initial_state'):
                 # Pass only the goal dictionary now
                 orchestrator.seed_core.set_initial_state(
                     goal=SEED_INITIAL_GOAL # Use the correct constant
                 )
             elif hasattr(orchestrator.seed_core, 'current_goal'): # Fallback
                 orchestrator.seed_core.current_goal = SEED_INITIAL_GOAL # Use the correct constant
                 logger.warning("Seed Core lacks set_initial_state method, setting current_goal attribute directly.")
             else:
                  logger.critical("Cannot set initial goal: Seed Core missing set_initial_state method and current_goal attribute.")
                  # Consider exiting if initial goal cannot be set
                  # sys.exit(1)

        # Run the orchestrator
        orchestrator.run(max_cycles=None) # Example: Run indefinitely

    except Exception as main_exception:
         logger.critical(f"FATAL ERROR during Orchestrator setup or run: {main_exception}", exc_info=True)
         if orchestrator and hasattr(orchestrator, 'stop') and callable(orchestrator.stop):
             logger.info("Attempting emergency stop...")
             try: orchestrator.stop()
             except Exception as stop_err: logger.error(f"Emergency stop failed: {stop_err}")
    finally:
        # --- Post-Run Analysis ---
        logger.info("\n--- Post-Run Analysis ---")
        if orchestrator and hasattr(orchestrator, 'memory') and orchestrator.memory:
            try:
                # Define a helper function for cleaner analysis logging
                def dump_mem_summary(label: str, filter_func: Callable[[Dict], bool], limit: int = 3, newest: bool = True):
                    try:
                        entries = orchestrator.memory.find_lifelong_by_criteria(filter_func, limit=limit, newest_first=newest)
                        # >>> FIX #4 START <<< More robust summary generation
                        summaries = []
                        for e in entries:
                            key = e.get('key', 'no_key')
                            data_part = e.get('data', {})
                            # Try getting 'message', fallback to limited json dump of data
                            try:
                                # Prioritize 'message', then try dumping the whole data dict
                                # Limit the length of the summary content
                                summary_content = data_part.get('message')
                                if summary_content is None:
                                    # Safely dump data_part, limiting length
                                    summary_content = json.dumps(data_part, default=str, ensure_ascii=False)
                                    summary_content = summary_content[:100] + ("..." if len(summary_content) > 100 else "")
                                else:
                                    # Ensure message is a string and limit length
                                    summary_content = str(summary_content)[:100] + ("..." if len(str(summary_content)) > 100 else "")

                            except Exception as summary_err:
                                logger.debug(f"Error summarizing data for key '{key}': {summary_err}") # Debug log for summary errors
                                summary_content = "[Data Summary Error]"
                            summaries.append(f"({key}: {summary_content})")
                        # >>> FIX #4 END <<<
                        logger.info(f"\n{label} (Last {len(entries)}): {', '.join(summaries)}")
                    except Exception as e:
                        # Log the error with traceback for better debugging
                        logger.error(f"Error retrieving/summarizing '{label}': {e}", exc_info=True)


                latest_epi = orchestrator.memory.get_latest_episodic(5);
                epi_summary = [f"({e.get('id', '?')}: {e.get('data',{}).get('event_type', '?')})" for e in latest_epi]
                logger.info(f"Last {len(latest_epi)} Episodic: {', '.join(epi_summary)}")

                dump_mem_summary("Last SEED Evals", lambda e: e.get('key','').startswith("SEED_Evaluation"))
                dump_mem_summary("Last SEED Decisions", lambda e: e.get('key','').startswith("SEED_Decision"))
                dump_mem_summary("Last SEED Goals", lambda e: e.get('key','').startswith("seed_goal_set"), limit=2)
                dump_mem_summary("Last CoreMod Writes", lambda e: e.get('key','').startswith("SEED_Action_MODIFY_CORE_CODE"), limit=2)
                dump_mem_summary("Last CoreMod Verifications", lambda e: e.get('key','').startswith("SEED_Action_VERIFY_CORE"), limit=2)
                dump_mem_summary("Last CoreMod Tests", lambda e: e.get('key','').startswith("SEED_Action_TEST_CORE"), limit=2)
                dump_mem_summary("Last Restarts Req", lambda e: e.get('key','').startswith(RESTART_SIGNAL_EVENT_TYPE), limit=5)
                dump_mem_summary("Recent Errors", lambda e: 'Error' in e.get('tags', []) or 'Critical' in e.get('tags',[]))

                # Log learning state summary
                current_params = orchestrator.memory.get_learning_parameter('')
                param_summary = json.dumps({cat: {p: v.get('value') for p, v in params.items()} if cat=="evaluation_weights" else params.get('value') for cat, params in current_params.items()}, indent=2)
                logger.info(f"\nFinal Learning Parameters:\n{param_summary}")
                rules = orchestrator.memory.get_behavioral_rules()
                logger.info(f"\nFinal Behavioral Rules ({len(rules)}): {list(rules.keys())}")

            except Exception as post_run_error:
                logger.error(f"Post-run analysis error: {post_run_error}", exc_info=True)
        else:
            logger.warning("Orchestrator/memory not available for post-run analysis.")
        logger.info("\n--- RSIAI Seed Execution Finished ---")

# --- END OF FILE seed/main.py ---