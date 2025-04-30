# --- START OF FILE main.py ---

# modular_agi/main.py
"""
Main entry point for the ModularAGI Orchestrator.
Initializes components based on configuration and runs the main control loop,
driving the Upper Level strategic cycle. Includes self-restart mechanism
and manages persistence of essential state including the lower_level agent pool.
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
# Keep TF import for lower_level agents
import tensorflow as tf
from typing import Optional, List, Dict, Any, Callable
from collections import deque # For loading memory state

# --- Configuration ---
# Use updated constant names
from .config import (
    ALIGNMENT_DIRECTIVE, ALIGNMENT_CORE_LOGIC, ALIGNMENT_MISSION,
    MEMORY_SAVE_FILE, RESTART_SIGNAL_EVENT_TYPE, # Keep restart signal name
    SEED_INITIAL_GOAL # Use updated goal constant
)

# --- Core Components ---
# Use updated names/paths
from .memory_system import MemorySystem
# Upper Level (Renamed) Components
from .upper_level.llm_service import UpperLevel_LLMService
from .upper_level.vm_service import UpperLevel_VMService
from .upper_level.evaluator import UpperLevel_SuccessEvaluator
from .upper_level.sensory import UpperLevel_SensoryRefiner
from .upper_level.core import UpperLevel_Core # Renamed Upper Level Core class

# Setup root logger
logger = logging.getLogger(__name__)

# --- Constants for Self-Restart Mechanism ---
# RESTART_SIGNAL_EVENT_TYPE kept from config
RESTART_STATE_FILE = MEMORY_SAVE_FILE.replace(".pkl", "_restart_state.pkl")

# --- Main Orchestrator Class ---
# Renamed class
class Main_Orchestrator:
    """
    Initializes and coordinates the ModularAGI system. Runs the main execution loop,
    driving Upper Level strategic cycles, including checks for self-restart.
    Handles persistence of upper_level state, memory, and the lower_level agent pool.
    """
    def __init__(self, load_restart_state: bool = False):
        """ Initializes all core components and services based on config. """
        # Updated log message
        logger.info("--- Initializing Main Orchestrator ---")
        start_time = time.time()
        self.is_restarting = load_restart_state # Flag if this is a restart run

        # Initialize Memory System (loads standard memory file if it exists)
        # MemorySystem now handles agent pool state loading internally via load_memory()
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

        # Initialize Upper Level Services & Helper Components
        # Use updated class names
        logger.info("Initializing Upper Level Services/Shared Components...")
        self.llm_service: UpperLevel_LLMService = UpperLevel_LLMService()
        self.vm_service: UpperLevel_VMService = UpperLevel_VMService()
        self.success_evaluator: UpperLevel_SuccessEvaluator = UpperLevel_SuccessEvaluator()
        self.sensory_refiner: UpperLevel_SensoryRefiner = UpperLevel_SensoryRefiner()

        # Initialize Upper Level Core (Strategic decision-maker)
        # Use updated class name
        logger.info("Initializing Upper Level Core...")
        # Renamed internal variable, pass updated component instances
        self.upper_level_core: UpperLevel_Core = UpperLevel_Core(
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

        # --- Restore Orchestrator/UpperLevel State from Loaded Data (if restarting) ---
        # Memory and Agent Pool are loaded separately after init
        if loaded_state_components and isinstance(loaded_state_components, dict):
            # Restore UpperLevel state (Goal, Base LowerLevel Config)
            # Use updated state key
            upper_level_state = loaded_state_components.get('upper_level_state')
            if upper_level_state and isinstance(upper_level_state, dict):
                 # Updated log message
                 logger.info("Restoring UpperLevel state (goal, lower_level config) from loaded data.")
                 initial_goal = upper_level_state.get('current_goal', {})
                 # Use updated key name
                 initial_lower_level_config = upper_level_state.get('lower_level_base_config', {})
                 # Use renamed core and method
                 if hasattr(self.upper_level_core, 'set_initial_state'):
                      self.upper_level_core.set_initial_state(initial_goal, initial_lower_level_config)
                 else: # Fallback if method missing
                      self.upper_level_core.current_goal = initial_goal
                      self.upper_level_core.lower_level_base_config = initial_lower_level_config
                      logger.warning("UpperLevel Core lacks set_initial_state method, setting attributes directly.")

            # Restore Orchestrator state
            orchestrator_state = loaded_state_components.get('orchestrator_state')
            if orchestrator_state and isinstance(orchestrator_state, dict):
                 self.total_cycles_run = orchestrator_state.get('total_cycles_run', 0)
                 # Sync cycle count if UpperLevel Core keeps its own internal counter
                 # Use renamed core
                 if hasattr(self.upper_level_core, 'cycle_count'):
                     self.upper_level_core.cycle_count = self.total_cycles_run
                 logger.info(f"Restored orchestrator cycle count to {self.total_cycles_run}.")

            # Load Lower Level Agent Pool State via UpperLevel Core
            # Use updated state key
            agent_pool_state_loaded = loaded_state_components.get('agent_pool_state')
            if agent_pool_state_loaded is not None:
                 # Use renamed core and method
                 if hasattr(self.upper_level_core, 'load_agent_pool_state') and callable(self.upper_level_core.load_agent_pool_state):
                     # Updated log message
                     logger.info("Loading LowerLevel Agent Pool state into UpperLevel Core...")
                     load_success = self.upper_level_core.load_agent_pool_state(agent_pool_state_loaded)
                     # Updated log message
                     if load_success: logger.info("LowerLevel Agent Pool state successfully loaded.")
                     else: logger.error("UpperLevel Core reported failure loading agent pool state.")
                 else:
                     # Updated log message
                     logger.error("Cannot load Agent Pool: UpperLevel Core missing 'load_agent_pool_state' method.")
            else:
                 # Updated log message
                 logger.warning("Restart state file missing 'agent_pool_state'. UpperLevel Core will initialize a new pool if needed.")

        init_duration = time.time() - start_time
        # Updated log message
        logger.info(f"--- Main Orchestrator Initialized ({init_duration:.2f}s) ---")


    def _check_for_restart_signal(self) -> bool:
        """ Checks memory for the upper_level restart signal event. """
        try:
            # Use updated constant name
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
            # 1. UpperLevel State
            # Use updated state key and internal var names
            state_to_save['upper_level_state'] = {
                'current_goal': copy.deepcopy(self.upper_level_core.current_goal),
                'lower_level_base_config': copy.deepcopy(self.upper_level_core.lower_level_base_config),
            }

            # 2. Memory System State (Core data only, agent pool saved separately)
            logger.info("Saving core memory data structures for restart...")
            state_to_save['memory_system_state'] = {
                 'episodic': list(self.memory._episodic_memory),
                 'lifelong': self.memory._lifelong_memory,
                 'lifelong_keys': list(self.memory._lifelong_keys_by_age)
            }

            # 3. Orchestrator State
            state_to_save['orchestrator_state'] = {
                'total_cycles_run': self.total_cycles_run,
            }

            # 4. Lower Level Agent Pool State
            # Updated log message
            logger.info("Requesting LowerLevel Agent Pool state from UpperLevel Core...")
            # Use renamed core and method
            if hasattr(self.upper_level_core, 'get_agent_pool_state') and callable(self.upper_level_core.get_agent_pool_state):
                agent_pool_state = self.upper_level_core.get_agent_pool_state()
                if agent_pool_state is not None:
                    # Use updated state key
                    state_to_save['agent_pool_state'] = agent_pool_state
                    # Updated log message
                    logger.info(f"Received LowerLevel Agent Pool state (Type: {type(agent_pool_state)}).")
                else:
                    # Updated log message
                    logger.warning("UpperLevel Core returned None for agent pool state.")
                    state_to_save['agent_pool_state'] = None
            else:
                # Updated log message
                logger.error("CRITICAL: UpperLevel Core missing 'get_agent_pool_state' method needed for restart!")
                state_to_save['agent_pool_state'] = None

            # 5. Write to restart state file
            with open(RESTART_STATE_FILE, 'wb') as f:
                pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Essential state successfully serialized to {RESTART_STATE_FILE}.")
            return True

        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Failed to serialize state for restart: {e}", exc_info=True)
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
        if hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            try: self.vm_service.disconnect(); logger.info("VM Service disconnected pre-restart.")
            except Exception as e: logger.error(f"Error disconnecting VM Service pre-restart: {e}")

        # TF Session clear remains for lower_level agents
        try:
            tf.keras.backend.clear_session(); gc.collect();
            logger.info("TF Session cleared and garbage collected pre-restart.")
        except Exception as e: logger.error(f"Error during cleanup pre-restart: {e}")

        python_executable = sys.executable
        entry_module = 'modular_agi.main' # Assuming execution as module
        restart_args = [python_executable, '-m', entry_module, '--restarted']

        logger.info(f"Executing restart command: {' '.join(restart_args)}")
        try: os.execv(python_executable, restart_args) # Replace current process
        except Exception as e:
            logger.critical(f"FATAL: os.execv failed to restart process: {e}", exc_info=True)
            sys.exit(1)

    # Renamed method
    def _main_loop(self, max_cycles: Optional[int] = None):
        """ The main orchestrator loop, driving upper_level cycles. """
        logger.info("Starting main orchestrator loop...")
        while self.is_running:
            loop_start_time = time.monotonic()

            # --- Primary Action: Trigger Upper Level Strategic Cycle ---
            # Updated log message
            logger.debug(f"Orchestrator: Triggering UpperLevel Cycle {self.total_cycles_run + 1}...")
            # Renamed variable
            upper_level_cycle_completed_successfully = False
            try:
                # Check if upper_level core and its method exist
                # Use renamed core and method
                if not self.upper_level_core or not hasattr(self.upper_level_core, 'run_strategic_cycle') or not callable(self.upper_level_core.run_strategic_cycle):
                    # Updated log message
                    logger.critical("Upper Level Core object invalid or missing 'run_strategic_cycle'. Stopping.")
                    self.is_running = False; break

                # --- Execute Upper Level Cycle ---
                # Use renamed core and method
                self.upper_level_core.run_strategic_cycle()
                # --- End Execute Upper Level Cycle ---

                self.total_cycles_run += 1
                # Use renamed variable
                upper_level_cycle_completed_successfully = True
                # Updated log message
                logger.debug(f"Orchestrator: Completed UpperLevel Cycle {self.total_cycles_run}")

            except Exception as cycle_err:
                 # Updated log message
                 logger.error(f"Error during UpperLevel strategic cycle {self.total_cycles_run + 1}: {cycle_err}", exc_info=True)
                 try: self.memory.log("UPPER_LEVEL_CycleError", {"cycle": self.total_cycles_run + 1, "error": str(cycle_err)}, tags=['UpperLevel','Error','Cycle']) # Updated key/tag
                 except Exception as log_err: logger.error(f"Failed log cycle error: {log_err}")

            # --- Check for Restart Signal AFTER the cycle ---
            # Use renamed variable
            if upper_level_cycle_completed_successfully and self._check_for_restart_signal():
                if self._serialize_state_for_restart():
                    self._trigger_restart()
                else:
                    logger.critical("Failed serialize state for restart. Aborting restart and stopping.")
                    self.is_running = False
                break

            # --- Check Termination Condition ---
            if max_cycles is not None and self.total_cycles_run >= max_cycles:
                logger.info(f"Orchestrator: Reached max cycles ({max_cycles}). Stopping.")
                self.is_running = False

            # --- Yield/Sleep ---
            loop_elapsed = time.monotonic() - loop_start_time
            sleep_time = max(0.01, 0.05 - loop_elapsed)
            time.sleep(sleep_time)

        logger.info("Orchestrator main loop finished.")

    def run(self, max_cycles: Optional[int] = None):
        """ Starts the main orchestrator loop. """
        if self.is_running: logger.warning("Orchestrator already running."); return

        run_type = f'Max Cycles: {max_cycles}' if max_cycles else 'Indefinite'
        logger.info(f"\n*** Starting ModularAGI Run ({run_type}) ***")
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
        logger.info(f"\n--- Stopping ModularAGI Run (Duration: {run_duration:.2f}s) ---")
        self.is_running = False

        # --- CRITICAL FIX: Save Agent Pool State before final memory save ---
        agent_pool_state = None
        if hasattr(self, 'upper_level_core') and self.upper_level_core:
            # Use renamed core and method
            if hasattr(self.upper_level_core, 'get_agent_pool_state') and callable(self.upper_level_core.get_agent_pool_state):
                try:
                    # Updated log message
                    logger.info("Retrieving final lower_level agent pool state for saving...")
                    agent_pool_state = self.upper_level_core.get_agent_pool_state()
                    # Update memory system's copy before saving memory
                    if agent_pool_state is not None:
                        self.memory.update_agent_pool_state(agent_pool_state)
                        # Updated log message
                        logger.info(f"Agent pool state (Size: {len(agent_pool_state)}) updated in MemorySystem for final save.")
                    else:
                        logger.warning("UpperLevel Core returned None for agent pool state. Not saving pool.")
                except Exception as pool_err:
                    logger.error(f"Error getting agent pool state during stop: {pool_err}", exc_info=True)
            else:
                logger.warning("UpperLevel Core does not have 'get_agent_pool_state' method. Cannot save agent pool.")
        # --- End Agent Pool Save Logic ---


        if hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            logger.info("Disconnecting VM Service...")
            try: self.vm_service.disconnect()
            except Exception as e: logger.error(f"Error disconnecting VM Service: {e}", exc_info=True)

        # Final Memory Save (Now includes agent pool state)
        logger.info("Performing final memory save (incl. agent pool state)...")
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'save_memory'):
             try:
                 self.memory.save_memory() # save_memory now saves the pool state held by memory
             except Exception as e: logger.error(f"Error during final memory save: {e}", exc_info=True)

        # TF Session clear for lower_level agents
        logger.info("Collecting garbage...")
        try:
             tf.keras.backend.clear_session(); gc.collect();
             logger.debug("TF session cleared and GC collected.")
        except Exception as e: logger.error(f"Error during final GC cleanup: {e}", exc_info=True)

        # Updated log message
        logger.info(f"ModularAGI Run Stopped. Total UpperLevel Cycles: {self.total_cycles_run}")
        self.start_time = 0.0

# --- Script Execution Block ---
if __name__ == "__main__":
    is_restarted = '--restarted' in sys.argv
    if is_restarted: logger.warning("--- Orchestrator performing restart ---"); time.sleep(1)

    # --- TensorFlow GPU Configuration ---
    logger.info("--- TensorFlow Configuration ---"); logger.info(f"TF Version: {tf.__version__}")
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]; logical_gpus = tf.config.experimental.list_logical_devices('GPU'); logger.info(f"Found {len(gpus)} Physical GPUs, Configured {len(logical_gpus)} Logical GPUs (Mem Growth).")
            except RuntimeError as e: logger.error(f"GPU Memory Growth Error: {e}.")
        else: logger.info("No GPUs detected by TensorFlow.")
    except Exception as e: logger.error(f"GPU Config Error: {e}", exc_info=True)
    logger.info("-----------------------------")

    # --- Print Alignment Reminder ---
    logger.info("\n--- AGI Alignment Reminder ---"); logger.info(f"Mission: {ALIGNMENT_MISSION}"); logger.info(f"Core Logic: {ALIGNMENT_CORE_LOGIC}"); logger.info(f"Directive: {ALIGNMENT_DIRECTIVE}"); logger.info("--------------------------------")

    # --- Initialize and Run Orchestrator ---
    orchestrator = None
    try:
        # Initialize components, potentially loading upper_level/Orchestrator state if restarting
        # Use renamed class
        orchestrator = Main_Orchestrator(load_restart_state=is_restarted)

        # --- Load Memory Data from Restart File (Agent Pool loaded by orchestrator __init__) ---
        if is_restarted and os.path.exists(RESTART_STATE_FILE):
             logger.info("Loading memory data from restart file into existing instance...")
             loaded_state_components = None
             try:
                 with open(RESTART_STATE_FILE, 'rb') as f: loaded_state_components = pickle.load(f)

                 # Load Memory (Episodic/Lifelong only, Agent Pool loaded via orchestrator init now)
                 mem_state = loaded_state_components.get('memory_system_state')
                 if mem_state and isinstance(mem_state, dict):
                     orchestrator.memory._episodic_memory = deque(mem_state.get('episodic', []), maxlen=orchestrator.memory.max_episodic_size)
                     orchestrator.memory._lifelong_memory = mem_state.get('lifelong', {})
                     loaded_keys = mem_state.get('lifelong_keys', [])
                     valid_keys = [k for k in loaded_keys if k in orchestrator.memory._lifelong_memory]
                     orchestrator.memory._lifelong_keys_by_age = deque(valid_keys, maxlen=orchestrator.memory.max_lifelong_size)
                     logger.info(f"MemorySystem state restored ({len(orchestrator.memory._episodic_memory)} ep, {len(orchestrator.memory._lifelong_memory)} ll).")
                     # Rebuild vector index if needed
                     if orchestrator.memory.vector_search_enabled:
                          logger.info("Rebuilding vector index after loading restart state memory...")
                          orchestrator.memory._rebuild_vector_index()
                 else: logger.error("Restart state file missing 'memory_system_state'.")

                 # Clean up restart file after successful load
                 try: os.remove(RESTART_STATE_FILE); logger.info(f"Removed restart state file: {RESTART_STATE_FILE}")
                 except OSError as e: logger.error(f"Failed remove restart state file post-load: {e}")

             except Exception as load_err: logger.error(f"Error processing restart state file after init: {load_err}", exc_info=True)

        # Set initial goal/config if not restarting or if state wasn't loaded
        # Use renamed core and goal constant
        if not is_restarted or not orchestrator.upper_level_core.current_goal:
             # Use updated goal constant name
             logger.info("Setting initial UpperLevel goal and LowerLevel config...")
             # Use renamed core and method
             if hasattr(orchestrator.upper_level_core, 'set_initial_state'):
                 orchestrator.upper_level_core.set_initial_state(
                     goal=UPPER_LEVEL_INITIAL_GOAL, # Updated constant
                     lower_level_config={} # Start with empty base lower_level config
                 )
             else: # Fallback
                 orchestrator.upper_level_core.current_goal = UPPER_LEVEL_INITIAL_GOAL
                 orchestrator.upper_level_core.lower_level_base_config = {}
                 logger.warning("UpperLevel Core lacks set_initial_state method, setting attributes directly.")

        # Run the orchestrator
        orchestrator.run(max_cycles=None) # Example: Run indefinitely

    except Exception as main_exception:
         logger.critical(f"FATAL ERROR during Orchestrator setup or run: {main_exception}", exc_info=True)
         if orchestrator and hasattr(orchestrator, 'stop') and callable(orchestrator.stop):
             logger.info("Attempting emergency stop...")
             try: orchestrator.stop()
             except Exception as stop_err: logger.error(f"Emergency stop failed: {stop_err}")
    finally:
        # --- Post-Run Analysis (Updated Memory Keys) ---
        logger.info("\n--- Post-Run Analysis ---")
        if orchestrator and hasattr(orchestrator, 'memory') and orchestrator.memory:
            try:
                def dump_mem(label: str, filter_func: Callable[[Dict], bool], limit: int = 3, newest: bool = True):
                    try: data=orchestrator.memory.find_lifelong_by_criteria(filter_func,limit=limit,newest_first=newest); logger.info(f"\n{label} ({len(data)}):\n{json.dumps(data,indent=2,default=str)}")
                    except Exception as e: logger.error(f"Error retrieving '{label}': {e}")

                latest_epi = orchestrator.memory.get_latest_episodic(5); logger.info(f"Last {len(latest_epi)} Episodic:\n{json.dumps(latest_epi, indent=2, default=str)}")
                # Use updated keys
                dump_mem("Last UpperLevel Evals", lambda e: e.get('key','').startswith("UPPER_LEVEL_Evaluation"))
                dump_mem("Last UpperLevel Decisions", lambda e: e.get('key','').startswith("UPPER_LEVEL_Decision"))
                dump_mem("Last LowerLevel Summaries", lambda e: e.get('key','').startswith("LOWER_LEVEL_Summary"), limit=2)
                dump_mem("Last LowerLevel Sequences", lambda e: e.get('key','').startswith("LOWER_LEVEL_SuccessfulSequence"), limit=2)
                dump_mem("Last UpperLevel Goals", lambda e: e.get('key','').startswith("upper_level_goal_set"), limit=2)
                dump_mem("Last UpperLevel FS Writes", lambda e: e.get('key','').startswith("UPPER_LEVEL_Action_WRITE_FILE"), limit=2)
                dump_mem("Last UpperLevel Restarts", lambda e: e.get('key','').startswith("UPPER_LEVEL_Action_REQUEST_RESTART"), limit=2)
                dump_mem("Recent Errors", lambda e: 'Error' in e.get('tags', []) or 'Critical' in e.get('tags',[]))
                dump_mem("Restart Signals", lambda e: e.get('key','').startswith(RESTART_SIGNAL_EVENT_TYPE), limit=5)
            except Exception as post_run_error: logger.error(f"Post-run analysis error: {post_run_error}", exc_info=True)
        else: logger.warning("Orchestrator/memory not available for post-run analysis.")
        logger.info("\n--- ModularAGI Execution Finished ---")

# --- END OF FILE main.py ---