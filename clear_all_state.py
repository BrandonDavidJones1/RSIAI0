# clear_all_state.py
# Script to clear the main memory state (pkl file) and delete the restart state file.

import os
import logging
import sys
import copy # Needed for deepcopy in memory system potentially

# Ensure the current directory (RSIAI0) is in the path if running as script
# This helps Python find the 'seed' package
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration and File Paths ---
MEMORY_SAVE_FILE = None
RESTART_STATE_FILE = None
config_loaded = False
try:
    # Import the specific config variable needed
    from seed.config import MEMORY_SAVE_FILE as CONFIG_MEM_FILE
    MEMORY_SAVE_FILE = CONFIG_MEM_FILE # Store the actual path
    # Derive the restart state filename exactly like in main.py
    RESTART_STATE_FILE = MEMORY_SAVE_FILE.replace(".pkl", "_restart_state.pkl")
    logging.info(f"Using main memory file: '{MEMORY_SAVE_FILE}'")
    logging.info(f"Using restart state file: '{RESTART_STATE_FILE}'")
    config_loaded = True
except ImportError:
    logging.error("Could not import MEMORY_SAVE_FILE from seed.config.")
    logging.error("Make sure you are running this script from the 'RSIAI0' directory,")
    logging.error("and your virtual environment is active.")
    # Provide likely defaults as a fallback, adjust if your MEMORY_SAVE_FILE is different
    MEMORY_SAVE_FILE = "seed_bootstrap_memory.pkl"
    RESTART_STATE_FILE = "seed_bootstrap_memory_restart_state.pkl"
    logging.warning(f"Assuming default main memory file: '{MEMORY_SAVE_FILE}'")
    logging.warning(f"Assuming default restart state file: '{RESTART_STATE_FILE}'")
except Exception as e:
    logging.critical(f"Failed to determine filenames from config: {e}", exc_info=True)
    # Exit if config is crucial and missing
    sys.exit("Exiting due to configuration error.")

# --- Clear Main Memory State ---
print("\n--- Clearing Main Memory State ---")
if MEMORY_SAVE_FILE and config_loaded: # Proceed only if config was loaded successfully
    main_memory_cleared = False
    orchestrator_instance = None
    try:
        # Import orchestrator only when needed
        from seed.main import Main_Orchestrator
        logging.info("Initializing orchestrator to access memory system...")
        # Initialize without trying to load restart state for this operation
        # This will load the existing memory file first, which is necessary before clearing
        orchestrator_instance = Main_Orchestrator(load_restart_state=False)

        if orchestrator_instance and hasattr(orchestrator_instance, 'memory') and orchestrator_instance.memory:
            logging.info(f"Attempting to clear memory and save to '{MEMORY_SAVE_FILE}'...")
            try:
                # Call the method to clear memory and save the empty/default state
                orchestrator_instance.memory.clear_all_memory()
                logging.info(f"Success: Main memory cleared and saved to '{MEMORY_SAVE_FILE}'.")
                main_memory_cleared = True
            except Exception as clear_err:
                logging.error(f"Error during memory clearing process: {clear_err}", exc_info=True)
        else:
            logging.error("Failed to access the memory system via orchestrator.")

    except ImportError as import_err:
         logging.error(f"Could not import required modules ({import_err}). Is the path correct and venv active?")
    except Exception as init_err:
        logging.error(f"Error initializing orchestrator: {init_err}", exc_info=True)
    finally:
        # Attempt cleanup if orchestrator was created (disconnect VM etc)
        if orchestrator_instance and hasattr(orchestrator_instance, 'stop'):
             try: orchestrator_instance.stop()
             except Exception: pass # Ignore errors during stop in cleanup

        if not main_memory_cleared:
            logging.warning(f"Main memory state in '{MEMORY_SAVE_FILE}' might NOT be cleared.")
else:
     logging.error("Skipping main memory clear because config failed to load.")


# --- Delete Restart State File ---
print("\n--- Deleting Restart State File ---")
if RESTART_STATE_FILE: # Proceed even if config loading failed but we have a default guess
    try:
        logging.info(f"Attempting to delete file: {RESTART_STATE_FILE}...")
        os.remove(RESTART_STATE_FILE)
        logging.info(f"Success: File '{RESTART_STATE_FILE}' was deleted.")
    except FileNotFoundError:
        logging.info(f"File '{RESTART_STATE_FILE}' not found. No deletion needed.")
    except PermissionError:
        logging.error(f"Permission denied to delete '{RESTART_STATE_FILE}'. Check file permissions.")
    except OSError as e:
        logging.error(f"OS error deleting file '{RESTART_STATE_FILE}': {e}")
    except Exception as e:
        logging.error(f"Unexpected error during restart file deletion: {e}", exc_info=True)
else:
    logging.error("Skipping restart state file deletion because filename could not be determined.")

print("\n--- State Clearing Process Finished ---")