# seed/main.py
"""
Main entry point for the RSIAI Seed Orchestrator.
Initializes components based on configuration and runs the main control loop,
driving the Seed core strategic cycle. Includes self-restart mechanism
and manages persistence of essential state.
Optionally runs in Genetic Algorithm (GA) mode to manage a population of Seed variants.
"""
import time
import json
import os
import gc
import sys
import pickle
import copy
import traceback
import logging
import subprocess
import pathlib # <<< Added for path management
import shutil  # <<< Added for GA mode (copying codebase)
import argparse # <<< Added for command-line arguments

# Keep TF import if still potentially needed elsewhere (e.g. future internal models)
import tensorflow as tf
from typing import Optional, List, Dict, Any, Callable, Tuple
from collections import deque

# --- Configuration ---
from .config import (
    ALIGNMENT_DIRECTIVE, ALIGNMENT_CORE_LOGIC, ALIGNMENT_MISSION,
    MEMORY_SAVE_FILENAME, RESTART_STATE_FILENAME, # Use FILENAME constants
    RESTART_SIGNAL_EVENT_TYPE,
    SEED_INITIAL_GOAL,
    # GA Mode Config
    ENABLE_DARWINIAN_MODE, GA_POPULATION_SIZE, GA_MAX_GENERATIONS,
    GA_VARIANTS_BASE_DIR, GA_SEED_PROJECT_SOURCE_DIR
)

# --- Core Components ---
from .memory_system import MemorySystem
from .llm_service import Seed_LLMService
from .vm_service import Seed_VMService
from .evaluator import Seed_SuccessEvaluator
from .sensory import Seed_SensoryRefiner
from .core import Seed_Core

# Setup root logger
logger = logging.getLogger(__name__) # Standard logger for the module

# --- Constants for Self-Restart Mechanism ---
# RESTART_STATE_FILE is now instance-specific in Main_Orchestrator

# --- Main Orchestrator Class (Single Seed Instance Runner) ---
class Main_Orchestrator:
    """
    Initializes and coordinates a single RSIAI Seed instance. Runs the main execution loop,
    driving Seed Core strategic cycles, including checks for self-restart.
    Handles persistence of core state and memory for this instance.
    Can be run as a main process or as a "variant worker" in GA mode.
    """
    def __init__(self,
                 load_restart_state: bool = False,
                 variant_id: Optional[str] = None,
                 variant_base_path_str: Optional[str] = None): # <<<< MODIFIED
        """
        Initializes all core components and services.
        Args:
            load_restart_state: If True, attempts to load state from a restart file.
            variant_id: Identifier if this orchestrator is running as a GA variant.
            variant_base_path_str: Base directory for this variant's files (if applicable).
        """
        self.variant_id = variant_id
        instance_name = f"SeedVariant-{self.variant_id}" if self.variant_id else "SeedMaster"
        self.logger = logging.getLogger(f"{__name__}.{instance_name}") # Instance-specific logger

        self.logger.info(f"--- Initializing {instance_name} Orchestrator ---")
        start_time = time.time()
        self.is_restarting = load_restart_state

        # Determine base path for instance files
        if variant_base_path_str:
            self.instance_base_path = pathlib.Path(variant_base_path_str).resolve()
            self.instance_base_path.mkdir(parents=True, exist_ok=True)
        else:
            # Default run (not a variant), use current working directory or project root
            self.instance_base_path = pathlib.Path(".").resolve() # Or specify project root

        # Parameterize memory and restart file paths
        self.memory_save_file_path = self.instance_base_path / MEMORY_SAVE_FILENAME
        self.restart_state_file_path = self.instance_base_path / RESTART_STATE_FILENAME
        self.logger.info(f"Instance base path: {self.instance_base_path}")
        self.logger.info(f"Memory save file: {self.memory_save_file_path}")
        self.logger.info(f"Restart state file: {self.restart_state_file_path}")


        # Pass instance-specific memory path to MemorySystem config override
        memory_system_config = {'save_file_path': str(self.memory_save_file_path)}
        if self.variant_id and MEMORY_VECTOR_DB_PATH: # Give variants unique vector DB paths
            memory_system_config['vector_index_path'] = str(self.instance_base_path / f"_vector_db_variant_{self.variant_id}" / pathlib.Path(MEMORY_VECTOR_DB_PATH).name)
        self.memory: MemorySystem = MemorySystem(config=memory_system_config)


        loaded_state_components = None
        if self.is_restarting and self.restart_state_file_path.exists():
            self.logger.warning(f"Restart flag set. Attempting load component state from: {self.restart_state_file_path}")
            try:
                with open(self.restart_state_file_path, 'rb') as f:
                    loaded_state_components = pickle.load(f)
                self.logger.info("Successfully loaded component state from restart file.")
            except Exception as e:
                self.logger.error(f"Failed load state from restart file '{self.restart_state_file_path}': {e}. Proceeding with standard init.", exc_info=True)
                loaded_state_components = None

        self.logger.info("Initializing Seed Services/Shared Components...")
        self.llm_service: Seed_LLMService = Seed_LLMService(memory_system=self.memory)
        self.vm_service: Seed_VMService = Seed_VMService()
        self.success_evaluator: Seed_SuccessEvaluator = Seed_SuccessEvaluator()
        self.sensory_refiner: Seed_SensoryRefiner = Seed_SensoryRefiner()

        self.logger.info("Initializing Seed Core...")
        self.seed_core: Seed_Core = Seed_Core(
            llm_service=self.llm_service,
            vm_service=self.vm_service,
            memory_system=self.memory,
            success_evaluator=self.success_evaluator,
            sensory_refiner=self.sensory_refiner,
        )

        self.total_cycles_run: int = 0
        self.is_running: bool = False
        self.start_time_orchestrator: float = 0.0 # Renamed to avoid conflict

        if loaded_state_components and isinstance(loaded_state_components, dict):
            seed_core_state = loaded_state_components.get('seed_core_state')
            if seed_core_state and isinstance(seed_core_state, dict):
                 self.logger.info("Restoring Seed Core state (goal) from loaded data.")
                 initial_goal = seed_core_state.get('current_goal', {})
                 if initial_goal:
                     if hasattr(self.seed_core, 'set_initial_state'):
                          self.seed_core.set_initial_state(initial_goal)
                     elif hasattr(self.seed_core, 'current_goal'):
                          self.seed_core.current_goal = initial_goal
                          self.logger.warning("Seed Core lacks set_initial_state method, setting current_goal attribute directly.")
                     else:
                           self.logger.error("Could not restore goal: Seed Core lacks set_initial_state method and current_goal attribute.")
                 else:
                       self.logger.warning("Restart state file loaded, but missing 'current_goal' in 'seed_core_state'.")

            orchestrator_state = loaded_state_components.get('orchestrator_state')
            if orchestrator_state and isinstance(orchestrator_state, dict):
                 self.total_cycles_run = orchestrator_state.get('total_cycles_run', 0)
                 if hasattr(self.seed_core, 'cycle_count'):
                     self.seed_core.cycle_count = self.total_cycles_run
                 self.logger.info(f"Restored orchestrator cycle count to {self.total_cycles_run}.")

        init_duration = time.time() - start_time
        self.logger.info(f"--- {instance_name} Orchestrator Initialized ({init_duration:.2f}s) ---")


    def _check_for_restart_signal(self) -> bool:
        """ Checks memory for the seed restart signal event. """
        try:
            restart_signals = self.memory.find_lifelong_by_criteria(
                lambda e: e.get('key', '').startswith(RESTART_SIGNAL_EVENT_TYPE),
                limit=1, newest_first=True )
            if restart_signals:
                self.logger.warning(f"RESTART SIGNAL DETECTED in memory (Event: {restart_signals[0].get('key')}).")
                return True
        except Exception as e: self.logger.error(f"Error checking for restart signal: {e}", exc_info=True)
        return False

    def _serialize_state_for_restart(self) -> bool:
        """ Saves essential state needed for restart to the instance's restart file. """
        self.logger.info(f"Serializing essential state to restart file: {self.restart_state_file_path}")
        state_to_save = {}
        try:
            state_to_save['seed_core_state'] = {
                'current_goal': copy.deepcopy(self.seed_core.current_goal),
            }
            state_to_save['orchestrator_state'] = {
                'total_cycles_run': self.total_cycles_run,
            }
            with open(self.restart_state_file_path, 'wb') as f:
                pickle.dump(state_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.logger.info(f"Minimal restart state successfully serialized to {self.restart_state_file_path}.")

            self.logger.info("Saving full memory state before triggering restart...")
            if hasattr(self, 'memory') and self.memory:
                 self.memory.save_memory()
            else:
                 self.logger.error("Memory object not available for pre-restart save!")
                 return False

            return True
        except Exception as e:
            self.logger.critical(f"CRITICAL ERROR: Failed to serialize state or save memory for restart: {e}", exc_info=True)
            if self.restart_state_file_path.exists():
                try: os.remove(self.restart_state_file_path)
                except OSError as cleanup_err: self.logger.error(f"Failed to remove corrupt restart file '{self.restart_state_file_path}': {cleanup_err}")
            return False

    def _trigger_restart(self):
        """ Triggers a restart of this Seed instance. """
        self.logger.warning("--- TRIGGERING SELF-RESTART (via new console for Windows) ---")

        if hasattr(self, 'vm_service') and hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            try: self.vm_service.disconnect(); self.logger.info("VM Service disconnected pre-restart.")
            except Exception as e: self.logger.error(f"Error disconnecting VM Service pre-restart: {e}")

        if 'tensorflow' in sys.modules and 'tf' in globals():
            try: tf.keras.backend.clear_session(); gc.collect(); self.logger.info("TF Session cleared and garbage collected pre-restart.")
            except Exception as e: self.logger.error(f"Error during TF cleanup pre-restart: {e}")
        else:
            gc.collect(); self.logger.info("Garbage collected pre-restart (TF not imported/used or 'tf' not in globals).")

        try: sys.stdout.flush(); sys.stderr.flush()
        except Exception as e: print(f"WARNING: Error flushing standard streams pre-restart: {e}", file=sys.stderr)

        print("INFO: Shutting down logging system prior to new console restart...", file=sys.stderr)
        logging.shutdown()

        python_executable = sys.executable
        entry_module = 'seed.main'
        command_to_run = [python_executable, '-m', entry_module, '--restarted']

        # <<< MODIFIED: Add variant args if this is a variant instance >>>
        if self.variant_id:
            command_to_run.extend(['--variant-id', self.variant_id])
        if self.instance_base_path != pathlib.Path(".").resolve(): # Only add if not default CWD
            command_to_run.extend(['--variant-base-path', str(self.instance_base_path)])
        # <<< END MODIFICATION >>>

        project_root_dir = None
        try:
            project_root_dir = pathlib.Path(GA_SEED_PROJECT_SOURCE_DIR).resolve()
            if not project_root_dir.is_dir():
                print(f"WARNING: GA_SEED_PROJECT_SOURCE_DIR '{project_root_dir}' does not seem to be a valid directory for CWD. Defaulting to script parent's parent.", file=sys.stderr)
                project_root_dir = pathlib.Path(__file__).resolve().parent.parent # Fallback
        except Exception as e_path:
            print(f"WARNING: Could not determine project_root_dir for new process CWD: {e_path}. CWD will be inherited or default.", file=sys.stderr)
            project_root_dir = None

        print(f"INFO: Attempting to launch new console with: {' '.join(command_to_run)}", file=sys.stderr)
        if project_root_dir: print(f"INFO: Setting CWD for new process to: {project_root_dir}", file=sys.stderr)

        try:
            if sys.platform == "win32":
                subprocess.Popen(command_to_run, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd=project_root_dir)
            else:
                subprocess.Popen(command_to_run, cwd=project_root_dir)
            print("INFO: New process launched. Current process will now exit.", file=sys.stderr)
            sys.exit(0)
        except Exception as e:
            print(f"FATAL: Failed to launch new process for restart: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.exit(1)


    def _main_loop(self, max_cycles: Optional[int] = None):
        self.logger.info("Starting main orchestrator loop...")
        while self.is_running:
            loop_start_time = time.monotonic()
            self.logger.debug(f"Orchestrator: Triggering Seed Core Cycle {self.total_cycles_run + 1}...")
            seed_core_cycle_completed_successfully = False
            try:
                if not self.seed_core or not hasattr(self.seed_core, 'run_strategic_cycle') or not callable(self.seed_core.run_strategic_cycle):
                    self.logger.critical("Seed Core object invalid or missing 'run_strategic_cycle'. Stopping.")
                    self.is_running = False; break
                self.seed_core.run_strategic_cycle()
                self.total_cycles_run += 1
                seed_core_cycle_completed_successfully = True
                self.logger.debug(f"Orchestrator: Completed Seed Core Cycle {self.total_cycles_run}")
            except Exception as cycle_err:
                 self.logger.error(f"Error during Seed Core strategic cycle {self.total_cycles_run + 1}: {cycle_err}", exc_info=True)
                 try:
                     self.memory.log("SEED_CycleCriticalError", {"cycle": self.total_cycles_run + 1, "error": str(cycle_err)}, tags=['Seed','Error','Cycle','Critical'])
                 except Exception as log_err: self.logger.error(f"Failed log cycle error: {log_err}")

            if seed_core_cycle_completed_successfully and self._check_for_restart_signal():
                if self._serialize_state_for_restart():
                    self._trigger_restart()
                else:
                    self.logger.critical("Failed serialize state/save memory for restart. Aborting restart and stopping.")
                    self.is_running = False
                break

            if max_cycles is not None and self.total_cycles_run >= max_cycles:
                self.logger.info(f"Orchestrator: Reached max cycles ({max_cycles}). Stopping.")
                self.is_running = False

            loop_elapsed = time.monotonic() - loop_start_time
            sleep_time = max(0.01, 0.02 - loop_elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
        self.logger.info("Orchestrator main loop finished.")

    def run(self, max_cycles: Optional[int] = None):
        if self.is_running: self.logger.warning("Orchestrator already running."); return
        run_type = f'Max Cycles: {max_cycles}' if max_cycles else 'Indefinite'
        self.logger.info(f"\n*** Starting RSIAI Seed Run ({run_type}) ***")
        self.is_running = True; self.start_time_orchestrator = time.time()
        try:
             if self.is_running: self._main_loop(max_cycles=max_cycles)
        except KeyboardInterrupt: self.logger.info("\nOrchestrator: KeyboardInterrupt received. Initiating shutdown...")
        except Exception as e:
             self.logger.critical(f"Orchestrator: UNHANDLED EXCEPTION in main loop: {e}", exc_info=True)
             if hasattr(self, 'memory') and self.memory:
                 try: self.memory.log("OrchestratorError", {"error": str(e), "traceback": traceback.format_exc()}, tags=['Critical', 'Error', 'Orchestrator'])
                 except Exception as log_err: self.logger.error(f"Failed log critical orchestrator error: {log_err}")
        finally:
             if self.is_running: self.is_running = False
             self.logger.info("Orchestrator performing cleanup...")
             self.stop()

    def stop(self):
        if not self.is_running and self.start_time_orchestrator == 0 and self.total_cycles_run == 0:
            self.logger.info("Orchestrator already stopped or not fully run.")
            return
        run_duration = time.time() - self.start_time_orchestrator if self.start_time_orchestrator > 0 else 0
        self.logger.info(f"\n--- Stopping RSIAI Seed Run (Duration: {run_duration:.2f}s) ---")
        self.is_running = False

        if hasattr(self, 'vm_service') and hasattr(self.vm_service, 'disconnect') and callable(self.vm_service.disconnect):
            self.logger.info("Disconnecting VM Service...")
            try: self.vm_service.disconnect()
            except Exception as e: self.logger.error(f"Error disconnecting VM Service: {e}", exc_info=True)

        self.logger.info("Performing final memory save...")
        if hasattr(self, 'memory') and self.memory and hasattr(self.memory, 'save_memory'):
             try: self.memory.save_memory()
             except Exception as e: self.logger.error(f"Error during final memory save: {e}", exc_info=True)

        if 'tensorflow' in sys.modules and 'tf' in globals():
            self.logger.info("Collecting garbage and clearing TF session...")
            try:
                 tf.keras.backend.clear_session(); gc.collect();
                 self.logger.debug("TF session cleared and GC collected.")
            except KeyboardInterrupt:
                self.logger.warning("KeyboardInterrupt received during TensorFlow session cleanup. Skipping further TF cleanup.")
            except Exception as e: self.logger.error(f"Error during final GC/TF cleanup: {e}", exc_info=True)
        else:
             self.logger.info("Collecting garbage...")
             gc.collect()
        self.logger.info(f"RSIAI Seed Run Stopped. Total Seed Cycles: {self.total_cycles_run}")
        self.start_time_orchestrator = 0.0


# --- <<< NEW: Genetic Algorithm Orchestrator Class >>> ---
class GA_Orchestrator:
    """
    Manages a population of Seed variants for evolutionary self-improvement.
    This is a placeholder for future Darwinian Gödel Machine capabilities.
    """
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GA_Orchestrator")
        self.population_size = GA_POPULATION_SIZE
        self.max_generations = GA_MAX_GENERATIONS
        self.variants_base_dir = pathlib.Path(GA_VARIANTS_BASE_DIR).resolve()
        self.seed_project_source_dir = pathlib.Path(GA_SEED_PROJECT_SOURCE_DIR).resolve()
        self.current_population: List[Dict[str, Any]] = [] # Stores info about each variant
        self.active_processes: Dict[str, subprocess.Popen] = {} # variant_id -> Popen object
        self.global_memory: Optional[MemorySystem] = None # For GA-level events

        self.logger.info(f"GA Orchestrator Initialized: PopSize={self.population_size}, MaxGens={self.max_generations}")
        self.variants_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize a global memory for the GA orchestrator itself
        ga_mem_path = self.variants_base_dir / "ga_orchestrator_memory.pkl"
        self.global_memory = MemorySystem(config={'save_file_path': str(ga_mem_path)})
        self.global_memory.log("GA_Event", {"event": "GA_Orchestrator_Initialized"}, tags=["GeneticAlgorithm", "Init"])


    def _copy_seed_codebase(self, variant_path: pathlib.Path) -> bool:
        """Copies the base Seed project to the variant's path."""
        self.logger.info(f"Copying base Seed codebase from '{self.seed_project_source_dir}' to '{variant_path}'...")
        try:
            if variant_path.exists():
                shutil.rmtree(variant_path) # Clean up old variant dir if exists
            variant_path.mkdir(parents=True, exist_ok=True)

            # Define items to ignore during copy
            ignore_patterns = shutil.ignore_patterns(
                '.git*', '*__pycache__*', '*.pyc', '*.pyo', '.DS_Store',
                GA_VARIANTS_BASE_DIR + "*", # Don't copy the variants dir into a variant
                CORE_CODE_MODIFICATION_BACKUP_DIR, '_core_verify_temp',
                '*.log', '*.pkl', '*.bak', '.env', 'venv', '.venv', 'docs', '_build', '*.egg-info',
                # Add more specific files/dirs of the GA orchestrator if it lives in project root
            )
            # Copy to a temporary name then rename, or copy contents directly
            # shutil.copytree copies a dir *into* another if dest exists.
            # So, we copy contents of source_dir into variant_path
            for item in self.seed_project_source_dir.iterdir():
                dest_item = variant_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest_item, ignore=ignore_patterns, dirs_exist_ok=True)
                else:
                    if not shutil.ignore_patterns(*ignore_patterns.patterns)(str(self.seed_project_source_dir), [item.name]): # Check if item should be ignored
                         shutil.copy2(item, dest_item)
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy codebase for variant at '{variant_path}': {e}", exc_info=True)
            return False

    def _spawn_variant(self, variant_id: str, generation: int, initial_code_path: Optional[pathlib.Path] = None) -> Optional[Dict[str, Any]]:
        """
        Creates, configures, and launches a new Seed variant instance.
        Args:
            variant_id: Unique ID for the variant.
            generation: Current generation number.
            initial_code_path: Path to the codebase for this variant (if pre-modified).
                               If None, copies from GA_SEED_PROJECT_SOURCE_DIR.
        Returns:
            A dictionary with variant info if successful, else None.
        """
        self.logger.info(f"Spawning Variant ID: {variant_id} (Generation: {generation})")
        variant_path = self.variants_base_dir / f"variant_{variant_id}"

        if initial_code_path:
            self.logger.info(f"Using provided codebase path for variant: {initial_code_path}")
            # Assume initial_code_path is already prepared correctly in variant_path
            if not initial_code_path.is_dir() or initial_code_path != variant_path:
                self.logger.error(f"Provided initial_code_path '{initial_code_path}' is not valid or doesn't match expected variant_path '{variant_path}'.")
                # If code is elsewhere, copy it to the standard variant_path
                if not self._copy_seed_codebase(variant_path): # This will overwrite variant_path
                     return None
                # If initial_code_path was meant to be *the* source (e.g. after crossover/mutation),
                # then this copy logic needs adjustment based on how variation produces the code.
                # For now, assume variation writes *into* variant_path.
        elif not self._copy_seed_codebase(variant_path):
            return None

        # Placeholder: Modify variant's config or code here if needed (e.g., specific LLM temp, goal)
        # For now, variants will run with the copied default config from the source.
        # A real GA would modify the config or the code (e.g. core.py) in variant_path BEFORE launching.

        python_executable = sys.executable
        entry_module = 'seed.main' # Assuming running from project root where 'seed' is a module
        command_to_run = [
            python_executable, '-m', entry_module,
            '--variant-id', variant_id,
            '--variant-base-path', str(variant_path)
            # Add '--restarted' only if this spawn is a restart of an existing variant process
        ]
        self.logger.info(f"Launching variant '{variant_id}' with command: {' '.join(command_to_run)}")
        try:
            # Run from the project's main directory so module resolution works for 'seed.main'
            # The variant itself will use its own variant_base_path for its files.
            process = subprocess.Popen(command_to_run, cwd=str(self.seed_project_source_dir))
            self.active_processes[variant_id] = process
            variant_info = {
                'id': variant_id,
                'path': variant_path,
                'process_pid': process.pid,
                'generation': generation,
                'fitness': 0.0, # Initialize fitness
                'status': 'running',
                'start_time': time.time()
            }
            if self.global_memory: self.global_memory.log("GA_VariantSpawned", {"variant_id": variant_id, "generation": generation, "pid": process.pid}, tags=["GeneticAlgorithm", "Variant"])
            return variant_info
        except Exception as e:
            self.logger.error(f"Failed to launch variant '{variant_id}': {e}", exc_info=True)
            if self.global_memory: self.global_memory.log("GA_Event", {"event": "VariantSpawnFailed", "variant_id": variant_id, "error": str(e)}, tags=["GeneticAlgorithm", "Error"])
            return None

    def _initialize_population(self):
        """Creates the initial population of Seed variants."""
        self.logger.info(f"Initializing population (Size: {self.population_size})...")
        self.current_population = []
        for i in range(self.population_size):
            variant_id = f"gen0_var{i:03d}"
            variant_info = self._spawn_variant(variant_id, generation=0)
            if variant_info:
                self.current_population.append(variant_info)
            else:
                self.logger.error(f"Failed to initialize variant {variant_id}. Population may be smaller.")
        self.logger.info(f"Initialized {len(self.current_population)} variants.")

    def _evaluate_population_fitness(self):
        """Placeholder: Evaluates the fitness of each variant in the current population."""
        self.logger.info("Evaluating population fitness (Placeholder)...")
        for variant_info in self.current_population:
            if variant_info['status'] == 'running':
                # Placeholder: Read fitness from variant's memory or log file
                # For example, variant's MemorySystem could store a 'fitness_score' key
                variant_memory_file = variant_info['path'] / MEMORY_SAVE_FILENAME
                fitness = 0.0
                try:
                    if variant_memory_file.exists():
                        with open(variant_memory_file, 'rb') as f:
                            mem_data = pickle.load(f)
                        # Example: Look for a specific lifelong entry or parse evaluations
                        # This is highly dependent on how variants report their performance.
                        # For now, a random fitness for demonstration.
                        # In a real system, parse mem_data['lifelong'] for SEED_Evaluation entries
                        # or a dedicated "variant_fitness" entry.
                        fitness = round(random.random(), 3) # Placeholder
                        last_evals = [v['data'].get('overall_success', 0) for k, v in mem_data.get('lifelong', {}).items() if k.startswith("SEED_Evaluation")]
                        if last_evals:
                            fitness = round(sum(last_evals) / len(last_evals), 3) if len(last_evals) > 0 else 0.0

                    process = self.active_processes.get(variant_info['id'])
                    if process and process.poll() is not None: # Process terminated
                        variant_info['status'] = 'terminated_early'
                        fitness = -1.0 # Penalize early termination
                        self.logger.warning(f"Variant {variant_info['id']} terminated early (exit code {process.returncode}).")


                except Exception as e:
                    self.logger.error(f"Error reading fitness for variant {variant_info['id']} from {variant_memory_file}: {e}")
                    fitness = -0.5 # Penalize if fitness cannot be read

                variant_info['fitness'] = fitness
                self.logger.info(f"Variant {variant_info['id']} Fitness: {variant_info['fitness']}")
                if self.global_memory: self.global_memory.log("GA_FitnessUpdate", {"variant_id": variant_info['id'], "fitness": fitness, "status": variant_info['status']}, tags=["GeneticAlgorithm", "Fitness"])


    def _perform_selection(self) -> List[Dict[str, Any]]:
        """Placeholder: Selects variants for the next generation based on fitness."""
        self.logger.info("Performing selection (Placeholder)...")
        # Example: Tournament selection or simple top N
        sorted_population = sorted([v for v in self.current_population if v['status'] != 'terminated_early' and v['fitness'] >= 0], key=lambda x: x['fitness'], reverse=True)
        # Select top 50% as parents, ensure at least 2 if possible
        num_parents = max(2, len(sorted_population) // 2) if len(sorted_population) > 1 else len(sorted_population)
        parents = sorted_population[:num_parents]
        self.logger.info(f"Selected {len(parents)} parents for next generation.")
        return parents

    def _apply_genetic_operators(self, parents: List[Dict[str, Any]], next_generation_num: int) -> List[Dict[str,Any]]:
        """
        Placeholder: Creates new offspring from parents using crossover and mutation.
        Returns a list of paths to the *newly created and modified* codebases for offspring.
        """
        self.logger.info(f"Applying genetic operators to {len(parents)} parents (Placeholder)...")
        offspring_codebase_infos = [] # List of {'id': str, 'path': Path} for new offspring code

        if not parents:
            self.logger.warning("No parents selected, cannot create offspring.")
            return []

        target_offspring_count = self.population_size
        current_offspring_idx = 0

        while len(offspring_codebase_infos) < target_offspring_count:
            offspring_id = f"gen{next_generation_num}_var{current_offspring_idx:03d}"
            offspring_path = self.variants_base_dir / f"variant_{offspring_id}"
            current_offspring_idx += 1

            # Create a clean copy for the offspring first
            if offspring_path.exists(): shutil.rmtree(offspring_path) # Clean slate
            
            # Choose parent(s)
            p1_info = random.choice(parents)
            
            # Copy parent's code to offspring path
            self.logger.debug(f"Copying parent {p1_info['id']}'s code from {p1_info['path']} to {offspring_path} for new offspring {offspring_id}")
            if not self._copy_seed_codebase(offspring_path): # This uses base seed code, should use parent's
                 shutil.copytree(p1_info['path'], offspring_path, dirs_exist_ok=True) # Copy parent's actual code
            
            # Placeholder: Crossover
            if random.random() < GA_CROSSOVER_PROBABILITY and len(parents) > 1:
                p2_info = random.choice([p for p in parents if p['id'] != p1_info['id']] or [p1_info]) # Avoid self-crossover if possible
                self.logger.info(f"Applying Crossover (Placeholder) between {p1_info['id']} and {p2_info['id']} for offspring {offspring_id} in {offspring_path}")
                # In a real system:
                # 1. Read code files from p1_info['path'] and p2_info['path']
                # 2. Use LLM to combine them into new files within offspring_path
                #    e.g., "Take function X from core.py of parent1 and function Y from core.py of parent2, merge into new_core.py for offspring"
                # For now, we just log. The offspring currently has p1's code.

            # Placeholder: Mutation
            if random.random() < GA_MUTATION_PROBABILITY:
                self.logger.info(f"Applying Mutation (Placeholder) to offspring {offspring_id} in {offspring_path}")
                # In a real system:
                # 1. Select a file in offspring_path (e.g., core.py)
                # 2. Use LLM to propose a small change (e.g., modify a constant, tweak logic in a function)
                # 3. Apply this change to the file in offspring_path (e.g. using Seed's MODIFY_CORE_CODE logic adapted or direct write)
                # For now, we just log.

            offspring_codebase_infos.append({'id': offspring_id, 'path': offspring_path, 'generation': next_generation_num})
            if len(offspring_codebase_infos) >= self.population_size : break
        
        self.logger.info(f"Generated {len(offspring_codebase_infos)} offspring codebases.")
        return offspring_codebase_infos


    def _terminate_variant_process(self, variant_id: str):
        """Terminates a variant's process and cleans up."""
        process = self.active_processes.pop(variant_id, None)
        if process:
            self.logger.info(f"Terminating variant {variant_id} (PID: {process.pid})...")
            try:
                process.terminate() # Send SIGTERM
                process.wait(timeout=5) # Wait for graceful shutdown
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Variant {variant_id} did not terminate gracefully, sending SIGKILL.")
                process.kill()
                process.wait(timeout=2)
            except Exception as e:
                self.logger.error(f"Error terminating variant {variant_id}: {e}")
            self.logger.info(f"Variant {variant_id} process terminated (Final RC: {process.returncode}).")
            if self.global_memory: self.global_memory.log("GA_VariantTerminated", {"variant_id": variant_id, "return_code": process.returncode}, tags=["GeneticAlgorithm", "Variant"])


    def run_evolution_loop(self):
        """Main loop for the Genetic Algorithm."""
        self.logger.info("--- Starting Darwinian Evolution Loop ---")
        self._initialize_population()

        for gen in range(self.max_generations):
            self.logger.info(f"\n--- Generation {gen + 1} / {self.max_generations} ---")
            if self.global_memory: self.global_memory.log("GA_Event", {"event": "GenerationStart", "generation": gen + 1, "population_size": len(self.current_population)}, tags=["GeneticAlgorithm"])

            # 1. Evaluate Fitness (after giving variants some time to run)
            # In a real system, you'd wait for variants to complete tasks or run for a certain duration.
            # This is highly simplified.
            time.sleep(10) # Simulate variants running for a bit
            self._evaluate_population_fitness()

            # 2. Selection
            parents = self._perform_selection()
            if not parents and gen < self.max_generations -1 : # Stop if no parents and not the last generation
                self.logger.warning("No parents selected to continue evolution. Stopping GA.")
                if self.global_memory: self.global_memory.log("GA_Event", {"event": "Extinction", "generation": gen + 1}, tags=["GeneticAlgorithm", "Critical"])
                break

            # 3. Create Offspring (Variation: Crossover & Mutation - placeholders)
            offspring_codebase_infos = self._apply_genetic_operators(parents, next_generation_num=gen + 1)

            # 4. Terminate old population (or part of it, e.g., if using elitism)
            self.logger.info("Terminating current generation's active variants...")
            for old_variant in list(self.current_population): # Iterate over copy
                self._terminate_variant_process(old_variant['id'])
            self.current_population = [] # Clear old population list entries

            # 5. Spawn new generation from offspring codebases
            if offspring_codebase_infos:
                self.logger.info("Spawning new generation from offspring...")
                for offspring_info in offspring_codebase_infos:
                    new_variant_info = self._spawn_variant(
                        variant_id=offspring_info['id'],
                        generation=offspring_info['generation'],
                        initial_code_path=offspring_info['path'] # Pass the path to the (potentially) modified code
                    )
                    if new_variant_info:
                        self.current_population.append(new_variant_info)
                self.logger.info(f"Spawned {len(self.current_population)} variants for generation {gen + 1}.")
            else:
                 self.logger.warning(f"No offspring generated for generation {gen+1}. Evolution cannot continue.")
                 if self.global_memory: self.global_memory.log("GA_Event", {"event": "NoOffspring", "generation": gen + 1}, tags=["GeneticAlgorithm", "Critical"])
                 break


            # Simple check for population health
            if not self.current_population:
                self.logger.critical(f"Population extinct after generation {gen + 1}. Stopping GA.")
                if self.global_memory: self.global_memory.log("GA_Event", {"event": "Extinction", "generation": gen + 1}, tags=["GeneticAlgorithm", "Critical"])
                break
            
            if self.global_memory: self.global_memory.save_memory() # Save GA orchestrator's memory periodically

        self.logger.info("--- Darwinian Evolution Loop Finished ---")
        self.logger.info("Terminating any remaining active variants...")
        for variant in list(self.current_population):
            self._terminate_variant_process(variant['id'])
        if self.global_memory: self.global_memory.save_memory()


# --- Script Execution Block ---
if __name__ == "__main__":
    # --- Argument Parsing (for variant mode) ---
    parser = argparse.ArgumentParser(description="RSIAI Seed Orchestrator")
    parser.add_argument('--restarted', action='store_true', help='Flag indicating the script is being restarted.')
    parser.add_argument('--variant-id', type=str, default=None, help='Identifier for this instance if run as a GA variant.')
    parser.add_argument('--variant-base-path', type=str, default=None, help='Base path for variant-specific files.')
    cli_args = parser.parse_args()

    # --- Root Logger Setup (moved here to be effective earlier) ---
    # Configure the root logger before any other logger instances are created by modules
    # The level here will be the effective minimum for all handlers unless overridden
    # This formatting will apply to all log messages not captured by more specific handlers
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    script_logger = logging.getLogger(__name__) # Logger for this main script part

    # --- TensorFlow Configuration ---
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

    orchestrator: Optional[Union[Main_Orchestrator, GA_Orchestrator]] = None # For type hinting
    try:
        if ENABLE_DARWINIAN_MODE:
            script_logger.info("--- RSIAI Seed GA Orchestrator Mode Enabled ---")
            if cli_args.variant_id:
                script_logger.error("GA_Orchestrator mode should not be run with --variant-id. This indicates a misconfiguration or recursive GA launch.")
                sys.exit(1)
            orchestrator = GA_Orchestrator()
            orchestrator.run_evolution_loop() # GA mode runs its own loop
        else:
            script_logger.info("--- RSIAI Seed Single Instance Mode ---")
            if cli_args.variant_id:
                script_logger.info(f"Running as Variant ID: {cli_args.variant_id}, Base Path: {cli_args.variant_base_path}")

            orchestrator = Main_Orchestrator(
                load_restart_state=cli_args.restarted,
                variant_id=cli_args.variant_id,
                variant_base_path_str=cli_args.variant_base_path
            )

            # If this is a restarted instance, clean up its specific restart file
            if cli_args.restarted and orchestrator.restart_state_file_path.exists():
                 script_logger.info("Processing restart state file post-init (primarily for cleanup)...")
                 try:
                     if not orchestrator.seed_core.current_goal: # This check relies on seed_core being available
                         script_logger.warning("Post-init check: Seed Core goal seems not restored from restart file or was empty.")
                     os.remove(orchestrator.restart_state_file_path)
                     script_logger.info(f"Removed restart state file: {orchestrator.restart_state_file_path}")
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

            orchestrator.run(max_cycles=None) # Single instance run

    except Exception as main_exception:
         script_logger.critical(f"FATAL ERROR during Orchestrator setup or run: {main_exception}", exc_info=True)
         if orchestrator and hasattr(orchestrator, 'stop') and callable(orchestrator.stop): # For Main_Orchestrator
             script_logger.info("Attempting emergency stop...")
             try: orchestrator.stop()
             except Exception as stop_err: script_logger.error(f"Emergency stop failed: {stop_err}")
         # TODO: Add similar emergency stop for GA_Orchestrator if it has one (e.g., terminate all variants)
    finally:
        if not ENABLE_DARWINIAN_MODE and isinstance(orchestrator, Main_Orchestrator): # Post-run analysis for single instance
            script_logger.info("\n--- Post-Run Analysis (Single Instance) ---")
            if orchestrator and hasattr(orchestrator, 'memory') and orchestrator.memory:
                try:
                    def dump_mem_summary(label: str, filter_func: Callable[[Dict], bool], limit: int = 3, newest: bool = True):
                        try:
                            # Ensure orchestrator.memory is not None before calling methods on it
                            if orchestrator and orchestrator.memory:
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
                            else:
                                script_logger.warning(f"Cannot dump '{label}': Memory system not available.")
                        except Exception as e_dump:
                            script_logger.error(f"Error retrieving/summarizing '{label}': {e_dump}", exc_info=True)

                    if orchestrator and orchestrator.memory:
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
                        if current_params:
                            param_summary = json.dumps({
                                cat: {p: v.get('value') for p, v in params.items()} if isinstance(params, dict) and cat=="evaluation_weights"
                                else params.get('value') if isinstance(params, dict)
                                else params
                                for cat, params in current_params.items()
                            }, indent=2, default=str)
                            script_logger.info(f"\nFinal Learning Parameters:\n{param_summary}")
                        else:
                            script_logger.warning("Could not retrieve final learning parameters for summary.")

                        rules = orchestrator.memory.get_behavioral_rules()
                        script_logger.info(f"\nFinal Behavioral Rules ({len(rules)}): {list(rules.keys())}")
                    else:
                         script_logger.warning("Orchestrator memory not available for post-run summary.")

                except Exception as post_run_error:
                    script_logger.error(f"Post-run analysis error: {post_run_error}", exc_info=True)
            elif ENABLE_DARWINIAN_MODE and isinstance(orchestrator, GA_Orchestrator):
                script_logger.info("\n--- Post-Run Analysis (GA Mode) ---")
                script_logger.info(f"GA Orchestrator finished. Final population status (if any was running) can be found in logs.")
                if orchestrator.global_memory:
                    script_logger.info(f"GA Orchestrator global memory saved to: {orchestrator.global_memory.save_file_path}")
            else:
                script_logger.warning("Orchestrator/memory not available for post-run analysis or GA mode was not active.")
        script_logger.info("\n--- RSIAI Seed Execution Finished ---")

# --- END OF FILE seed/main.py ---