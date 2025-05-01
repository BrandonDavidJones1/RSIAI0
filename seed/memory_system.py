# --- START OF FILE seed/memory_system.py ---

# seed/memory_system.py
"""
Defines the MemorySystem class for storing episodic and lifelong memories,
Seed learning parameters, and behavioral rules.
Handles persistence via pickling. Vector search capabilities are optional but disabled in current config.
"""
import time
import copy
import json
import uuid
import os
import pickle
import traceback
import logging
import contextlib
from collections import deque
import numpy as np
import math
import re
import pathlib # Added for save_memory directory creation
from typing import Dict, Any, Optional, List, Union, Deque, Tuple, Callable

# Import config constants (updated for RSIAI Seed)
# Now using relative import as config.py is in the same package
from .config import (
    MEMORY_MAX_EPISODIC_SIZE, MEMORY_MAX_LIFELONG_SIZE, MEMORY_SAVE_FILE,
    MEMORY_LIFELONG_EVENT_TYPES, MEMORY_LIFELONG_TAGS,
    MEMORY_ENABLE_VECTOR_SEARCH, # Flag to control vector search inclusion (currently False)
    # Conditionally import vector config ONLY if enabled (currently won't import)
    MEMORY_VECTOR_DB_PATH, MEMORY_VECTOR_DIM, MEMORY_VECTOR_AUTO_SAVE_INTERVAL,
    # Import Seed learning parameter defaults
    SEED_LEARNING_PARAMETERS
)

logger = logging.getLogger(__name__)

# Vector Search Libraries (Optional - only import if enabled)
VECTOR_SEARCH_ENABLED_CONFIG = MEMORY_ENABLE_VECTOR_SEARCH # Should be False from config
if VECTOR_SEARCH_ENABLED_CONFIG:
    try:
        import faiss
        import torch # Added for torch.no_grad() context
        from sentence_transformers import SentenceTransformer
        VECTOR_SEARCH_LIBS_AVAILABLE = True
        logger.warning("Vector search is enabled in config, but libraries might not be used if RSIAI core logic doesn't require it.")
    except ImportError:
        faiss = None
        torch = None # type: ignore
        SentenceTransformer = None
        VECTOR_SEARCH_LIBS_AVAILABLE = False
        logging.getLogger(__name__).warning("FAISS, sentence-transformers or torch not found, but vector search was enabled in config. DISABLING vector search.")
else:
    faiss = None
    torch = None # type: ignore
    SentenceTransformer = None
    VECTOR_SEARCH_LIBS_AVAILABLE = False


# Type Aliases
MemoryEntry = Dict[str, Any]
FilterFunc = Callable[[MemoryEntry], bool]
BehavioralRule = Dict[str, Any] # {rule_id: str, trigger_pattern: dict, suggested_response: str, ...}


# Vector batching constants (only relevant if enabled)
VECTOR_INDEX_BATCH_UPDATE_SIZE = 50
VECTOR_INDEX_BATCH_UPDATE_INTERVAL_SEC = 10.0

class MemorySystem:
    """
    Manages episodic/lifelong memory, Seed learning parameters/rules,
    persistence, and optional vector search.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """ Initializes the MemorySystem, loads data, and sets up vector search if enabled. """
        cfg = config if config else {}
        logger.info("Initializing MemorySystem...")

        self.max_episodic_size: int = cfg.get('max_episodic_size', MEMORY_MAX_EPISODIC_SIZE)
        self._episodic_memory: Deque[MemoryEntry] = deque(maxlen=self.max_episodic_size)
        self.max_lifelong_size: int = cfg.get('max_lifelong_size', MEMORY_MAX_LIFELONG_SIZE)
        self._lifelong_memory: Dict[str, MemoryEntry] = {}
        self._lifelong_keys_by_age: Deque[str] = deque(maxlen=self.max_lifelong_size)
        self.save_file_path: str = cfg.get('save_file_path', MEMORY_SAVE_FILE)

        # <<<< Seed Learning State >>>>
        self._learning_params_key = "seed_learning_parameters_state"
        self._behavioral_rules_key = "seed_behavioral_rules_state"
        # Initialize with defaults from config, load_memory will overwrite if file exists
        self._learning_parameters: Dict = copy.deepcopy(SEED_LEARNING_PARAMETERS)
        self._behavioral_rules: Dict[str, BehavioralRule] = {} # rule_id -> rule_data

        # Vector Search Setup - Gated by config AND library availability
        self.vector_search_enabled: bool = VECTOR_SEARCH_ENABLED_CONFIG and VECTOR_SEARCH_LIBS_AVAILABLE
        self.vector_index: Optional['faiss.Index'] = None
        self.vector_embedding_model: Optional['SentenceTransformer'] = None
        self.vector_index_path: Optional[str] = None
        self.vector_dimension: Optional[int] = None
        self.vector_id_map: Dict[int, str] = {}
        self._vector_index_dirty_counter: int = 0
        self._vector_auto_save_interval: Optional[int] = None
        self._pending_vector_adds: Dict[str, List[float]] = {}
        self._pending_vector_removals: set[str] = set()
        self._last_vector_batch_update_time: float = time.monotonic()

        if self.vector_search_enabled:
            logger.info("Vector search enabled in config and libraries found. Initializing...")
            self.vector_index_path = cfg.get('vector_index_path', MEMORY_VECTOR_DB_PATH)
            self.vector_dimension = cfg.get('vector_dimension', MEMORY_VECTOR_DIM)
            self._vector_auto_save_interval = cfg.get('vector_save_interval', MEMORY_VECTOR_AUTO_SAVE_INTERVAL)

            if not self.vector_index_path or not self.vector_dimension or not self._vector_auto_save_interval:
                 logger.error("Vector search enabled, but required config (DB_PATH, DIM, AUTO_SAVE_INTERVAL) is missing or None. DISABLING vector search.")
                 self.vector_search_enabled = False
            else:
                try:
                    model_name = cfg.get('vector_embedding_model', 'all-MiniLM-L6-v2')
                    logger.info(f"Loading sentence transformer '{model_name}' for vector search...")
                    self.vector_embedding_model = SentenceTransformer(model_name)
                    model_dim = self.vector_embedding_model.get_sentence_embedding_dimension()
                    if model_dim != self.vector_dimension:
                         logger.warning(f"Config MEMORY_VECTOR_DIM ({self.vector_dimension}) != Model dim ({model_dim}). Using model dim: {model_dim}")
                         self.vector_dimension = model_dim
                    logger.info(f"Embedding model loaded. Dimension: {self.vector_dimension}")
                    self._initialize_vector_index(force_create=False)
                except Exception as e:
                    logger.error(f"Failed to initialize vector search components: {e}", exc_info=True)
                    self.vector_search_enabled = False # Disable if init fails
        else:
            logger.info("Vector search disabled (by config or missing libraries).")

        logger.info(f"MemorySystem Init Complete. Save='{self.save_file_path}', Vector Search: {'Enabled' if self.vector_search_enabled else 'Disabled'}")
        # Load memory state AND learning state
        self.load_memory()

    # --- Vector Search Methods (Conditional - Currently Disabled by Config) ---
    def _initialize_vector_index(self, force_create: bool = False):
        """ Initializes or loads the FAISS index and ID map (if vector search enabled). """
        if not self.vector_search_enabled or not self.vector_index_path or faiss is None:
            return False

        map_path = self.vector_index_path + ".map.pkl"
        index_exists = os.path.exists(self.vector_index_path); map_exists = os.path.exists(map_path)
        should_load = not force_create and index_exists and map_exists

        try:
            if should_load:
                logger.info(f"Loading FAISS index from {self.vector_index_path}...")
                self.vector_index = faiss.read_index(self.vector_index_path)
                with open(map_path, 'rb') as f: self.vector_id_map = pickle.load(f)
                logger.info(f"Loaded FAISS index ({self.vector_index.ntotal} entries) & map ({len(self.vector_id_map)} entries).")
                if not isinstance(self.vector_index, faiss.Index) or not hasattr(self.vector_index, 'd'): raise ValueError("Loaded FAISS index object is invalid.")
                if self.vector_index.d != self.vector_dimension: logger.warning(f"Loaded index dim ({self.vector_index.d}) != config/model dim ({self.vector_dimension}). Rebuilding."); self._rebuild_vector_index(); return True
                if self.vector_index.ntotal != len(self.vector_id_map): logger.warning(f"FAISS index size ({self.vector_index.ntotal}) != ID map size ({len(self.vector_id_map)}). Rebuilding."); self._rebuild_vector_index(); return True
                return True
            else:
                logger.info("Creating new FAISS index (IndexIDMap wrapper around IndexFlatL2).")
                # Ensure vector_dimension is not None before creating index
                if self.vector_dimension is None:
                    logger.error("Cannot create FAISS index: vector_dimension is None.")
                    self.vector_search_enabled = False
                    return False
                cpu_index = faiss.IndexFlatL2(self.vector_dimension); self.vector_index = faiss.IndexIDMap(cpu_index)
                self.vector_id_map = {}; self._vector_index_dirty_counter = 0
                self._pending_vector_adds.clear(); self._pending_vector_removals.clear()
                # Corrected indentation for try...except blocks
                if force_create:
                    if index_exists:
                        try:
                            os.remove(self.vector_index_path)
                            logger.info(f"Removed old index file: {self.vector_index_path}")
                        except Exception as e_rm:
                            logger.warning(f"Could not remove old index file '{self.vector_index_path}': {e_rm}")
                    if map_exists:
                        try:
                            os.remove(map_path)
                            logger.info(f"Removed old map file: {map_path}")
                        except Exception as e_rm:
                            logger.warning(f"Could not remove old map file '{map_path}': {e_rm}")
                return True
        except Exception as e:
            logger.error(f"Error initializing/loading FAISS index: {e}", exc_info=True)
            self.vector_search_enabled = False; return False

    def _rebuild_vector_index(self):
        """ Rebuilds the entire FAISS index from current lifelong memory entries (if vector search enabled). """
        if not self.vector_search_enabled or self.vector_dimension is None: return
        logger.warning("Rebuilding FAISS index from lifelong memory...")
        if not self._initialize_vector_index(force_create=True):
            logger.error("Failed to create a new index during rebuild process. Aborting rebuild."); self.vector_search_enabled = False; return

        keys_to_index = list(self._lifelong_keys_by_age); embeddings_list: List[np.ndarray] = []; faiss_ids: List[int] = []; new_id_map: Dict[int, str] = {}; next_faiss_id = 0
        for key in keys_to_index:
             entry = self._lifelong_memory.get(key);
             if not entry: continue
             # Don't index internal state entries
             if key in [self._learning_params_key, self._behavioral_rules_key]: continue
             embedding = self._get_or_create_embedding(entry)
             if embedding is not None and len(embedding) == self.vector_dimension:
                 embeddings_list.append(np.array(embedding, dtype=np.float32))
                 current_id = next_faiss_id; faiss_ids.append(current_id); new_id_map[current_id] = key; next_faiss_id += 1
             else: logger.debug(f"Skipping entry '{key}' during index rebuild (invalid/missing embedding or internal state).")

        if embeddings_list:
            try:
                embeddings_np = np.vstack(embeddings_list).astype('float32'); faiss_ids_np = np.array(faiss_ids).astype('int64')
                if not self.vector_index: raise RuntimeError("Vector index is None during rebuild add.")
                self.vector_index.add_with_ids(embeddings_np, faiss_ids_np); self.vector_id_map = new_id_map
                logger.info(f"FAISS index rebuilt with {self.vector_index.ntotal} entries.")
                self._vector_index_dirty_counter = 0; self._pending_vector_adds.clear(); self._pending_vector_removals.clear()
                self._save_vector_index()
            except Exception as e: logger.error(f"Error adding vectors during index rebuild: {e}", exc_info=True); self.vector_search_enabled = False
        else: logger.info("No valid embeddings found in lifelong memory for index rebuild.")

    def _update_vector_index_batch(self, force_update: bool = False):
        """ Processes pending adds/removals and updates the FAISS index in batches (if vector search enabled). """
        if not self.vector_search_enabled or not self.vector_index or self.vector_dimension is None: return
        pending_changes_count = len(self._pending_vector_adds) + len(self._pending_vector_removals)
        time_since_last_update = time.monotonic() - self._last_vector_batch_update_time

        should_update = force_update or \
                        (pending_changes_count >= VECTOR_INDEX_BATCH_UPDATE_SIZE) or \
                        (pending_changes_count > 0 and time_since_last_update >= VECTOR_INDEX_BATCH_UPDATE_INTERVAL_SEC)

        if not should_update: return

        if pending_changes_count == 0:
             self._last_vector_batch_update_time = time.monotonic(); return

        logger.debug(f"Updating vector index batch: Adds={len(self._pending_vector_adds)}, Rems={len(self._pending_vector_removals)}")
        try:
            # 1. Process Removals
            ids_to_remove_list: List[int] = []
            keys_actually_removed = set()
            if self._pending_vector_removals:
                faiss_ids_to_remove_set = set()
                keys_to_remove_now = self._pending_vector_removals.copy()
                for f_id, key in list(self.vector_id_map.items()):
                    if key in keys_to_remove_now:
                        faiss_ids_to_remove_set.add(f_id)
                        del self.vector_id_map[f_id]
                        keys_actually_removed.add(key)

                if faiss_ids_to_remove_set:
                    ids_to_remove_list = list(faiss_ids_to_remove_set)
                    remove_ids_np = np.array(ids_to_remove_list, dtype='int64')
                    remove_count = self.vector_index.remove_ids(remove_ids_np) # Returns num removed
                    if remove_count != len(ids_to_remove_list):
                        logger.warning(f"Vector index removal count mismatch. Expected {len(ids_to_remove_list)}, removed {remove_count}. Rebuilding index.")
                        self._rebuild_vector_index() # Rebuild on inconsistency
                        return # Exit after rebuild starts
                    self._vector_index_dirty_counter += remove_count
                    logger.debug(f"Batch removed {remove_count} vector entries for {len(keys_actually_removed)} keys.")
                self._pending_vector_removals.difference_update(keys_actually_removed) # Remove processed keys

            # 2. Process Additions/Updates
            embeddings_to_add: List[np.ndarray] = []
            faiss_ids_to_add: List[int] = []
            keys_added = set()
            if self._pending_vector_adds:
                 # Add keys that were not just removed in this same batch update cycle
                 keys_to_add_now = list(self._pending_vector_adds.keys() - keys_actually_removed)
                 next_faiss_id = (max(self.vector_id_map.keys()) + 1) if self.vector_id_map else 0
                 for key in keys_to_add_now:
                      embedding_list = self._pending_vector_adds.pop(key, None) # Use pop with default None
                      if embedding_list is None: continue # Skip if somehow removed concurrently

                      embedding_np = np.array(embedding_list, dtype='float32')
                      if embedding_np.shape[0] == self.vector_dimension:
                          embeddings_to_add.append(embedding_np)
                          # Ensure the next ID is unique
                          while next_faiss_id in self.vector_id_map: next_faiss_id += 1
                          faiss_ids_to_add.append(next_faiss_id)
                          self.vector_id_map[next_faiss_id] = key
                          keys_added.add(key)
                          next_faiss_id += 1
                      else: logger.warning(f"Skipping add for key '{key}': Dimension mismatch ({embedding_np.shape[0]} vs {self.vector_dimension}).")

                 if embeddings_to_add:
                      embeddings_np = np.vstack(embeddings_to_add).astype('float32')
                      faiss_ids_np = np.array(faiss_ids_to_add).astype('int64')
                      self.vector_index.add_with_ids(embeddings_np, faiss_ids_np)
                      self._vector_index_dirty_counter += len(embeddings_to_add)
                      logger.debug(f"Batch added/updated {len(embeddings_to_add)} vectors for {len(keys_added)} keys.")

            self._last_vector_batch_update_time = time.monotonic()
            self._save_vector_index_if_needed()

        except Exception as e:
            logger.error(f"Error during vector index batch update: {e}", exc_info=True)
            # Consider disabling vector search temporarily or triggering a rebuild on error
            # self.vector_search_enabled = False


    def _save_vector_index_if_needed(self):
        """ Saves the FAISS index and map if the dirty counter reaches the threshold (if vector search enabled). """
        if self.vector_search_enabled and self._vector_auto_save_interval is not None and self._vector_index_dirty_counter >= self._vector_auto_save_interval:
            self._save_vector_index()

    def _save_vector_index(self):
        """ Saves the current FAISS index and ID map to disk (if vector search enabled). Flushes pending batch updates first. """
        if not self.vector_search_enabled or not self.vector_index or not self.vector_index_path: return

        self._update_vector_index_batch(force_update=True) # Ensure pending changes are written
        map_path = self.vector_index_path + ".map.pkl"
        # Check again after batch update, as it might disable vector search on error
        if not self.vector_search_enabled or not self.vector_index or not self.vector_index_path:
             logger.warning("Skipping vector index save as it was disabled during batch update.")
             return

        logger.info(f"Saving FAISS index ({self.vector_index.ntotal} entries) & map ({len(self.vector_id_map)})...")
        try:
            save_dir = os.path.dirname(self.vector_index_path); save_dir and os.makedirs(save_dir, exist_ok=True)
            # Ensure index is not empty before saving (can happen after errors/removals)
            if self.vector_index.ntotal > 0:
                 faiss.write_index(self.vector_index, self.vector_index_path);
                 with open(map_path, 'wb') as f: pickle.dump(self.vector_id_map, f)
                 self._vector_index_dirty_counter = 0
                 logger.debug(f"Saved FAISS index to {self.vector_index_path} and map to {map_path}")
            else:
                 logger.warning("Skipping save of empty FAISS index.")
                 # Optionally remove old files if index becomes empty
                 if os.path.exists(self.vector_index_path): os.remove(self.vector_index_path)
                 if os.path.exists(map_path): os.remove(map_path)
                 self._vector_index_dirty_counter = 0 # Reset counter even if empty
        except Exception as e: logger.error(f"Error saving FAISS index/map: {e}", exc_info=True)

    def _get_text_for_embedding(self, entry_data: Any) -> str:
         """ Extracts relevant text from memory entry data for generating embeddings. """
         text_parts = []; max_len = 512
         if isinstance(entry_data, dict):
             # Prioritize keys likely containing meaningful text
             prioritized_keys = [
                 'reasoning', 'summary', 'message', 'query', 'goal', 'command', 'error', 'decision', 'evaluation',
                 'description', 'details', 'analysis_result', 'suggested_response', 'trigger_pattern' # Added learning keys
                 ]
             for key in prioritized_keys:
                 value = entry_data.get(key);
                 if value:
                     try:
                         # Avoid embedding excessively long strings directly
                         value_str = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
                         if len(value_str) > max_len * 0.8: # Truncate long individual values
                              value_str = value_str[:int(max_len*0.8)] + "..."
                         text_parts.append(f"{key.replace('_',' ').title()}: {value_str}");
                     except Exception: pass # Ignore errors stringifying individual values
             # Fallback for simple dicts if no prioritized keys found
             if not text_parts:
                  try:
                      # Exclude keys less likely to be useful for semantic similarity
                      exclude_keys = {'raw_response', 'traceback', 'embedding', 'config_used', 'final_filesystem',
                                      'final_signals', 'action_params', 'params_hash', 'timestamp', 'exit_code',
                                      'creation_timestamp', 'last_triggered_timestamp', 'trigger_count', 'rule_id'}
                      simple_data = {k: v for k,v in entry_data.items() if k not in exclude_keys and isinstance(v, (str, int, float, bool))}
                      if simple_data: text_parts.append(json.dumps(simple_data, default=str))
                  except Exception: pass
         elif isinstance(entry_data, str): text_parts.append(entry_data)
         # Corrected indentation for try...except block
         elif isinstance(entry_data, list):
             try:
                 list_str = ", ".join(map(str, entry_data))
                 text_parts.append(f"Sequence: {list_str}")
             except Exception:
                 pass # Ignore errors converting list elements
         full_text = ". ".join(filter(None, text_parts)); return full_text[:max_len].strip() # Ensure final truncation

    def _get_or_create_embedding(self, lifelong_entry: MemoryEntry) -> Optional[List[float]]:
         """ Gets embedding list from entry if exists, otherwise creates, stores, and returns it (if vector search enabled). """
         if not self.vector_search_enabled or not self.vector_embedding_model or self.vector_dimension is None:
             return None
         # Ensure we don't try to embed internal state entries
         entry_key = lifelong_entry.get('key')
         if entry_key in [self._learning_params_key, self._behavioral_rules_key]: return None

         existing_embedding_list = lifelong_entry.get('embedding')
         # Basic validation of existing embedding
         if isinstance(existing_embedding_list, list) and len(existing_embedding_list) == self.vector_dimension:
             # Optional: Check if data has changed significantly since embedding was created? (More complex)
             return existing_embedding_list

         try:
             text_to_embed = self._get_text_for_embedding(lifelong_entry.get('data'));
             if not text_to_embed: logger.debug(f"No text found for embedding for key {lifelong_entry.get('key', '?')}"); return None

             # Use torch.no_grad() context manager if torch is available
             with (torch.no_grad() if torch else contextlib.nullcontext()): # type: ignore
                 embedding_np = self.vector_embedding_model.encode([text_to_embed], convert_to_numpy=True)[0]

             if not isinstance(embedding_np, np.ndarray): logger.error(f"Embedding model did not return numpy array for key {lifelong_entry.get('key')}. Type: {type(embedding_np)}"); return None
             embedding_np = embedding_np.astype(np.float32) # Ensure correct dtype
             if embedding_np.shape[0] != self.vector_dimension: logger.error(f"Embedding dim mismatch ({embedding_np.shape[0]} vs {self.vector_dimension}) for key {lifelong_entry.get('key')}"); return None

             embedding_list = embedding_np.tolist()
             lifelong_entry['embedding'] = embedding_list # MUTATE entry dict to store the new embedding
             logger.debug(f"Created and stored embedding for key {lifelong_entry.get('key')}")
             return embedding_list
         except Exception as e: logger.warning(f"Failed to create embedding for key {lifelong_entry.get('key', '?')}: {e}", exc_info=False); return None # Reduce log noise


    # --- Episodic Memory Methods ---
    def add_episodic_memory(self, data: Dict[str, Any], tags: Optional[List[str]] = None) -> Optional[str]:
        if data is None: logger.warning("Attempted to add None data to episodic memory."); return None
        entry_id = f"ep_{uuid.uuid4().hex[:8]}"; entry: MemoryEntry = {'id': entry_id, 'timestamp': time.time(), 'data': copy.deepcopy(data), 'tags': tags or []}
        try: self._episodic_memory.append(entry); return entry_id
        except Exception as e: logger.error(f"Failed to add entry to episodic memory: {e}", exc_info=True); return None
    def get_latest_episodic(self, count: int = 10) -> List[MemoryEntry]: return list(self._episodic_memory)[-count:]
    def find_episodic_by_criteria(self, filter_function: FilterFunc, limit: Optional[int] = None, newest_first: bool = False) -> List[MemoryEntry]:
        results: List[MemoryEntry] = []; memory_list = list(self._episodic_memory); iterator = reversed(memory_list) if newest_first else memory_list
        for entry in iterator:
             try:
                 if filter_function(entry): results.append(entry)
                 if limit is not None and len(results) >= limit: break
             except Exception as e: logger.warning(f"Error applying filter function to episodic entry {entry.get('id','?')}: {e}")
        return results
    def get_episodic_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        for entry in self._episodic_memory:
            if entry.get('id') == entry_id: return entry
        return None

    # --- Lifelong Memory Methods ---
    def add_lifelong_memory(self, key: str, data: Any, tags: Optional[List[str]] = None) -> Optional[str]:
        """ Adds/updates lifelong memory. Queues entry for vector index batch update if enabled and not internal state. """
        if not isinstance(key, str) or not key.strip(): logger.error(f"Invalid lifelong memory key: '{key}'."); return None
        if data is None: logger.warning(f"Attempted to add None data to lifelong memory for key '{key}'. Skipping."); return None

        entry_id: Optional[str] = None; removed_key: Optional[str] = None; timestamp = time.time()
        entry: MemoryEntry = {'timestamp': timestamp, 'key': key, 'data': copy.deepcopy(data), 'tags': tags or []}
        # Explicitly set embedding to None initially, will be populated later if needed
        entry['embedding'] = None

        try:
            is_update = key in self._lifelong_memory
            if not is_update and len(self._lifelong_keys_by_age) >= self.max_lifelong_size:
                # Evict oldest entry if full
                oldest_key = self._lifelong_keys_by_age.popleft()
                if oldest_key in self._lifelong_memory:
                    removed_key = oldest_key
                    del self._lifelong_memory[oldest_key]
                    logger.info(f"Lifelong memory full. Removed oldest entry: '{oldest_key}'")
                    # Queue removal from vector index if enabled and it wasn't internal state
                    if removed_key and self.vector_search_enabled and removed_key not in [self._learning_params_key, self._behavioral_rules_key]:
                        self._pending_vector_removals.add(removed_key)
                        self._pending_vector_adds.pop(removed_key, None) # Remove if pending add

            if is_update:
                existing_entry = self._lifelong_memory[key]
                entry['id'] = existing_entry.get('id'); # Preserve existing ID
                entry['embedding'] = existing_entry.get('embedding') # Preserve existing embedding initially
                existing_entry.update(entry); # Update the entry in place
                entry_id = existing_entry['id'];
                logger.debug(f"Updated lifelong memory key '{key}'.")
                # Move key to the end of the age deque
                self._lifelong_keys_by_age = deque((k for k in self._lifelong_keys_by_age if k != key), maxlen=self.max_lifelong_size)
                self._lifelong_keys_by_age.append(key)
            else:
                entry['id'] = f"ll_{uuid.uuid4().hex[:8]}"; # Generate new ID
                self._lifelong_memory[key] = entry;
                self._lifelong_keys_by_age.append(key)
                entry_id = entry['id'];
                logger.debug(f"Added new lifelong memory key '{key}'.")

            # Handle Vector Index Update (Only if enabled and not internal state)
            if self.vector_search_enabled and key not in [self._learning_params_key, self._behavioral_rules_key]:
                # Re-create embedding if it's an update or first time add
                embedding_list = self._get_or_create_embedding(self._lifelong_memory[key])
                if embedding_list is not None:
                     self._pending_vector_adds[key] = embedding_list
                     self._pending_vector_removals.discard(key) # Ensure it's not marked for removal
                else:
                    # If embedding failed, ensure it's removed if it existed before
                    self._pending_vector_removals.add(key)
                    self._pending_vector_adds.pop(key, None)
                self._update_vector_index_batch() # Trigger batch update check

            return entry_id
        except Exception as e: logger.error(f"Failed add/update lifelong memory for key '{key}': {e}", exc_info=True); return None

    def get_lifelong_memory(self, key: str) -> Optional[MemoryEntry]: return self._lifelong_memory.get(key)
    def get_all_lifelong_keys(self) -> List[str]: return list(self._lifelong_memory.keys())
    def get_all_lifelong_entries(self, include_internal: bool = False) -> List[MemoryEntry]:
        """ Retrieves all lifelong entries, ordered by age (newest first). Option to include internal state entries. """
        entries = []
        for key in reversed(self._lifelong_keys_by_age):
             if key in self._lifelong_memory:
                  if include_internal or key not in [self._learning_params_key, self._behavioral_rules_key]:
                       entries.append(self._lifelong_memory[key])
        return entries


    def remove_lifelong_memory(self, key: str) -> bool:
        """ Removes entry from lifelong memory and queues removal from vector index if enabled. """
        if key in self._lifelong_memory:
            # Prevent removal of internal state keys via this method
            if key in [self._learning_params_key, self._behavioral_rules_key]:
                logger.warning(f"Attempted to remove internal state key '{key}' via remove_lifelong_memory. Use specific methods if needed.")
                return False
            try:
                del self._lifelong_memory[key]
                self._lifelong_keys_by_age = deque((k for k in self._lifelong_keys_by_age if k != key), maxlen=self.max_lifelong_size)
                # Queue vector removal if enabled
                if self.vector_search_enabled:
                    self._pending_vector_removals.add(key)
                    self._pending_vector_adds.pop(key, None) # Remove from pending adds if present
                    self._update_vector_index_batch() # Trigger batch update check
                logger.info(f"Removed lifelong memory entry: '{key}' (queued for vector removal if enabled)")
                return True
            except Exception as e: logger.error(f"Error removing lifelong memory key '{key}': {e}", exc_info=True); return False
        return False

    def find_lifelong_by_criteria(self, filter_function: FilterFunc, limit: Optional[int] = None, newest_first: bool = False) -> List[MemoryEntry]:
        results: List[MemoryEntry] = []; keys_to_check = list(self._lifelong_keys_by_age); iterator = reversed(keys_to_check) if newest_first else keys_to_check
        for key in iterator:
            # Skip internal state keys during general searches
            if key in [self._learning_params_key, self._behavioral_rules_key]: continue
            if key in self._lifelong_memory:
                entry = self._lifelong_memory[key]
                try:
                    if filter_function(entry): results.append(entry)
                    if limit is not None and len(results) >= limit: break
                except Exception as e: logger.warning(f"Error applying filter function to lifelong entry {key}: {e}")
        return results

    # --- REMOVED Agent Pool State Methods ---

    # --- Seed Learning Mechanism Persistence Methods ---
    def load_learning_state(self):
        """ Loads learning parameters and rules from lifelong memory during startup. Called internally by __init__ after load_memory. """
        logger.debug("Attempting to load learning parameters and rules from memory...")
        # Load Parameters
        params_entry = self.get_lifelong_memory(self._learning_params_key)
        if params_entry and isinstance(params_entry.get('data'), dict):
            loaded_params = params_entry['data']
            # Merge loaded params with defaults from config, respecting structure and bounds
            merged_params = copy.deepcopy(SEED_LEARNING_PARAMETERS)
            for category, default_category_config in SEED_LEARNING_PARAMETERS.items():
                # Check if category exists in loaded data
                if category in loaded_params and isinstance(loaded_params[category], dict):
                    loaded_category_data = loaded_params[category]
                    # Decide structure based on category name (as defined in config)
                    if category == "evaluation_weights": # Nested structure
                        for name, default_config in default_category_config.items():
                            if name in loaded_category_data and isinstance(loaded_category_data[name], dict):
                                if 'value' in loaded_category_data[name]:
                                    loaded_value = loaded_category_data[name]['value']
                                    # Validate and merge
                                    if isinstance(loaded_value, type(default_config['value'])):
                                        if 'min' in default_config and loaded_value < default_config['min']: loaded_value = default_config['min']
                                        if 'max' in default_config and loaded_value > default_config['max']: loaded_value = default_config['max']
                                        merged_params[category][name]['value'] = loaded_value
                                    else: logger.warning(f"Type mismatch loading param '{category}.{name}'. Using default.")
                    else: # Direct structure (e.g., rule_application_mode, llm_query_temperature)
                        if 'value' in loaded_category_data:
                            loaded_value = loaded_category_data['value']
                             # Validate and merge (using the top-level category config from defaults)
                            if isinstance(loaded_value, type(default_category_config['value'])):
                                if 'min' in default_category_config and loaded_value < default_category_config['min']: loaded_value = default_category_config['min']
                                if 'max' in default_category_config and loaded_value > default_category_config['max']: loaded_value = default_category_config['max']
                                if 'options' in default_category_config and loaded_value not in default_category_config['options']: loaded_value = default_category_config['value']
                                merged_params[category]['value'] = loaded_value
                            else: logger.warning(f"Type mismatch loading param '{category}'. Using default.")
            self._learning_parameters = merged_params
            logger.info(f"Loaded and merged learning parameters state (Using Config Structure, Loaded Timestamp: {params_entry.get('timestamp')}).")
        else:
            logger.info("No existing learning parameters found in memory, using defaults from config. Saving defaults.")
            # self._learning_parameters is already initialized with defaults
            self.save_learning_parameters() # Save initial defaults

        # Load Rules
        rules_entry = self.get_lifelong_memory(self._behavioral_rules_key)
        if rules_entry and isinstance(rules_entry.get('data'), dict):
            # Basic validation of loaded rules structure
            validated_rules = {}
            for rule_id, rule_data in rules_entry['data'].items():
                if isinstance(rule_data, dict) and \
                   isinstance(rule_data.get('trigger_pattern'), dict) and \
                   isinstance(rule_data.get('suggested_response'), str) and \
                   isinstance(rule_data.get('rule_id'), str) and rule_data['rule_id'] == rule_id:
                    validated_rules[rule_id] = rule_data
                else:
                    logger.warning(f"Skipping load of invalid behavioral rule data for id '{rule_id}'.")
            self._behavioral_rules = validated_rules
            logger.info(f"Loaded {len(self._behavioral_rules)} behavioral rules (Timestamp: {rules_entry.get('timestamp')}).")
        else:
             logger.info("No existing behavioral rules found in memory.")
             self._behavioral_rules = {} # Ensure it's initialized

    def save_learning_parameters(self):
        """ Saves the current learning parameters to lifelong memory. """
        logger.debug(f"Saving learning parameters to memory key '{self._learning_params_key}'")
        # Use add_lifelong_memory which handles updates and age tracking
        self.add_lifelong_memory(self._learning_params_key, copy.deepcopy(self._learning_parameters), tags=['Seed', 'Learning', 'Parameter', 'Config', 'InternalState'])

    def get_learning_parameter(self, name: str) -> Optional[Any]:
        """ Gets the current value of a specific learning parameter (e.g., 'evaluation_weights.goal_prog' or 'llm_query_temperature.value'). """
        parts = name.split('.')
        if not parts or not parts[0]: # Handle empty string query for all params
             return copy.deepcopy(self._learning_parameters) # Return all if name is empty

        category = parts[0]
        if category not in self._learning_parameters:
            logger.warning(f"Learning parameter category '{category}' not found.")
            return None

        current_level_state = self._learning_parameters[category]

        if len(parts) == 1: # Requesting whole category dict
            return copy.deepcopy(current_level_state)

        if category == "evaluation_weights": # Nested structure
            if len(parts) == 2: # Requesting specific weight config dict (e.g., evaluation_weights.execution)
                param_name = parts[1]
                if param_name in current_level_state and isinstance(current_level_state[param_name], dict):
                     return copy.deepcopy(current_level_state[param_name])
                else: logger.warning(f"Learning parameter '{name}' not found."); return None
            elif len(parts) == 3 and parts[2] == 'value': # Requesting specific weight value (e.g., evaluation_weights.execution.value)
                param_name = parts[1]
                if param_name in current_level_state and isinstance(current_level_state[param_name], dict) and 'value' in current_level_state[param_name]:
                     return current_level_state[param_name]['value']
                else: logger.warning(f"Learning parameter value for '{name}' not found."); return None
            else: logger.warning(f"Invalid query format for nested parameter '{name}'."); return None
        else: # Direct structure (e.g., llm_query_temperature)
            if len(parts) == 2: # Requesting specific key (e.g., llm_query_temperature.value or llm_query_temperature.min)
                 key_name = parts[1]
                 if key_name in current_level_state:
                      return current_level_state[key_name]
                 else: logger.warning(f"Key '{key_name}' not found in learning parameter category '{category}'."); return None
            else: logger.warning(f"Invalid query format for direct parameter '{name}'."); return None

    def update_learning_parameter(self, name: str, value: Any) -> bool:
        """ Updates a learning parameter, respecting bounds/types defined in config. Saves automatically. """
        parts = name.split('.')
        if len(parts) < 2: logger.error(f"Invalid learning parameter name '{name}'. Must include category and key (e.g., 'llm_query_temperature.value')."); return False

        category = parts[0]
        if category not in SEED_LEARNING_PARAMETERS or category not in self._learning_parameters:
            logger.error(f"Invalid learning parameter category '{category}'."); return False

        # Get config for validation and state for update
        default_category_config = SEED_LEARNING_PARAMETERS[category]
        current_category_state = self._learning_parameters[category]

        try:
            param_config = None
            param_state = None
            expected_type = None
            value_key_path = None # e.g., ['value'] or ['sub_param', 'value']

            if category == "evaluation_weights": # Nested structure
                if len(parts) != 3 or parts[2] != 'value': logger.error(f"Invalid parameter name '{name}'. Use format 'evaluation_weights.param_name.value'."); return False
                param_name = parts[1]
                if param_name not in default_category_config or param_name not in current_category_state: logger.error(f"Invalid parameter sub-key '{param_name}' for category '{category}'."); return False

                param_config = default_category_config[param_name]
                param_state = current_category_state[param_name] # This dict holds 'value', 'min' etc.
                expected_type = type(param_config.get('value'))
                value_key_path = ['value'] # Update 'value' within this sub-dict

            else: # Direct structure
                if len(parts) != 2 or parts[1] != 'value': logger.error(f"Invalid parameter name '{name}'. Use format 'category_name.value'."); return False
                param_config = default_category_config # Top-level config holds min/max/options
                param_state = current_category_state # Top-level state holds 'value'
                expected_type = type(param_config.get('value'))
                value_key_path = ['value'] # Update 'value' at this level

            # --- Perform Validation and Update ---
            if not isinstance(value, expected_type):
                 logger.error(f"Type mismatch for param '{name}'. Expected {expected_type}, got {type(value)}."); return False
            # Bounds check (using param_config which points to the dict containing min/max)
            if 'min' in param_config and value < param_config['min']:
                 logger.warning(f"Clamping value for '{name}' to min bound {param_config['min']}. Requested: {value}"); value = param_config['min']
            if 'max' in param_config and value > param_config['max']:
                 logger.warning(f"Clamping value for '{name}' to max bound {param_config['max']}. Requested: {value}"); value = param_config['max']
            # Options check (using param_config)
            if 'options' in param_config and value not in param_config['options']:
                 logger.error(f"Invalid option for param '{name}'. Got '{value}', allowed: {param_config['options']}."); return False

            # Check if value actually changed before updating and saving
            # Access the 'value' correctly using param_state and value_key_path
            current_value = param_state[value_key_path[0]] # Assumes depth 1 for value path for now

            if current_value != value:
                param_state[value_key_path[0]] = value # Update the 'value' field in the state dict
                logger.info(f"Updated learning parameter '{name}' to {value}.")
                self.save_learning_parameters() # Persist change
            else:
                logger.debug(f"Learning parameter '{name}' already set to {value}. No update needed.")
            return True

        except Exception as e:
            logger.error(f"Error updating learning parameter '{name}': {e}", exc_info=True); return False

    def save_behavioral_rules(self):
        """ Saves the current set of behavioral rules to lifelong memory. """
        logger.debug(f"Saving {len(self._behavioral_rules)} behavioral rules to memory key '{self._behavioral_rules_key}'")
        self.add_lifelong_memory(self._behavioral_rules_key, copy.deepcopy(self._behavioral_rules), tags=['Seed', 'Learning', 'RuleInduction', 'Config', 'InternalState'])

    def add_behavioral_rule(self, rule_data: Dict) -> Optional[str]:
        """ Adds or updates a behavioral rule. Returns the rule ID. """
        # Validate input structure
        if not isinstance(rule_data.get('trigger_pattern'), dict) or not isinstance(rule_data.get('suggested_response'), str):
             logger.error("Invalid rule_data format for adding behavioral rule. Requires 'trigger_pattern' (dict) and 'suggested_response' (str).")
             return None

        # Determine rule ID: use provided or generate new
        rule_id = rule_data.get('rule_id')
        is_update = False
        if rule_id:
            if not isinstance(rule_id, str) or not rule_id.strip():
                 logger.error(f"Invalid rule_id provided: '{rule_id}'. Generating new ID.")
                 rule_id = f"rule_{uuid.uuid4().hex[:8]}"
            elif rule_id in self._behavioral_rules:
                 is_update = True
                 logger.info(f"Updating existing behavioral rule '{rule_id}'.")
            else:
                 logger.info(f"Adding new behavioral rule with specified ID '{rule_id}'.")
        else:
            rule_id = f"rule_{uuid.uuid4().hex[:8]}"
            logger.info(f"Adding new behavioral rule with generated ID '{rule_id}'.")

        # Construct the rule entry
        new_rule: BehavioralRule = {
            'rule_id': rule_id,
            'trigger_pattern': copy.deepcopy(rule_data['trigger_pattern']),
            'suggested_response': rule_data['suggested_response'],
            'creation_timestamp': self._behavioral_rules[rule_id].get('creation_timestamp', time.time()) if is_update else time.time(),
            'last_updated_timestamp': time.time(),
            'last_triggered_timestamp': self._behavioral_rules[rule_id].get('last_triggered_timestamp') if is_update else None,
            'trigger_count': self._behavioral_rules[rule_id].get('trigger_count', 0) if is_update else 0,
            'llm_reasoning': rule_data.get('reasoning', '') # Store LLM reasoning if provided
        }

        self._behavioral_rules[rule_id] = new_rule
        self.save_behavioral_rules() # Persist the change
        return rule_id

    def remove_behavioral_rule(self, rule_id: str) -> bool:
         if rule_id in self._behavioral_rules:
              del self._behavioral_rules[rule_id]
              logger.info(f"Removed behavioral rule '{rule_id}'.")
              self.save_behavioral_rules()
              return True
         logger.warning(f"Attempted to remove non-existent rule '{rule_id}'.")
         return False

    def get_behavioral_rules(self) -> Dict[str, BehavioralRule]:
         """ Returns a copy of the current behavioral rules. """
         return copy.deepcopy(self._behavioral_rules)

    def update_rule_trigger_stats(self, rule_id: str):
         """ Updates the stats when a rule is triggered and saves. """
         if rule_id in self._behavioral_rules:
              self._behavioral_rules[rule_id]['last_triggered_timestamp'] = time.time()
              self._behavioral_rules[rule_id]['trigger_count'] = self._behavioral_rules[rule_id].get('trigger_count', 0) + 1
              logger.debug(f"Rule '{rule_id}' triggered (Count: {self._behavioral_rules[rule_id]['trigger_count']}).")
              # Save rules after stats update
              self.save_behavioral_rules()
         else:
              logger.warning(f"Attempted to update trigger stats for non-existent rule '{rule_id}'.")

    # --- General & Persistence Methods ---
    def clear_all_memory(self):
        """ Clears all episodic/lifelong memories, learning state, and vector index. """
        logger.warning("Clearing ALL memories (episodic, lifelong, learning state, vector index)...")
        self._episodic_memory.clear(); self._lifelong_memory.clear(); self._lifelong_keys_by_age.clear()
        # Reset learning state to defaults
        self._learning_parameters = copy.deepcopy(SEED_LEARNING_PARAMETERS)
        self._behavioral_rules = {}
        logger.info("Learning parameters reset to defaults. Behavioral rules cleared.")
        # Clear internal state from memory dict before saving/rebuilding index
        self._lifelong_memory.pop(self._learning_params_key, None)
        self._lifelong_memory.pop(self._behavioral_rules_key, None)

        if self.vector_search_enabled:
            logger.info("Resetting vector index..."); self._pending_vector_adds.clear(); self._pending_vector_removals.clear()
            if self._initialize_vector_index(force_create=True):
                 # Rebuild index from remaining (non-internal) entries if any, then save
                 self._rebuild_vector_index() # This calls _save_vector_index internally
            else: logger.error("Failed to reset vector index during clear_all_memory.")
        logger.info("All memories cleared.")
        # Save the cleared state (including default learning params)
        self.save_memory()


    def log(self, event_type: str, data: Any, tags: Optional[List[str]] = None) -> Optional[str]:
        """ Logs event, storing lifelong/episodic based on config. """
        tags = tags or [];
        # Check against configured lifelong event types OR tags
        is_lifelong = event_type in MEMORY_LIFELONG_EVENT_TYPES or any(tag in MEMORY_LIFELONG_TAGS for tag in tags)

        if is_lifelong:
            # Sanitize event type to be used as part of the key
            safe_event_type = re.sub(r'[^a-zA-Z0-9_.-]', '_', event_type)
            # Create a unique key including a timestamp and UUID part
            key = f"{safe_event_type}_{int(time.time())}_{uuid.uuid4().hex[:4]}"
            return self.add_lifelong_memory(key, data, tags=tags)
        else:
            # Standardize episodic entry format
            episodic_data = {"event_type": event_type, "details": data}
            return self.add_episodic_memory(episodic_data, tags=tags)


    def retrieve_context(self, query: Optional[str] = None, limit: int = 7) -> Dict[str, Any]:
         """ Retrieves diverse context potentially relevant for Seed core decision making. """
         if self.vector_search_enabled: self._update_vector_index_batch(force_update=True)

         logger.debug(f"Retrieving Seed context (limit={limit}, query='{query}')...")
         context: Dict[str, Any] = {'error': None};
         # Adjust limits based on overall limit
         episodic_limit = max(1, limit // 3);
         lifelong_limit_each = max(1, math.ceil(limit / 4.0)); # Distribute limit among different types
         search_limit = limit # Use full limit for direct searches

         try:
             # Recent Episodic Events (Less emphasis compared to ModularAGI)
             context['recent_episodic'] = self.get_latest_episodic(episodic_limit)

             # Key Lifelong Events (Using find_lifelong_by_criteria which excludes internal state)
             context['recent_goals'] = self.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("seed_goal_set"), limit=lifelong_limit_each, newest_first=True)
             context['recent_seed_decisions'] = self.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Decision"), limit=lifelong_limit_each, newest_first=True)
             context['recent_seed_evaluations'] = self.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Evaluation"), limit=lifelong_limit_each, newest_first=True)
             context['recent_core_mod_actions'] = self.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Action_MODIFY_CORE_CODE") or e.get('key','').startswith("SEED_Action_TEST_CORE") or e.get('key','').startswith("SEED_Action_VERIFY_CORE"), limit=lifelong_limit_each, newest_first=True)
             context['recent_learning_actions'] = self.find_lifelong_by_criteria(lambda e: e.get('key','').startswith("SEED_Action_UPDATE_LEARNING") or e.get('key','').startswith("SEED_Action_INDUCE_BEHAVIORAL"), limit=lifelong_limit_each, newest_first=True)
             context['recent_errors'] = self.find_lifelong_by_criteria(lambda e: 'Error' in e.get('tags',[]) or 'Critical' in e.get('tags',[]), limit=lifelong_limit_each, newest_first=True)

             # Search Results
             context['text_search_results'] = []
             context['vector_search_results'] = []
             if query:
                 context['text_search_results'] = self.search_memory_text(query, search_type='both', limit=search_limit)
                 if self.vector_search_enabled:
                     context['vector_search_results'] = self.search_memory_vector(query, k=search_limit)

             # Add current learning state to context
             context['current_learning_parameters'] = self.get_learning_parameter('') # Get all params
             context['current_behavioral_rules'] = self.get_behavioral_rules()

         except Exception as e: logger.error(f"Error retrieving Seed context: {e}", exc_info=True); context["error"] = f"Context retrieval failed: {e}"
         return context

    def search_memory_vector(self, query: str, k: int = 10) -> List[MemoryEntry]:
        """ Performs vector similarity search on lifelong memory (if enabled). """
        if not self.vector_search_enabled or not self.vector_embedding_model or not self.vector_index:
            logger.debug("Vector search called but components unavailable/disabled."); return []
        if not query or not isinstance(query, str): logger.warning("Vector search requires a valid query string."); return []
        if k <= 0: return []

        self._update_vector_index_batch(force_update=True) # Ensure index is up-to-date
        logger.debug(f"Performing vector search (k={k}): '{query[:100]}...'")
        try:
            # Use torch.no_grad() context manager if torch is available
            with (torch.no_grad() if torch else contextlib.nullcontext()): # type: ignore
                 q_embed_np = self.vector_embedding_model.encode([query], convert_to_numpy=True).astype('float32')

            # Check index size before searching
            if self.vector_index.ntotal == 0: logger.debug("Vector search skipped: Index is empty."); return []

            distances, faiss_ids = self.vector_index.search(q_embed_np, k); results: List[MemoryEntry] = []
            if faiss_ids.size > 0:
                for i, f_id_np in enumerate(faiss_ids[0]):
                    f_id = int(f_id_np);
                    if f_id == -1: continue # FAISS uses -1 for invalid/missing IDs
                    ll_key = self.vector_id_map.get(f_id)
                    if ll_key:
                        entry = self.get_lifelong_memory(ll_key)
                        if entry:
                             ecopy = copy.deepcopy(entry); ecopy['vector_similarity_score'] = float(distances[0][i]); results.append(ecopy)
                        else: logger.warning(f"Vector search inconsistency: FAISS ID {f_id} mapped to key '{ll_key}', but entry not found.")
                    else: logger.warning(f"Vector search inconsistency: Found FAISS ID {f_id} which is not present in the ID map.")
            logger.debug(f"Vector search found {len(results)} results.")
            return results
        except Exception as e: logger.error(f"Vector search error: {e}", exc_info=True); return []

    def search_memory_text(self, query: str, search_type: str = 'both', limit: int = 20) -> List[MemoryEntry]:
        """ Performs basic text search (substring matching) across memory entries, excluding internal state. """
        if not query or limit <= 0: return []
        logger.debug(f"Basic text search (limit={limit}, type='{search_type}'): '{query[:100]}...'")
        lower_query = query.lower(); matched: List[MemoryEntry] = []

        # Search Episodic Memory
        if search_type in ['episodic', 'both']:
             # Iterate newest first for text search relevance
             for entry in reversed(list(self._episodic_memory)):
                  if len(matched) >= limit: break
                  try:
                      try: entry_str = json.dumps(entry, default=str).lower()
                      except Exception: entry_str = str(entry).lower() # Fallback string conversion
                  except Exception as str_err: logger.warning(f"Error converting episodic entry {entry.get('id','?')} to string for search: {str_err}"); entry_str = ""
                  if lower_query in entry_str: matched.append(entry)

        # Search Lifelong Memory (excluding internal state)
        if len(matched) < limit and search_type in ['lifelong', 'both']:
             keys_to_check = list(self._lifelong_keys_by_age)
             # Iterate newest first
             for key in reversed(keys_to_check):
                 if len(matched) >= limit: break
                 # Skip internal state keys
                 if key in [self._learning_params_key, self._behavioral_rules_key]: continue
                 if key in self._lifelong_memory:
                     entry = self._lifelong_memory[key]
                     try:
                         try: entry_str = json.dumps(entry, default=str).lower()
                         except Exception: entry_str = str(entry).lower() # Fallback string conversion
                     except Exception as str_err: logger.warning(f"Error converting lifelong entry {key} to string for search: {str_err}"); entry_str = ""
                     if lower_query in entry_str: matched.append(entry)

        return matched

    def save_memory(self):
        """ Saves memory state (episodic, lifelong, LEARNING STATE) AND forces vector index save. """
        # --- Trigger saves for learning state before main save ---
        self.save_learning_parameters()
        self.save_behavioral_rules()
        # --- End Trigger ---

        if not self.save_file_path: logger.warning("Memory save_file_path not set. Skipping memory save."); return
        logger.info(f"Saving memory state to {self.save_file_path}...")
        try:
            # Ensure save directory exists
            save_dir_path = pathlib.Path(self.save_file_path).parent
            save_dir_path.mkdir(parents=True, exist_ok=True)

            # Prepare data to save
            memory_data = {
                'episodic': list(self._episodic_memory),
                'lifelong': self._lifelong_memory,
                'lifelong_keys': list(self._lifelong_keys_by_age),
            }
            # Pickle the data
            with open(self.save_file_path, 'wb') as f: pickle.dump(memory_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved memory state ({len(memory_data['episodic'])} ep, {len(memory_data['lifelong'])} ll, Learning Rules: {len(self._behavioral_rules)}).")

            # Save vector index if enabled
            if self.vector_search_enabled:
                self._save_vector_index()

        except Exception as e: logger.error(f"Error saving memory state: {e}", exc_info=True)

    def load_memory(self):
        """ Loads memory state (episodic, lifelong, learning state) from the pickle file. Also loads/initializes vector index. """
        if not self.save_file_path or not os.path.exists(self.save_file_path):
            logger.info(f"No memory file found at '{self.save_file_path}'. Starting with empty memory (incl. default learning params).")
            # Ensure default learning state is saved if starting fresh
            self.save_learning_parameters()
            self.save_behavioral_rules()
            if self.vector_search_enabled: self._initialize_vector_index(force_create=False)
            return

        logger.info(f"Loading memory state from {self.save_file_path}...")
        try:
            with open(self.save_file_path, 'rb') as f: memory_data = pickle.load(f)

            # Load Episodic
            loaded_episodic = memory_data.get('episodic', []); self._episodic_memory = deque(loaded_episodic, maxlen=self.max_episodic_size)

            # Load Lifelong (Main Dictionary)
            self._lifelong_memory = memory_data.get('lifelong', {}); loaded_keys = memory_data.get('lifelong_keys', [])

            # Load and Validate Lifelong Age Tracker
            valid_keys = [k for k in loaded_keys if k in self._lifelong_memory]; self._lifelong_keys_by_age = deque(valid_keys, maxlen=self.max_lifelong_size)
            keys_in_deque = set(self._lifelong_keys_by_age); keys_to_remove = [k for k in self._lifelong_memory if k not in keys_in_deque]
            if keys_to_remove: logger.warning(f"Removing {len(keys_to_remove)} lifelong entries present in memory dict but missing from loaded age tracker."); [self._lifelong_memory.pop(k, None) for k in keys_to_remove]

            logger.info(f"Loaded base memory state ({len(self._episodic_memory)} ep, {len(self._lifelong_memory)} ll).")

            # --- Load Learning State (AFTER main lifelong dict is loaded) ---
            self.load_learning_state()
            # --- End Load Learning State ---

            # Initialize or Load Vector Index (and potentially rebuild if needed)
            if self.vector_search_enabled:
                 if not self._initialize_vector_index(force_create=False): logger.warning("Failed to initialize/load vector index after loading memory state. Vector search disabled."); self.vector_search_enabled = False

        except (EOFError, pickle.UnpicklingError, TypeError, AttributeError) as load_err:
            logger.error(f"Error loading memory file '{self.save_file_path}' ({type(load_err).__name__}). It might be corrupt, empty, or from an incompatible version. Clearing memory.", exc_info=False)
            self.clear_all_memory() # Start fresh if load fails badly
        except Exception as e:
            logger.error(f"Unexpected error loading memory state: {e}", exc_info=True);
            logger.warning("Starting with empty memory due to load failure.")
            self.clear_all_memory() # Start fresh on unexpected errors

# --- END OF FILE seed/memory_system.py ---