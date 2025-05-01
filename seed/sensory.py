# --- START OF FILE seed/sensory.py ---

# RSIAI/seed/sensory.py
"""
Defines the Seed_SensoryRefiner class.
Processes raw state information (simulated or real VM snapshot) into a
structured dictionary suitable for Seed core reasoning.
"""
import time
import json
import os
import logging
import traceback
import hashlib
from typing import Dict, Any, Optional, Union

# Import updated config constants using relative import
from .config import SEED_INITIAL_GOAL # Use updated constant name

logger = logging.getLogger(__name__)

# Type alias for the output dictionary
RefinedInput = Dict[str, Any]

# Renamed class
class Seed_SensoryRefiner:
    """ Processes raw system state snapshots into a structured dictionary for Seed Core. """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """ Initializes the Sensory Refiner. """
        self.config = config if config else {}
        self.use_nn = False # NN mode not implemented
        self.model = None
        logger.info(f"Seed Sensory Refiner Initialized (Mode: Rules).") # Renamed log

    def refine(self, raw_state_data: Optional[Dict[str, Any]]) -> Optional[RefinedInput]:
        """
        Transforms raw state data snapshot from VMService into a refined dictionary.

        Args:
            raw_state_data (dict): Dictionary containing raw state snapshot from VMService.
                                   Expected keys: 'timestamp', 'filesystem', 'resources',
                                   'mode', 'cwd', 'target_path_hint', 'probe_errors'(opt),
                                   'parsing_error'(opt).

        Returns:
            Optional[RefinedInput]: A dictionary containing structured sensory information,
                                    or None if refinement fails or input is invalid.
        """
        if not raw_state_data or not isinstance(raw_state_data, dict):
            logger.error("SensoryRefiner Error: Invalid raw_state_data (None or not dict).")
            return None
        if raw_state_data.get("error"): # Check for explicit error from VM Service
            logger.error(f"SensoryRefiner Error: Raw state contains error from VMService: {raw_state_data['error']}")
            # Return a minimal error state? Or None? Returning None for now.
            return None
        if raw_state_data.get("probe_errors"):
            logger.warning(f"SensoryRefiner: Raw state includes probe errors: {raw_state_data['probe_errors']}")
        if raw_state_data.get("parsing_error"):
            logger.warning(f"SensoryRefiner: Raw state includes parsing error: {raw_state_data['parsing_error']}")

        return self._refine_rules(raw_state_data)

    def _refine_rules(self, raw_state_data: Dict[str, Any]) -> Optional[RefinedInput]:
        """ Refines sensory input using rule-based logic into a structured dictionary. """
        refined: RefinedInput = {
            'timestamp': raw_state_data.get('timestamp', time.time()),
            'summary': {}, # Top-level summary metrics
            'filesystem_snapshot': None, # Simplified view focused on target/CWD
            'resource_usage': None, # Dict with cpu, mem, disk percentages
            'process_summary': None, # Placeholder for process info (e.g., count, top consumers)
            'last_command_outcome': None, # Dict summarizing last command result (sim only currently)
            'errors_detected': [], # List of errors detected during probing/parsing
            'target_status': {}, # Dict summarizing status of target_path_hint
            'system_mode': raw_state_data.get('mode', 'unknown'), # 'simulation', 'docker', 'subprocess'
            'cwd': raw_state_data.get('cwd', '/'), # Current working directory reported by VMService
        }
        error_list = refined['errors_detected']
        if raw_state_data.get('probe_errors'): error_list.extend(raw_state_data['probe_errors'])
        if raw_state_data.get('parsing_error'): error_list.append(f"VMService parsing error: {raw_state_data['parsing_error']}")

        try:
            # 1. Resources
            raw_res = raw_state_data.get('resources', {})
            refined['resource_usage'] = {
                'cpu_percent': round(float(raw_res.get('cpu_load_percent', 0.0) or 0.0), 1),
                'memory_percent': round(float(raw_res.get('memory_usage_percent', 0.0) or 0.0), 1),
                'disk_percent': round(float(raw_res.get('disk_usage_percent', 0.0) or 0.0), 1),
            }
            refined['summary']['cpu_load'] = refined['resource_usage']['cpu_percent']
            refined['summary']['mem_load'] = refined['resource_usage']['memory_percent']

            # 2. Filesystem & Target Status
            raw_fs = raw_state_data.get('filesystem', {})
            target_hint_path = raw_state_data.get('target_path_hint')
            current_cwd = refined['cwd']
            # Initialize filesystem snapshot part
            fs_summary = {
                'target_path_hint': target_hint_path,
                'cwd_listing_present': False, # Whether listing for CWD was obtained
                'parent_dir_exists': None # Whether target hint's parent dir seems accessible
            }

            # Check CWD info if available in raw data
            cwd_info_raw = raw_fs.get(current_cwd)
            if isinstance(cwd_info_raw, dict) and cwd_info_raw.get('content_listing') is not None:
                 fs_summary['cwd_listing_present'] = True

            # Process target hint status
            target_status = {'exists': False, 'path': target_hint_path, 'type': None, 'hint_present': None, 'is_dir': False}
            if target_hint_path:
                target_info_raw = raw_fs.get(target_hint_path) # Get data specific to the hinted path
                if isinstance(target_info_raw, dict):
                    # Use content hint from the initial goal as fallback if needed
                    content_hint = SEED_INITIAL_GOAL.get('content_hint')
                    target_status = self._summarize_file_info(target_info_raw, target_hint_path, content_hint)

                    # Check parent directory existence based on target info
                    # If target exists OR if error isn't 'No such file', parent likely exists.
                    if target_status.get('exists') or \
                       ('No such file or directory' not in target_info_raw.get('error','')):
                         fs_summary['parent_dir_exists'] = True
                    else:
                         fs_summary['parent_dir_exists'] = False
                else:
                     # No info retrieved for target path
                     target_status['exists'] = False
                     fs_summary['parent_dir_exists'] = None # Cannot determine parent status either

            refined['target_status'] = target_status
            refined['filesystem_snapshot'] = fs_summary # Add the summary dict
            refined['summary']['target_exists'] = target_status.get('exists', False)
            refined['summary']['target_type'] = target_status.get('type')
            refined['summary']['target_hint_present'] = target_status.get('hint_present', False)

            # 3. Processes (Placeholder)
            refined['process_summary'] = None # E.g., could be {'count': N, 'top': [...]}
            refined['summary']['process_count'] = 0 # Placeholder

            # 4. Last Command Outcome (Simulation only currently)
            if refined['system_mode'] == 'simulation':
                last_cmd = raw_state_data.get('last_command_result')
                if isinstance(last_cmd, dict):
                    refined['last_command_outcome'] = {
                        'success': last_cmd.get('success', False),
                        'command': last_cmd.get('command', 'N/A'),
                        'reason': last_cmd.get('reason'),
                        'exit_code': last_cmd.get('exit_code'),
                        'has_stderr': bool(last_cmd.get('stderr')),
                        'stderr_snippet': (last_cmd.get('stderr') or '')[:100] # Limit length
                    }
                    # Add to top-level summary only if available
                    refined['summary']['last_cmd_success'] = refined['last_command_outcome']['success']

            # 5. Derive Overall Summary Metrics & Health
            refined['summary']['error_count'] = len(error_list)
            # Simple health estimation based on errors and resource load
            health = 1.0 - (min(refined['summary']['error_count'], 3) * 0.25) \
                         - (refined['summary'].get('cpu_load', 0)/100 * 0.15) \
                         - (refined['summary'].get('mem_load', 0)/100 * 0.10)
            refined['summary']['estimated_health'] = round(max(0.0, min(1.0, health)), 2)

        except Exception as e:
            logger.error(f"Seed SensoryRefiner rule processing error: {e}", exc_info=True) # Renamed log
            refined['errors_detected'].append(f"Refinement Error: {e}")
            refined['summary']['estimated_health'] = 0.0 # Indicate failure in health
            # Return the partially refined dict with the error noted? Or None?
            # Returning the dict allows caller to see partial state + errors.
            return refined

        return refined

    def _summarize_file_info(self, file_data_raw: Dict[str, Any], path: str, content_hint: Optional[str]) -> Dict[str, Any]:
        """ Helper to summarize file info from raw VMService state data for a specific path. """
        summary = {
            'exists': False, 'path': path, 'type': None, 'hint_present': None,
            'is_dir': False, 'size_bytes': None, 'content_hash': None,
            'mtime': None, 'owner': None, 'permissions': None, 'raw_stat': None,
            'error': None
        }
        if not isinstance(file_data_raw, dict):
             summary['error'] = "Invalid input data format"
             return summary

        summary['exists'] = file_data_raw.get('exists', False) # Use 'exists' flag from VMService probe
        summary['error'] = file_data_raw.get('error') # Copy error if present

        if not summary['exists']: return summary # Nothing more to parse if it doesn't exist

        summary['type'] = file_data_raw.get('type', 'unknown');
        summary['is_dir'] = (summary['type'] == 'directory')
        summary['owner'] = file_data_raw.get('owner')
        summary['permissions'] = file_data_raw.get('perms_symbolic') or file_data_raw.get('perms_octal') or file_data_raw.get('perms') # Prefer symbolic
        summary['mtime'] = file_data_raw.get('mtime') # Could be timestamp string or None
        summary['size_bytes'] = file_data_raw.get('size_bytes')
        summary['raw_stat'] = file_data_raw.get('stat_output') # Raw output if available

        # Handle content-specific checks only for files
        if summary['type'] == 'file':
            content = file_data_raw.get('content') # Content might be present in simulation or read_file result
            if content is not None: # If content was explicitly provided
                try:
                    content_str = content if isinstance(content, str) else str(content)
                    content_bytes = content_str.encode('utf-8', errors='ignore')
                    # Update size based on actual content if provided (might differ from stat)
                    summary['size_bytes'] = len(content_bytes)
                    summary['content_hash'] = hashlib.sha256(content_bytes).hexdigest()[:16] # Short hash
                    # Check for content hint presence
                    if content_hint:
                         summary['hint_present'] = (content_hint in content_str)
                    else:
                         summary['hint_present'] = None # Hint wasn't provided to check against
                except Exception as e:
                     logger.warning(f"Error processing file content for {path}: {e}")
                     summary['content_hash'] = None
                     summary['hint_present'] = None
            else:
                # Content not provided, can't check hint or hash
                summary['hint_present'] = None
                summary['content_hash'] = None
        else: # Directories or other types don't have content hints
            summary['hint_present'] = False

        return summary

# --- END OF FILE seed/sensory.py ---