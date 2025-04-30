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

# Import updated config constants
from ..config import SEED_INITIAL_GOAL # Use updated constant name

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
        if raw_state_data.get("error"):
            logger.error(f"SensoryRefiner Error: Raw state contains error: {raw_state_data['error']}")
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
            'summary': {},
            'filesystem_snapshot': None,
            'resource_usage': None,
            'process_summary': None,
            'last_command_outcome': None,
            'errors_detected': [],
            'target_status': {},
            'system_mode': raw_state_data.get('mode', 'unknown'),
            'cwd': raw_state_data.get('cwd', '/'),
        }
        error_list = refined['errors_detected']

        try:
            # 1. Resources
            raw_res = raw_state_data.get('resources', {})
            refined['resource_usage'] = {
                'cpu_percent': float(raw_res.get('cpu_load_percent', 0.0) or 0.0),
                'memory_percent': float(raw_res.get('memory_usage_percent', 0.0) or 0.0),
                'disk_percent': float(raw_res.get('disk_usage_percent', 0.0) or 0.0),
            }
            refined['summary']['cpu_load'] = refined['resource_usage']['cpu_percent']
            refined['summary']['mem_load'] = refined['resource_usage']['memory_percent']

            # 2. Filesystem & Target Status
            raw_fs = raw_state_data.get('filesystem', {})
            target_hint_path = raw_state_data.get('target_path_hint')
            current_cwd = refined['cwd']
            fs_summary = {'target_path_hint': target_hint_path, 'cwd_listing_present': False, 'parent_dir_exists': None}
            target_status = {'exists': False, 'path': None, 'type': None, 'hint_present': None, 'is_dir': False}

            cwd_info = raw_fs.get(current_cwd)
            if isinstance(cwd_info, dict) and cwd_info.get('content_listing') is not None:
                 fs_summary['cwd_listing_present'] = True

            if target_hint_path:
                target_info_raw = raw_fs.get(target_hint_path)
                target_status['path'] = target_hint_path

                if isinstance(target_info_raw, dict):
                    # Use updated constant name for fallback hint
                    content_hint = SEED_INITIAL_GOAL.get('content_hint','')
                    target_status = self._summarize_file_info(target_info_raw, target_hint_path, content_hint)
                else:
                     target_status['exists'] = False

                parent_dir = os.path.dirname(target_hint_path) if target_hint_path != '/' else '/'
                if isinstance(target_info_raw, dict) and target_info_raw.get('exists'):
                     fs_summary['parent_dir_exists'] = True
                elif isinstance(target_info_raw, dict) and 'No such file or directory' not in target_info_raw.get('error',''):
                     fs_summary['parent_dir_exists'] = True
                else:
                     fs_summary['parent_dir_exists'] = False

            refined['target_status'] = target_status
            refined['filesystem_snapshot'] = fs_summary
            refined['summary']['target_exists'] = target_status.get('exists', False)
            refined['summary']['target_type'] = target_status.get('type')
            refined['summary']['target_hint_present'] = target_status.get('hint_present', False)

            # 3. Processes
            refined['process_summary'] = None
            refined['summary']['process_count'] = 0

            # 4. Last Command Outcome
            if refined['system_mode'] == 'simulation': # Check against vm_service's simulation mode
                last_cmd = raw_state_data.get('last_command_result')
                if isinstance(last_cmd, dict):
                    refined['last_command_outcome'] = {
                        'success': last_cmd.get('success', False),
                        'command': last_cmd.get('command', 'N/A'),
                        'reason': last_cmd.get('reason'), 'exit_code': last_cmd.get('exit_code'),
                        'has_stderr': bool(last_cmd.get('stderr')),
                        'stderr_snippet': (last_cmd.get('stderr') or '')[:100] }
                    refined['summary']['last_probe_cmd_success'] = refined['last_command_outcome']['success']

            # 5. Add VMService reported errors
            if raw_state_data.get('probe_errors'):
                 error_list.extend(raw_state_data['probe_errors'])
            if raw_state_data.get('parsing_error'):
                 error_list.append(f"VMService parsing error: {raw_state_data['parsing_error']}")

            # 6. Derive Overall Summary Metrics & Health
            refined['summary']['error_count'] = len(error_list)
            health = 1.0 - (min(refined['summary']['error_count'], 3) * 0.25) \
                         - (refined['summary'].get('cpu_load', 0)/100 * 0.15) \
                         - (refined['summary'].get('mem_load', 0)/100 * 0.10)
            refined['summary']['estimated_health'] = max(0.0, min(1.0, health))

        except Exception as e:
            logger.error(f"Seed SensoryRefiner rule processing error: {e}", exc_info=True) # Renamed log
            return None

        return refined

    def _summarize_file_info(self, file_data_raw: Dict[str, Any], path: str, content_hint: Optional[str]) -> Dict[str, Any]:
        """ Helper to summarize file info from raw VMService state data. """
        summary = { 'exists': False, 'path': path, 'type': None, 'hint_present': None, 'is_dir': False, 'size_bytes': None, 'content_hash': None, 'mtime': None, 'owner': None, 'permissions': None, 'raw_stat': None }
        if not isinstance(file_data_raw, dict): return summary
        summary['exists'] = file_data_raw.get('exists', True)
        if not summary['exists']: return summary
        summary['type'] = file_data_raw.get('type', 'unknown'); summary['is_dir'] = summary['type'] == 'directory'; summary['owner'] = file_data_raw.get('owner'); summary['permissions'] = file_data_raw.get('perms_symbolic') or file_data_raw.get('perms_octal') or file_data_raw.get('perms'); summary['mtime'] = file_data_raw.get('mtime'); summary['size_bytes'] = file_data_raw.get('size_bytes'); summary['raw_stat'] = file_data_raw.get('stat_output')
        if summary['type'] == 'file':
            content = file_data_raw.get('content') # Content might be present in simulation mode
            if content is not None:
                try: content_str = content if isinstance(content, str) else str(content); content_bytes = content_str.encode('utf-8', errors='ignore'); summary['size_bytes'] = len(content_bytes); summary['content_hash'] = hashlib.sha256(content_bytes).hexdigest()[:16]; summary['hint_present'] = bool(content_hint and content_hint in content_str)
                except Exception as e: logger.warning(f"Error processing file content for {path}: {e}")
            else: summary['hint_present'] = None; summary['content_hash'] = None
        else: summary['hint_present'] = False
        return summary

# --- END OF FILE seed/sensory.py ---