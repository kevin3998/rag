# src/utils/file_operations.py
"""
Module: file_operations
Functionality: Provides utility functions for file input/output operations.
               This includes loading and saving JSON data, as well as managing
               checkpoint files for resuming long-running processes.
"""
import json
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_json_data(file_path: str) -> Any:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        raise

def save_json_data(data: Any, file_path: str, indent: int = 2):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        logger.info(f"Data successfully saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")
        raise

def load_checkpoint(checkpoint_path: str, default_checkpoint: dict) -> dict:
    """
    Loads checkpoint data from a file.
    Crucially, it converts the 'processed_ids' list back into a set for efficient lookups.
    """
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)

            loaded_checkpoint = {**default_checkpoint, **checkpoint}

            # --- FIX: Ensure processed_ids is loaded as a set ---
            processed_ids_list = loaded_checkpoint.get("processed_ids", [])
            if isinstance(processed_ids_list, list):
                loaded_checkpoint["processed_ids"] = set(processed_ids_list)
            else: # If it's something else, default to an empty set
                loaded_checkpoint["processed_ids"] = set()
            # --- END FIX ---

            logger.info(
                f"Checkpoint loaded from {checkpoint_path} | Processed: {loaded_checkpoint.get('total_processed', 0)} | Elapsed: {loaded_checkpoint.get('total_elapsed', 0.0):.1f}s"
            )
            return loaded_checkpoint
        except Exception as e:
            logger.warning(f"Checkpoint loading failed from {checkpoint_path}: {e}. Starting fresh.")
            return default_checkpoint

    logger.info("No checkpoint found. Starting fresh.")
    return default_checkpoint


def save_checkpoint(checkpoint_path: str, data_to_save: dict):
    """
    Saves checkpoint data to a file.
    Crucially, it converts the 'processed_ids' set into a list before saving.
    """
    try:
        # Work on a copy to avoid modifying the original dict in memory
        data_copy = data_to_save.copy()

        # --- FIX: Convert set to list for JSON serialization ---
        if 'processed_ids' in data_copy and isinstance(data_copy['processed_ids'], set):
            data_copy['processed_ids'] = list(data_copy['processed_ids'])
        # --- END FIX ---

        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data_copy, f, indent=2, ensure_ascii=False)

        # Using logger.debug for frequent saves to avoid cluttering the log
        logger.debug(f"Checkpoint saved to {checkpoint_path}")
    except TypeError as te:
        logger.error(f"CRITICAL: Failed to save checkpoint due to TypeError (object not serializable): {te}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")

def cleanup_checkpoint(checkpoint_path: str):
    """Safely removes the checkpoint file."""
    if os.path.exists(checkpoint_path):
        try:
            os.remove(checkpoint_path)
            logger.info(f"Checkpoint file {os.path.basename(checkpoint_path)} cleaned up.")
        except Exception as e:
            logger.warning(f"Checkpoint cleanup failed for {checkpoint_path}: {e}")