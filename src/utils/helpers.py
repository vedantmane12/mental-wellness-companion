"""
Helper functions for the Mental Wellness Companion
"""
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import torch

from src.utils.logger import logger


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seeds set to {seed}")


def generate_id(prefix: str = "id") -> str:
    """Generate a unique ID with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = hashlib.md5(str(random.random()).encode()).hexdigest()[:6]
    return f"{prefix}_{timestamp}_{random_str}"


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 2):
    """Save data to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.debug(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load data from JSON file"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.debug(f"Loaded JSON from {filepath}")
    return data


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    vec1_norm = normalize_vector(vec1)
    vec2_norm = normalize_vector(vec2)
    return np.dot(vec1_norm, vec2_norm)


def clip_value(value: float, min_val: float, max_val: float) -> float:
    """Clip a value to a range"""
    return max(min_val, min(value, max_val))


def moving_average(values: List[float], window: int = 10) -> List[float]:
    """Calculate moving average of a list of values"""
    if len(values) < window:
        return values
    
    averaged = []
    for i in range(len(values) - window + 1):
        window_values = values[i:i + window]
        averaged.append(sum(window_values) / window)
    
    return averaged


def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation history for display"""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        formatted.append(f"{role.upper()}: {content}")
    
    return "\n".join(formatted)


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_checkpoint_path(epoch: int, model_name: str = "model") -> Path:
    """Create a checkpoint filepath"""
    from config.settings import DATA_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_epoch_{epoch}_{timestamp}.pt"
    return DATA_DIR / "models" / filename