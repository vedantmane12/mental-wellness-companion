"""
Model configuration loader
Loads configuration from model_config.yaml
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass
from src.utils.logger import logger

# Get the config directory path
CONFIG_DIR = Path(__file__).parent
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"


@dataclass
class ModelConfig:
    """Model configuration container"""
    policy_network: Dict[str, Any]
    value_network: Dict[str, Any]
    state_features: Dict[str, Any]
    action_space: Dict[str, Any]
    reward_weights: Dict[str, float]
    training: Dict[str, Any]
    
    def __post_init__(self):
        # Ensure all required fields are present
        if not self.reward_weights:
            self.reward_weights = {
                "engagement": 0.4,
                "mood_improvement": 0.3,
                "resource_utilization": 0.2,
                "conversation_quality": 0.1,
                "safety_penalty": -1.0,
                "crisis_penalty": -10.0
            }


def load_model_config() -> ModelConfig:
    """Load model configuration from YAML file"""
    if not MODEL_CONFIG_PATH.exists():
        logger.warning(f"Model config file not found at {MODEL_CONFIG_PATH}, using defaults")
        return get_default_config()
    
    try:
        with open(MODEL_CONFIG_PATH, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dict to ModelConfig
        return ModelConfig(
            policy_network=config_dict.get('policy_network', {}),
            value_network=config_dict.get('value_network', {}),
            state_features=config_dict.get('state_features', {}),
            action_space=config_dict.get('action_space', {}),
            reward_weights=config_dict.get('reward_weights', {}),
            training=config_dict.get('training', {})
        )
    except Exception as e:
        logger.error(f"Error loading model config: {e}")
        return get_default_config()


def get_default_config() -> ModelConfig:
    """Get default model configuration"""
    return ModelConfig(
        policy_network={
            "input_dim": 256,
            "hidden_layers": [512, 256, 128],
            "activation": "relu",
            "dropout": 0.1,
            "output_dim": 19
        },
        value_network={
            "input_dim": 256,
            "hidden_layers": [512, 256, 128],
            "activation": "relu",
            "dropout": 0.1,
            "output_dim": 1
        },
        state_features={
            "emotional_dimensions": 5,
            "conversation_history_length": 5,
            "engagement_metrics": 3,
            "temporal_features": 2,
            "total_dim": 256
        },
        action_space={
            "conversation_strategy_dim": 8,
            "resource_recommendation_dim": 6,
            "response_tone_dim": 5,
            "check_in_timing_range": [1, 168]
        },
        reward_weights={
            "engagement": 0.4,
            "mood_improvement": 0.3,
            "resource_utilization": 0.2,
            "conversation_quality": 0.1,
            "safety_penalty": -1.0,
            "crisis_penalty": -10.0
        },
        training={
            "max_steps_per_episode": 10,
            "num_parallel_envs": 4,
            "normalize_advantage": True,
            "normalize_rewards": True,
            "entropy_coefficient": 0.01,
            "value_loss_coefficient": 0.5
        }
    )


# Load config on module import
model_config = load_model_config()

# Export
__all__ = ['model_config', 'ModelConfig', 'load_model_config']