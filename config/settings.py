"""
Global configuration settings for the Mental Wellness Companion
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
PROMPTS_DIR = CONFIG_DIR / "prompts"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "personas").mkdir(exist_ok=True)
(DATA_DIR / "conversations").mkdir(exist_ok=True)
(DATA_DIR / "knowledge").mkdir(exist_ok=True)
(DATA_DIR / "models").mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")  # Optional
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

# Validate API keys
if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
    raise ValueError("Please set your OPENAI_API_KEY in the .env file")

# Training Configuration
TRAINING_CONFIG = {
    "batch_size": int(os.getenv("BATCH_SIZE", "32")),
    "learning_rate": float(os.getenv("LEARNING_RATE", "0.0003")),
    "ppo_epochs": int(os.getenv("PPO_EPOCHS", "10")),
    "ppo_clip": float(os.getenv("PPO_CLIP", "0.2")),
    "gamma": float(os.getenv("GAMMA", "0.99")),
    "gae_lambda": float(os.getenv("GAE_LAMBDA", "0.95")),
    "buffer_size": 10000,
    "episode_length": 10,  # Max conversation turns per episode
    "num_episodes": 1000,  # Total training episodes
}

# Agent Configuration
AGENT_CONFIG = {
    "conversation_strategies": [
        "empathetic_listening",
        "cognitive_behavioral",
        "validation",
        "problem_solving",
        "mindfulness",
        "motivational",
        "psychoeducation",
        "supportive"
    ],
    "resource_types": [
        "article",
        "exercise",
        "video",
        "worksheet",
        "meditation",
        "professional_referral"
    ],
    "response_tones": ["supportive", "encouraging", "gentle", "direct", "challenging"],
}

# Safety Configuration
SAFETY_CONFIG = {
    "crisis_keywords": [
        "suicide", "kill myself", "end it all", "not worth living",
        "self-harm", "hurt myself", "die", "death wish"
    ],
    "max_conversation_length": 20,
    "professional_referral_threshold": 0.8,
    "crisis_response": """I'm very concerned about what you're sharing. Your safety is the top priority. 
    Please reach out to a crisis helpline immediately:
    - National Suicide Prevention Lifeline: 988
    - Crisis Text Line: Text HOME to 741741
    - International: https://findahelpline.com
    
    Would you like me to provide additional emergency resources?"""
}

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"