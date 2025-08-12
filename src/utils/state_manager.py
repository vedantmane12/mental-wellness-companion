"""
State management utilities for encoding and decoding states
"""
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import torch

from src.utils.logger import logger
from config.settings import AGENT_CONFIG


@dataclass
class ConversationState:
    """Represents the current state of a conversation"""
    emotional_state: np.ndarray  # 5D: anxiety, depression, stress, anger, happiness
    engagement_level: float  # 0-1 scale
    conversation_history: List[Dict[str, str]]
    session_duration: float  # minutes
    time_since_last: float  # hours
    current_topic: Optional[str] = None
    risk_level: float = 0.0  # 0-1 scale
    
    def to_vector(self) -> np.ndarray:
        """Convert state to a fixed-size vector representation"""
        # Emotional state (5 dims)
        emotional = self.emotional_state
        
        # Engagement and risk (2 dims)
        engagement_risk = np.array([self.engagement_level, self.risk_level])
        
        # Temporal features (2 dims)
        temporal = np.array([
            min(self.session_duration / 60.0, 1.0),  # Normalize to 0-1
            min(self.time_since_last / 168.0, 1.0)   # Normalize to 0-1 (week max)
        ])
        
        # Conversation history features (extract key metrics)
        history_features = self._extract_history_features()
        
        # Combine all features
        state_vector = np.concatenate([
            emotional,
            engagement_risk,
            temporal,
            history_features
        ])
        
        # Pad or truncate to fixed size (256 dims as configured)
        target_size = 256
        if len(state_vector) < target_size:
            state_vector = np.pad(state_vector, (0, target_size - len(state_vector)))
        else:
            state_vector = state_vector[:target_size]
        
        return state_vector.astype(np.float32)
    
    def _extract_history_features(self) -> np.ndarray:
        """Extract features from conversation history"""
        if not self.conversation_history:
            return np.zeros(20)  # Return zero vector if no history
        
        features = []
        
        # Message count features
        total_messages = len(self.conversation_history)
        user_messages = sum(1 for m in self.conversation_history if m.get("role") == "user")
        agent_messages = sum(1 for m in self.conversation_history if m.get("role") == "assistant")
        
        features.extend([
            min(total_messages / 20.0, 1.0),
            user_messages / max(total_messages, 1),
            agent_messages / max(total_messages, 1)
        ])
        
        # Length features (average message length)
        user_lengths = [len(m.get("content", "")) for m in self.conversation_history if m.get("role") == "user"]
        agent_lengths = [len(m.get("content", "")) for m in self.conversation_history if m.get("role") == "assistant"]
        
        avg_user_length = np.mean(user_lengths) if user_lengths else 0
        avg_agent_length = np.mean(agent_lengths) if agent_lengths else 0
        
        features.extend([
            min(avg_user_length / 500.0, 1.0),
            min(avg_agent_length / 500.0, 1.0)
        ])
        
        # Sentiment/emotion indicators (simplified)
        last_user_message = ""
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "user":
                last_user_message = msg.get("content", "").lower()
                break
        
        # Simple keyword-based sentiment
        positive_keywords = ["good", "better", "happy", "great", "thank", "helpful"]
        negative_keywords = ["bad", "worse", "sad", "angry", "frustrated", "unhappy"]
        
        positive_count = sum(1 for word in positive_keywords if word in last_user_message)
        negative_count = sum(1 for word in negative_keywords if word in last_user_message)
        
        features.extend([positive_count / 6.0, negative_count / 6.0])
        
        # Pad to fixed size
        features = np.array(features)
        if len(features) < 20:
            features = np.pad(features, (0, 20 - len(features)))
        
        return features[:20]
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to PyTorch tensor"""
        return torch.from_numpy(self.to_vector())


class StateEncoder:
    """Encodes conversation states for the neural network"""
    
    def __init__(self, state_dim: int = 256):
        self.state_dim = state_dim
        logger.info(f"Initialized StateEncoder with dimension {state_dim}")
    
    def encode(self, state: ConversationState) -> torch.Tensor:
        """Encode a conversation state into a tensor"""
        return state.to_tensor()
    
    def encode_batch(self, states: List[ConversationState]) -> torch.Tensor:
        """Encode a batch of states"""
        tensors = [state.to_tensor() for state in states]
        return torch.stack(tensors)
    
    def decode_action(self, action_probs: torch.Tensor) -> Dict[str, Any]:
        """Decode action probabilities into discrete actions"""
        action_probs = action_probs.detach().numpy() if isinstance(action_probs, torch.Tensor) else action_probs
        
        # Split action probabilities into components
        strategy_probs = action_probs[:8]
        resource_probs = action_probs[8:14]
        tone_probs = action_probs[14:19]
        
        # Select actions (can use argmax or sample)
        strategy_idx = np.argmax(strategy_probs)
        resource_idx = np.argmax(resource_probs)
        tone_idx = np.argmax(tone_probs)
        
        return {
            "conversation_strategy": AGENT_CONFIG["conversation_strategies"][strategy_idx],
            "resource_type": AGENT_CONFIG["resource_types"][resource_idx],
            "response_tone": AGENT_CONFIG["response_tones"][tone_idx],
            "strategy_confidence": float(strategy_probs[strategy_idx]),
            "resource_confidence": float(resource_probs[resource_idx]),
            "tone_confidence": float(tone_probs[tone_idx])
        }