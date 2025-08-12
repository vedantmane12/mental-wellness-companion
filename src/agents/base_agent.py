"""
Base Agent class for all mental wellness agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np

from src.utils.logger import logger


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, agent_type: str):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            agent_type: Type of agent
        """
        self.name = name
        self.agent_type = agent_type
        self.active = True
        self.conversation_history = []
        self.metrics = {
            "interactions": 0,
            "successful_interactions": 0,
            "average_rating": 0.0
        }
        
        logger.info(f"Initialized {agent_type} agent: {name}")
    
    @abstractmethod
    def process(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input and generate response
        
        Args:
            state: Current conversation state
            context: Additional context
        
        Returns:
            Agent response with action and metadata
        """
        pass
    
    @abstractmethod
    def update(self, feedback: Dict[str, Any]):
        """
        Update agent based on feedback
        
        Args:
            feedback: Feedback data
        """
        pass
    
    def reset(self):
        """Reset agent state for new conversation"""
        self.conversation_history = []
        logger.debug(f"Reset {self.name} agent")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return self.metrics
    
    def log_interaction(self, success: bool = True, rating: float = 0.5):
        """Log interaction metrics"""
        self.metrics["interactions"] += 1
        if success:
            self.metrics["successful_interactions"] += 1
        
        # Update average rating
        n = self.metrics["interactions"]
        current_avg = self.metrics["average_rating"]
        self.metrics["average_rating"] = ((n - 1) * current_avg + rating) / n