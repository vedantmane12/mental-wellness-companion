"""
Recommendation Agent using Contextual Bandits
Handles resource recommendations and timing optimization
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.rl.contextual_bandit import ContextualBandit
from src.utils.logger import logger
from config.settings import DATA_DIR


class RecommendationAgent(BaseAgent):
    """
    Agent responsible for resource recommendations using contextual bandits
    """
    
    def __init__(
        self,
        name: str = "RecommendationAgent",
        bandit_path: Optional[Path] = None
    ):
        """
        Initialize recommendation agent
        
        Args:
            name: Agent name
            bandit_path: Path to saved bandit state
        """
        super().__init__(name, "Recommendation")
        
        # Initialize contextual bandit
        self.bandit = ContextualBandit()
        
        # Load saved bandit if available
        if bandit_path and bandit_path.exists():
            self.bandit.load(bandit_path)
        else:
            # Try default location
            default_path = DATA_DIR / "models" / "contextual_bandit.json"
            if default_path.exists():
                self.bandit.load(default_path)
        
        # Resource metadata
        self.resource_metadata = self._load_resource_metadata()
        
        logger.info("Initialized RecommendationAgent with contextual bandit")
    
    def _load_resource_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load metadata about available resources"""
        return {
            "article": {
                "description": "Educational articles about mental health",
                "engagement_time": 5,  # minutes
                "effectiveness": 0.7
            },
            "exercise": {
                "description": "Interactive mental health exercises",
                "engagement_time": 10,
                "effectiveness": 0.8
            },
            "video": {
                "description": "Informative and supportive videos",
                "engagement_time": 8,
                "effectiveness": 0.75
            },
            "worksheet": {
                "description": "Structured worksheets for self-reflection",
                "engagement_time": 15,
                "effectiveness": 0.85
            },
            "meditation": {
                "description": "Guided meditation and mindfulness",
                "engagement_time": 10,
                "effectiveness": 0.8
            },
            "professional_referral": {
                "description": "Connection to professional help",
                "engagement_time": 0,
                "effectiveness": 1.0
            }
        }
    
    def process(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process state and recommend resources
        
        Args:
            state: Current conversation state
            context: Additional context
        
        Returns:
            Resource recommendation with metadata
        """
        try:
            # Get state vector
            state_vector = state.get("state_vector")
            if state_vector is None:
                state_vector = np.random.randn(256) * 0.1  # Fallback
            
            # Check if crisis - always recommend professional help
            risk_level = state.get("risk_level", 0.0)
            if risk_level > 0.8:
                return self._create_crisis_recommendation()
            
            # Select resource using bandit
            exploration_mode = "thompson" if context.get("training", False) else "greedy"
            resource, confidence = self.bandit.select_resource(
                state_vector,
                exploration_mode=exploration_mode
            )
            
            # Select timing
            user_preference = context.get("user_preference", None)
            hours, timing_category = self.bandit.select_timing(
                state_vector,
                user_preference=user_preference
            )
            
            # Get resource details
            metadata = self.resource_metadata.get(resource, {})
            
            # Create recommendation
            recommendation = {
                "resource_type": resource,
                "resource_metadata": metadata,
                "confidence": confidence,
                "timing_hours": hours,
                "timing_category": timing_category,
                "reason": self._generate_recommendation_reason(resource, state),
                "agent": self.name
            }
            
            # Log interaction
            self.log_interaction(success=True, rating=confidence)
            
            return recommendation
            
        except Exception as e:
            logger.error(f"Error in RecommendationAgent.process: {e}")
            return self._create_fallback_recommendation()
    
    def _create_crisis_recommendation(self) -> Dict[str, Any]:
        """Create crisis recommendation"""
        return {
            "resource_type": "professional_referral",
            "resource_metadata": self.resource_metadata["professional_referral"],
            "confidence": 1.0,
            "timing_hours": 0,  # Immediate
            "timing_category": "immediate",
            "reason": "Based on what you're sharing, I think it would be helpful to connect with a professional who can provide more specialized support.",
            "crisis": True,
            "agent": self.name
        }
    
    def _create_fallback_recommendation(self) -> Dict[str, Any]:
        """Create fallback recommendation"""
        return {
            "resource_type": "article",
            "resource_metadata": self.resource_metadata["article"],
            "confidence": 0.5,
            "timing_hours": 24,
            "timing_category": "daily",
            "reason": "Here's a resource that might be helpful.",
            "agent": self.name
        }
    
    def _generate_recommendation_reason(
        self,
        resource: str,
        state: Dict[str, Any]
    ) -> str:
        """Generate explanation for recommendation"""
        emotional_state = state.get("emotional_state", {})
        
        reasons = {
            "article": "This article can help you understand what you're experiencing better.",
            "exercise": "This exercise can help you practice coping strategies.",
            "video": "This video provides helpful insights and support.",
            "worksheet": "This worksheet can help you work through your thoughts systematically.",
            "meditation": "A meditation exercise might help you find some calm.",
            "professional_referral": "Speaking with a professional could provide additional support."
        }
        
        base_reason = reasons.get(resource, "This resource might be helpful.")
        
        # Add personalization based on emotional state
        if isinstance(emotional_state, dict):
            if emotional_state.get("anxiety", 0) > 0.7:
                base_reason += " It's particularly good for managing anxiety."
            elif emotional_state.get("depression", 0) > 0.7:
                base_reason += " It can help with low mood."
            elif emotional_state.get("stress", 0) > 0.7:
                base_reason += " It's designed to help reduce stress."
        
        return base_reason
    
    def update(self, feedback: Dict[str, Any]):
        """
        Update bandit based on feedback
        
        Args:
            feedback: Contains resource used, context, and reward
        """
        resource = feedback.get("resource_type")
        context = feedback.get("context", np.random.randn(256) * 0.1)
        reward = feedback.get("reward", 0.5)
        
        if resource:
            self.bandit.update_resource(resource, context, reward)
        
        # Update timing if provided
        timing_category = feedback.get("timing_category")
        engagement_maintained = feedback.get("engagement_maintained", True)
        
        if timing_category:
            self.bandit.update_timing(timing_category, engagement_maintained)
        
        # Save periodically
        if self.bandit.total_selections % 50 == 0:
            self.bandit.save()
        
        logger.debug(f"RecommendationAgent updated: resource={resource}, reward={reward:.2f}")
    
    def get_resource_statistics(self) -> Dict[str, Any]:
        """Get recommendation statistics"""
        return self.bandit.get_statistics()
    
    def get_best_resources_for_context(
        self,
        state_vector: np.ndarray,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Get top K resource recommendations for a context"""
        scores = {}
        
        for resource in self.bandit.arms.keys():
            # Calculate expected value
            mean_reward = self.bandit.arms[resource].get_mean_reward()
            context_score = self.bandit._compute_context_score(state_vector, resource)
            scores[resource] = mean_reward + 0.3 * context_score
        
        # Sort and return top K
        sorted_resources = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_resources[:top_k]