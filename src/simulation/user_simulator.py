"""
User Simulator for generating realistic user behavior during RL training
Maintains conversation state and emotional consistency
"""
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.simulation.openai_client import OpenAIClient
from src.utils.logger import logger
from src.utils.helpers import clip_value
from config.settings import SAFETY_CONFIG


class UserMood(Enum):
    """User mood states"""
    CRISIS = "crisis"
    VERY_LOW = "very_low"
    LOW = "low"
    NEUTRAL = "neutral"
    GOOD = "good"
    VERY_GOOD = "very_good"


@dataclass
class UserSession:
    """Represents a user's session state"""
    persona: Dict[str, Any]
    emotional_state: Dict[str, float]
    conversation_history: List[Dict[str, str]]
    session_turn: int
    engagement_level: float
    mood_trajectory: List[float]
    resources_used: List[str]
    
    def get_mood_score(self) -> float:
        """Calculate overall mood score from emotional state"""
        weights = {
            "anxiety": -0.3,
            "depression": -0.3,
            "stress": -0.2,
            "anger": -0.1,
            "happiness": 0.4
        }
        
        score = sum(
            self.emotional_state.get(emotion, 0.5) * weight
            for emotion, weight in weights.items()
        )
        
        # Normalize to 0-1 range
        return clip_value((score + 0.5) / 1.0, 0.0, 1.0)
    
    def get_mood_category(self) -> UserMood:
        """Get categorical mood from score"""
        score = self.get_mood_score()
        
        if self._check_crisis():
            return UserMood.CRISIS
        elif score < 0.2:
            return UserMood.VERY_LOW
        elif score < 0.35:
            return UserMood.LOW
        elif score < 0.6:
            return UserMood.NEUTRAL
        elif score < 0.8:
            return UserMood.GOOD
        else:
            return UserMood.VERY_GOOD
    
    def _check_crisis(self) -> bool:
        """Check if user is in crisis based on conversation"""
        if not self.conversation_history:
            return False
        
        # Check last user message for crisis keywords
        for msg in reversed(self.conversation_history):
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                for keyword in SAFETY_CONFIG["crisis_keywords"]:
                    if keyword in content:
                        return True
                break
        
        # Check emotional state thresholds
        if (self.emotional_state.get("depression", 0) > 0.9 or
            self.emotional_state.get("anxiety", 0) > 0.9):
            return True
        
        return False


class UserSimulator:
    """Simulates user behavior for RL training"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """
        Initialize user simulator
        
        Args:
            openai_client: Optional OpenAI client instance
        """
        self.client = openai_client or OpenAIClient()
        self.active_sessions: Dict[str, UserSession] = {}
        
        # Response patterns based on mood
        self.response_patterns = {
            UserMood.CRISIS: {
                "response_length": (10, 50),
                "engagement_prob": 0.9,
                "dropout_prob": 0.1
            },
            UserMood.VERY_LOW: {
                "response_length": (5, 30),
                "engagement_prob": 0.6,
                "dropout_prob": 0.3
            },
            UserMood.LOW: {
                "response_length": (10, 60),
                "engagement_prob": 0.7,
                "dropout_prob": 0.2
            },
            UserMood.NEUTRAL: {
                "response_length": (20, 100),
                "engagement_prob": 0.8,
                "dropout_prob": 0.1
            },
            UserMood.GOOD: {
                "response_length": (30, 150),
                "engagement_prob": 0.9,
                "dropout_prob": 0.05
            },
            UserMood.VERY_GOOD: {
                "response_length": (40, 200),
                "engagement_prob": 0.95,
                "dropout_prob": 0.02
            }
        }
        
        logger.info("Initialized UserSimulator")
    
    def create_session(self, persona: Dict[str, Any]) -> str:
        """
        Create a new user session
        
        Args:
            persona: User persona profile
        
        Returns:
            Session ID
        """
        session_id = f"session_{persona['id']}_{random.randint(1000, 9999)}"
        
        # Initialize emotional state from persona
        initial_emotional_state = persona.get("initial_emotional_state", {
            "anxiety": 0.5,
            "depression": 0.5,
            "stress": 0.5,
            "anger": 0.2,
            "happiness": 0.3
        })
        
        session = UserSession(
            persona=persona,
            emotional_state=initial_emotional_state.copy(),
            conversation_history=[],
            session_turn=0,
            engagement_level=0.7,  # Start with moderate engagement
            mood_trajectory=[],
            resources_used=[]
        )
        
        self.active_sessions[session_id] = session
        logger.info(f"Created session {session_id} for persona {persona['id']}")
        
        return session_id
    
    def get_initial_message(self, session_id: str) -> str:
        """
        Generate initial user message to start conversation
        
        Args:
            session_id: Session identifier
        
        Returns:
            Initial user message
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return "Hello, I'm looking for some support."
        
        persona = session.persona
        concerns = persona.get("mental_health", {}).get("primary_concerns", ["general stress"])
        mood = session.get_mood_category()
        
        # Generate contextual opening based on persona and mood
        templates = {
            UserMood.CRISIS: [
                "I really need help right now. Everything feels overwhelming.",
                "I don't know what to do anymore. I'm at my breaking point."
            ],
            UserMood.VERY_LOW: [
                f"I've been struggling with {concerns[0]} and it's getting worse.",
                "I don't feel like myself lately. Everything is so hard."
            ],
            UserMood.LOW: [
                f"Hi, I've been dealing with {concerns[0]} and could use some support.",
                "I'm not doing great. Been having a tough time with things."
            ],
            UserMood.NEUTRAL: [
                f"Hello, I'm looking for help with {concerns[0]}.",
                "Hi there, I'd like to talk about some things that have been on my mind."
            ],
            UserMood.GOOD: [
                f"Hi! I'm working on my {concerns[0]} and would like some guidance.",
                "Hello, I'm trying to improve my mental wellness and could use advice."
            ],
            UserMood.VERY_GOOD: [
                "Hi! I'm feeling pretty good but want to maintain my progress.",
                "Hello! Looking for strategies to keep my mental health on track."
            ]
        }
        
        message = random.choice(templates.get(mood, templates[UserMood.NEUTRAL]))
        
        # Add to conversation history
        session.conversation_history.append({
            "role": "user",
            "content": message
        })
        
        return message
    
    def generate_response(
        self,
        session_id: str,
        agent_message: str,
        agent_action: Dict[str, Any]
    ) -> Tuple[str, Dict[str, float], bool]:
        """
        Generate user response to agent message
        
        Args:
            session_id: Session identifier
            agent_message: Agent's message
            agent_action: Agent's action details (strategy, resource, tone)
        
        Returns:
            Tuple of (user_response, metrics, continue_conversation)
        """
        session = self.active_sessions.get(session_id)
        if not session:
            logger.error(f"Session {session_id} not found")
            return "I need to go.", {"engagement": 0, "mood_change": 0}, False
        
        # Add agent message to history
        session.conversation_history.append({
            "role": "assistant",
            "content": agent_message,
            "action": agent_action
        })
        
        # Check for conversation limits
        session.session_turn += 1
        if session.session_turn > SAFETY_CONFIG["max_conversation_length"]:
            return "I need to take a break now. Thank you for talking with me.", {
                "engagement": session.engagement_level,
                "mood_change": 0,
                "helpfulness": 0.7
            }, False
        
        # Generate response using OpenAI
        try:
            response_data = self.client.simulate_user_response(
                persona=session.persona,
                conversation_history=session.conversation_history,
                agent_message=agent_message,
                emotional_state=session.emotional_state
            )
            
            # Update session state
            session.emotional_state = response_data.get("emotional_state", session.emotional_state)
            session.engagement_level = response_data.get("engagement_level", session.engagement_level)
            
            # Track mood trajectory
            mood_score = session.get_mood_score()
            session.mood_trajectory.append(mood_score)
            
            # Add user response to history
            user_response = response_data.get("response", "...")
            session.conversation_history.append({
                "role": "user",
                "content": user_response
            })
            
            # Track resource usage
            if agent_action.get("resource_type"):
                session.resources_used.append(agent_action["resource_type"])
            
            # Calculate metrics for RL reward
            metrics = {
                "engagement": response_data.get("engagement_level", 0.5),
                "mood_change": response_data.get("mood_change", 0.0),
                "helpfulness": response_data.get("helpfulness_rating", 0.5),
                "session_length": session.session_turn,
                "resource_acceptance": 1.0 if response_data.get("helpfulness_rating", 0) > 0.6 else 0.0
            }
            
            # Determine if conversation should continue
            continue_conversation = response_data.get("continue_conversation", True)
            
            # Check dropout probability based on mood
            mood = session.get_mood_category()
            dropout_prob = self.response_patterns[mood]["dropout_prob"]
            if random.random() < dropout_prob:
                continue_conversation = False
                user_response += " I think I need to go now."
            
            logger.debug(f"Session {session_id} - Turn {session.session_turn}: "
                        f"Engagement={metrics['engagement']:.2f}, "
                        f"Mood={mood_score:.2f}")
            
            return user_response, metrics, continue_conversation
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._generate_fallback_response(session), {
                "engagement": 0.3,
                "mood_change": -0.1,
                "helpfulness": 0.3
            }, False
    
    def _generate_fallback_response(self, session: UserSession) -> str:
        """Generate fallback response when API fails"""
        mood = session.get_mood_category()
        
        fallback_responses = {
            UserMood.CRISIS: "I really need professional help...",
            UserMood.VERY_LOW: "I don't know... everything is just too much.",
            UserMood.LOW: "I'm trying but it's hard to focus right now.",
            UserMood.NEUTRAL: "I understand what you're saying.",
            UserMood.GOOD: "That makes sense, I'll think about it.",
            UserMood.VERY_GOOD: "Thank you, that's helpful advice!"
        }
        
        return fallback_responses.get(mood, "I see...")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary of a session for evaluation
        
        Args:
            session_id: Session identifier
        
        Returns:
            Session summary with metrics
        """
        session = self.active_sessions.get(session_id)
        if not session:
            return {}
        
        # Calculate overall metrics
        mood_improvement = 0.0
        if len(session.mood_trajectory) >= 2:
            mood_improvement = session.mood_trajectory[-1] - session.mood_trajectory[0]
        
        summary = {
            "session_id": session_id,
            "persona_id": session.persona["id"],
            "total_turns": session.session_turn,
            "final_engagement": session.engagement_level,
            "mood_improvement": mood_improvement,
            "mood_trajectory": session.mood_trajectory,
            "resources_used": session.resources_used,
            "unique_resources": len(set(session.resources_used)),
            "final_emotional_state": session.emotional_state,
            "conversation_complete": session.session_turn > 3,  # At least 3 turns
            "crisis_detected": session.get_mood_category() == UserMood.CRISIS
        }
        
        return summary
    
    def reset_session(self, session_id: str):
        """Reset or remove a session"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Reset session {session_id}")
    
    def evaluate_conversation_quality(
        self,
        session_id: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality of a completed conversation
        
        Args:
            session_id: Session identifier
        
        Returns:
            Quality metrics
        """
        session = self.active_sessions.get(session_id)
        if not session or not session.conversation_history:
            return {"overall_quality": 0.0}
        
        # Use OpenAI to evaluate
        evaluation = self.client.evaluate_conversation_quality(
            conversation=session.conversation_history,
            persona=session.persona
        )
        
        # Add session-specific metrics
        evaluation["mood_improvement"] = (
            session.mood_trajectory[-1] - session.mood_trajectory[0]
            if len(session.mood_trajectory) >= 2 else 0.0
        )
        evaluation["engagement_retention"] = session.engagement_level
        evaluation["session_completion"] = min(session.session_turn / 10.0, 1.0)
        
        return evaluation


# Test function
def test_user_simulator():
    """Test user simulator"""
    from src.simulation.persona_generator import PersonaGenerator
    
    # Generate a test persona
    generator = PersonaGenerator()
    personas = generator.generate_batch(batch_size=1, save=False)
    persona = personas[0]
    
    # Create simulator
    simulator = UserSimulator()
    
    # Create session
    session_id = simulator.create_session(persona)
    
    # Get initial message
    initial_msg = simulator.get_initial_message(session_id)
    logger.info(f"User: {initial_msg}")
    
    # Simulate a conversation turn
    agent_message = "I hear that you're going through a difficult time. Can you tell me more about what's been troubling you?"
    agent_action = {
        "conversation_strategy": "empathetic_listening",
        "resource_type": None,
        "response_tone": "supportive"
    }
    
    response, metrics, continue_conv = simulator.generate_response(
        session_id, agent_message, agent_action
    )
    
    logger.info(f"Agent: {agent_message}")
    logger.info(f"User: {response}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Continue: {continue_conv}")
    
    # Get session summary
    summary = simulator.get_session_summary(session_id)
    logger.info(f"Session summary: {summary}")
    
    return simulator, session_id


if __name__ == "__main__":
    test_user_simulator()