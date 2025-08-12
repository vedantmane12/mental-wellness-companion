"""
Reinforcement Learning Environment for Mental Wellness Companion
Gym-like environment for training agents
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import random

from src.simulation.persona_generator import PersonaGenerator
from src.simulation.user_simulator import UserSimulator
from src.utils.logger import logger, training_logger
from src.utils.state_manager import ConversationState, StateEncoder
from src.utils.helpers import clip_value
from config.settings import AGENT_CONFIG, TRAINING_CONFIG
from config.model_config import model_config


class MentalWellnessEnv(gym.Env):
    """
    Gym environment for mental wellness companion training
    """
    
    def __init__(
        self,
        personas: Optional[List[Dict[str, Any]]] = None,
        max_episode_length: int = 10,
        training_mode: bool = True
    ):
        """
        Initialize the environment
        
        Args:
            personas: List of personas to use (if None, generates new ones)
            max_episode_length: Maximum conversation turns per episode
            training_mode: Whether in training mode (affects randomness)
        """
        super().__init__()
        
        # Initialize components
        self.persona_generator = PersonaGenerator()
        self.user_simulator = UserSimulator()
        self.state_encoder = StateEncoder()
        
        # Load or generate personas
        if personas is None:
            logger.info("No personas provided, generating new batch...")
            self.personas = self.persona_generator.generate_batch(
                batch_size=100,
                prefix="training"
            )
        else:
            self.personas = personas
        
        logger.info(f"Environment initialized with {len(self.personas)} personas")
        
        # Environment configuration
        self.max_episode_length = max_episode_length
        self.training_mode = training_mode
        
        # Define action and observation spaces
        self._define_spaces()
        
        # Episode state
        self.current_episode = 0
        self.current_step = 0
        self.current_session_id = None
        self.current_persona = None
        self.current_state = None
        self.episode_rewards = []
        self.episode_metrics = []
        
        # Training statistics
        self.total_episodes = 0
        self.successful_episodes = 0
        self.crisis_episodes = 0
        
    def _define_spaces(self):
        """Define action and observation spaces"""
        # State space: 256-dimensional continuous vector
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(256,),
            dtype=np.float32
        )
        
        # Action space: Multi-discrete for different action components
        # [conversation_strategy(8), resource_type(6), response_tone(5)]
        self.action_space = spaces.MultiDiscrete([8, 6, 5])
        
        logger.info(f"Observation space: {self.observation_space}")
        logger.info(f"Action space: {self.action_space}")
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment for a new episode
        
        Args:
            seed: Random seed
            options: Additional options
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Select a random persona
        self.current_persona = random.choice(self.personas)
        
        # Create new session
        self.current_session_id = self.user_simulator.create_session(self.current_persona)
        
        # Reset episode state
        self.current_step = 0
        self.episode_rewards = []
        self.episode_metrics = []
        
        # Get initial user message
        initial_message = self.user_simulator.get_initial_message(self.current_session_id)
        
        # Create initial state
        session = self.user_simulator.active_sessions[self.current_session_id]
        self.current_state = ConversationState(
            emotional_state=np.array(list(session.emotional_state.values())),
            engagement_level=session.engagement_level,
            conversation_history=session.conversation_history,
            session_duration=0.0,
            time_since_last=0.0,
            current_topic="initial",
            risk_level=0.0
        )
        
        # Get observation
        observation = self.current_state.to_vector()
        
        # Info dict
        info = {
            "persona_id": self.current_persona["id"],
            "initial_message": initial_message,
            "emotional_state": session.emotional_state,
            "episode": self.current_episode
        }
        
        self.current_episode += 1
        training_logger.info(f"Episode {self.current_episode} started with persona {self.current_persona['id']}")
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take a step in the environment
        
        Args:
            action: Action to take [strategy_idx, resource_idx, tone_idx]
        
        Returns:
            observation, reward, terminated, truncated, info
        """
        self.current_step += 1
        
        # Decode action
        action_dict = self._decode_action(action)
        
        # Generate agent message based on action
        agent_message = self._generate_agent_message(action_dict)
        
        # Get user response
        user_response, metrics, continue_conversation = self.user_simulator.generate_response(
            self.current_session_id,
            agent_message,
            action_dict
        )
        
        # Calculate reward
        reward = self._calculate_reward(metrics, action_dict)
        self.episode_rewards.append(reward)
        self.episode_metrics.append(metrics)
        
        # Update state
        session = self.user_simulator.active_sessions[self.current_session_id]
        self.current_state = ConversationState(
            emotional_state=np.array(list(session.emotional_state.values())),
            engagement_level=session.engagement_level,
            conversation_history=session.conversation_history,
            session_duration=self.current_step * 2.0,  # Assume 2 minutes per turn
            time_since_last=0.0,
            current_topic=action_dict["conversation_strategy"],
            risk_level=self._calculate_risk_level(session)
        )
        
        # Get observation
        observation = self.current_state.to_vector()
        
        # Check termination conditions
        terminated = not continue_conversation or self.current_step >= self.max_episode_length
        truncated = False
        
        # Check for crisis
        if session.get_mood_category().value == "crisis":
            self.crisis_episodes += 1
            terminated = True
            reward -= 5.0  # Penalty for not preventing crisis
        
        # Success metrics
        if terminated and metrics.get("engagement", 0) > 0.6:
            self.successful_episodes += 1
        
        # Info dict
        info = {
            "step": self.current_step,
            "user_response": user_response,
            "agent_message": agent_message,
            "metrics": metrics,
            "action": action_dict,
            "emotional_state": session.emotional_state,
            "mood_score": session.get_mood_score(),
            "continue": continue_conversation
        }
        
        # Log step
        training_logger.debug(
            f"Step {self.current_step}: "
            f"Action={action_dict['conversation_strategy']}, "
            f"Reward={reward:.2f}, "
            f"Engagement={metrics['engagement']:.2f}"
        )
        
        # Episode summary when done
        if terminated:
            self._log_episode_summary()
        
        return observation, reward, terminated, truncated, info
    
    def _decode_action(self, action: np.ndarray) -> Dict[str, Any]:
        """Decode discrete action indices to action dict"""
        strategy_idx = int(action[0])
        resource_idx = int(action[1])
        tone_idx = int(action[2])
        
        return {
            "conversation_strategy": AGENT_CONFIG["conversation_strategies"][strategy_idx],
            "resource_type": AGENT_CONFIG["resource_types"][resource_idx],
            "response_tone": AGENT_CONFIG["response_tones"][tone_idx]
        }
    
    def _generate_agent_message(self, action_dict: Dict[str, Any]) -> str:
        """
        Generate agent message based on action
        
        This is a simplified version - in production, would use more sophisticated NLG
        """
        strategy = action_dict["conversation_strategy"]
        tone = action_dict["response_tone"]
        resource = action_dict["resource_type"]
        
        # Template-based generation (simplified)
        templates = {
            "empathetic_listening": {
                "supportive": "I hear how difficult this is for you. Your feelings are completely valid. Can you tell me more about {topic}?",
                "encouraging": "You're being so brave by sharing this. What you're feeling makes total sense. What aspect is most challenging?",
                "gentle": "Thank you for trusting me with this. It sounds really tough. Would you like to explore {topic} together?",
                "direct": "I understand you're struggling with {topic}. Let's focus on what we can work on right now.",
                "challenging": "I hear you, and I wonder if we could look at {topic} from a different angle?"
            },
            "cognitive_behavioral": {
                "supportive": "Let's examine these thoughts together. When you think about {topic}, what goes through your mind?",
                "encouraging": "You're already showing insight here. Can we identify any thought patterns around {topic}?",
                "gentle": "Sometimes our thoughts can influence how we feel. What thoughts come up about {topic}?",
                "direct": "Let's challenge that thought. What evidence do you have for and against it?",
                "challenging": "Is that thought helping you? What would be a more balanced way to think about {topic}?"
            },
            "validation": {
                "supportive": "Your feelings about {topic} are completely understandable given what you're going through.",
                "encouraging": "You have every right to feel this way. Anyone in your situation would struggle.",
                "gentle": "It makes perfect sense that you'd feel this way about {topic}.",
                "direct": "Your reaction is normal and valid. This is a difficult situation.",
                "challenging": "Your feelings are valid, and you also have the strength to work through this."
            },
            "problem_solving": {
                "supportive": "Let's work together to find some solutions for {topic}. What have you tried so far?",
                "encouraging": "You've got this! Let's brainstorm some strategies for dealing with {topic}.",
                "gentle": "When you're ready, we could explore some options for handling {topic}.",
                "direct": "Let's create an action plan. What's the first step you could take?",
                "challenging": "What would happen if you tried a completely different approach to {topic}?"
            },
            "mindfulness": {
                "supportive": "Let's take a moment to be present. Notice your breathing as we discuss {topic}.",
                "encouraging": "You're doing great at staying aware. Can you observe your feelings about {topic} without judgment?",
                "gentle": "Gently bring your attention to the present moment. What do you notice right now?",
                "direct": "Focus on your breath for a moment. Now, let's return to {topic} with fresh perspective.",
                "challenging": "Can you observe these difficult feelings without getting caught up in them?"
            }
        }
        
        # Get template
        strategy_templates = templates.get(strategy, templates["empathetic_listening"])
        message = strategy_templates.get(tone, strategy_templates["supportive"])
        
        # Replace topic placeholder
        topic = "what you're experiencing"
        if self.current_state and self.current_state.conversation_history:
            # Extract topic from last user message (simplified)
            for msg in reversed(self.current_state.conversation_history):
                if msg.get("role") == "user":
                    # Simple keyword extraction
                    keywords = ["anxiety", "depression", "stress", "work", "relationship", "sleep"]
                    for keyword in keywords:
                        if keyword in msg.get("content", "").lower():
                            topic = keyword
                            break
                    break
        
        message = message.replace("{topic}", topic)
        
        # Add resource recommendation if appropriate
        if resource != "professional_referral" and random.random() < 0.3:
            resource_additions = {
                "article": "\n\nI have an article that might help with this. Would you like me to share it?",
                "exercise": "\n\nThere's a helpful exercise we could try. Are you interested?",
                "video": "\n\nI know a video that explains this well. Should I send you the link?",
                "worksheet": "\n\nI have a worksheet that could help us work through this. Would that be useful?",
                "meditation": "\n\nWould you like to try a brief meditation exercise?",
            }
            message += resource_additions.get(resource, "")
        
        return message
    
    def _calculate_reward(self, metrics: Dict[str, float], action_dict: Dict[str, Any]) -> float:
        """
        Calculate reward based on metrics and action
        
        Args:
            metrics: Response metrics from user simulator
            action_dict: Action taken
        
        Returns:
            Reward value
        """
        # Load reward weights from config
        weights = model_config.reward_weights
        
        # Base reward components
        reward = 0.0
        
        # Engagement reward
        reward += weights["engagement"] * metrics.get("engagement", 0.5)
        
        # Mood improvement reward
        mood_change = metrics.get("mood_change", 0.0)
        reward += weights["mood_improvement"] * clip_value(mood_change, -1.0, 1.0)
        
        # Helpfulness reward
        reward += weights["conversation_quality"] * metrics.get("helpfulness", 0.5)
        
        # Resource utilization reward
        if action_dict["resource_type"] != "professional_referral":
            resource_acceptance = metrics.get("resource_acceptance", 0.0)
            reward += weights["resource_utilization"] * resource_acceptance
        
        # Length bonus (encourage longer productive conversations)
        if metrics.get("session_length", 0) > 3:
            reward += 0.1
        
        # Safety check
        if self.current_state and self.current_state.risk_level > 0.8:
            if action_dict["resource_type"] != "professional_referral":
                # Should have referred to professional
                reward += weights["safety_penalty"]
        
        # Normalize reward to reasonable range
        reward = clip_value(reward, -2.0, 2.0)
        
        return float(reward)
    
    def _calculate_risk_level(self, session) -> float:
        """Calculate risk level for safety monitoring"""
        risk = 0.0
        
        # Check emotional state
        emotional_state = session.emotional_state
        if emotional_state.get("depression", 0) > 0.8:
            risk += 0.3
        if emotional_state.get("anxiety", 0) > 0.8:
            risk += 0.2
        
        # Check mood trajectory
        if len(session.mood_trajectory) >= 2:
            mood_decline = session.mood_trajectory[-1] - session.mood_trajectory[-2]
            if mood_decline < -0.2:
                risk += 0.3
        
        # Check for crisis keywords
        if session.conversation_history:
            last_message = session.conversation_history[-1].get("content", "").lower()
            for keyword in ["hopeless", "worthless", "give up", "can't go on"]:
                if keyword in last_message:
                    risk += 0.5
                    break
        
        return clip_value(risk, 0.0, 1.0)
    
    def _log_episode_summary(self):
        """Log summary of completed episode"""
        if not self.episode_rewards:
            return
        
        avg_reward = np.mean(self.episode_rewards)
        total_reward = np.sum(self.episode_rewards)
        
        # Get session summary
        session_summary = self.user_simulator.get_session_summary(self.current_session_id)
        
        training_logger.info(
            f"Episode {self.current_episode} completed: "
            f"Steps={self.current_step}, "
            f"Total Reward={total_reward:.2f}, "
            f"Avg Reward={avg_reward:.2f}, "
            f"Mood Improvement={session_summary.get('mood_improvement', 0):.2f}, "
            f"Final Engagement={session_summary.get('final_engagement', 0):.2f}"
        )
        
        # Calculate success rate
        if self.current_episode > 0:
            success_rate = self.successful_episodes / self.current_episode
            crisis_rate = self.crisis_episodes / self.current_episode
            training_logger.info(
                f"Overall: Success Rate={success_rate:.2%}, "
                f"Crisis Rate={crisis_rate:.2%}"
            )
    
    def render(self):
        """Render the environment (text-based for now)"""
        if self.current_state and self.current_state.conversation_history:
            print("\n" + "="*50)
            print("CONVERSATION STATE")
            print("="*50)
            
            # Show last 3 messages
            recent_history = self.current_state.conversation_history[-3:]
            for msg in recent_history:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                print(f"{role}: {content[:100]}...")
            
            # Show current metrics
            session = self.user_simulator.active_sessions.get(self.current_session_id)
            if session:
                print("\nCURRENT METRICS:")
                print(f"Mood Score: {session.get_mood_score():.2f}")
                print(f"Engagement: {session.engagement_level:.2f}")
                print(f"Step: {self.current_step}/{self.max_episode_length}")
            print("="*50 + "\n")
    
    def close(self):
        """Clean up environment"""
        # Reset all sessions
        for session_id in list(self.user_simulator.active_sessions.keys()):
            self.user_simulator.reset_session(session_id)
        logger.info("Environment closed")


# Test function
def test_environment():
    """Test the environment"""
    # Create environment
    env = MentalWellnessEnv(personas=None, max_episode_length=5)
    
    # Test episode
    observation, info = env.reset()
    logger.info(f"Initial observation shape: {observation.shape}")
    logger.info(f"Initial info: {info['persona_id']}")
    
    done = False
    total_reward = 0
    
    while not done:
        # Random action for testing
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        logger.info(f"Step {info['step']}: Reward={reward:.2f}, Done={done}")
        env.render()
    
    logger.info(f"Episode completed. Total reward: {total_reward:.2f}")
    
    env.close()
    return env


if __name__ == "__main__":
    test_environment()