"""
Conversation Agent using PPO policy
Handles main conversation strategies and responses
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.rl.policy_network import PolicyNetwork
from src.utils.state_manager import StateEncoder
from src.agents.response_generator import ResponseGenerator
from src.utils.logger import logger
from config.settings import DATA_DIR, AGENT_CONFIG


class ConversationAgent(BaseAgent):
    """
    Agent responsible for conversation strategy selection using PPO
    """
    
    def __init__(self, name: str = "ConversationAgent", model_path: Optional[Path] = None):
        """
        Initialize conversation agent
        
        Args:
            name: Agent name
            model_path: Path to trained PPO model
        """
        super().__init__(name, "Conversation")
        
        # Initialize policy network
        self.policy_net = PolicyNetwork()
        self.state_encoder = StateEncoder()
        
        # Initialize response generator for LLM-enhanced responses
        self.response_generator = ResponseGenerator()
        
        # Load strategy templates as fallback
        self.strategy_templates = self._load_strategy_templates()
        
        # Control flags
        self.use_llm_enhancement = True  # Toggle for LLM vs template responses
        
        # Model loading with better error handling
        model_loaded = False
        
        if model_path and model_path.exists():
            try:
                self.load_model(model_path)
                model_loaded = True
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_path}: {e}")
        
        if not model_loaded:
            # Try to load best model
            best_model_path = DATA_DIR / "models" / "best_model.pt"
            if best_model_path.exists():
                try:
                    # Fix the PyTorch 2.x issue
                    checkpoint = torch.load(best_model_path, 
                                        map_location='cpu',
                                        weights_only=False)  # Allow pickle
                    if "policy_state_dict" in checkpoint:
                        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
                        model_loaded = True
                        logger.info("Loaded best model")
                except Exception as e:
                    logger.warning(f"Could not load best model: {e}")
        
        if not model_loaded:
            logger.warning("Using random initialization - training needed for optimal performance!")
        
        self.policy_net.eval()  # Set to evaluation mode
        
        logger.info(f"Initialized ConversationAgent with PPO policy (LLM enhancement: {self.use_llm_enhancement})")
    
    def _load_strategy_templates(self) -> Dict[str, Dict[str, str]]:
        """Load conversation strategy templates"""
        return {
            "empathetic_listening": {
                "opener": "I hear how difficult this is for you.",
                "follow_up": "Can you tell me more about",
                "validation": "Your feelings are completely valid.",
                "support": "I'm here to listen and understand."
            },
            "cognitive_behavioral": {
                "opener": "Let's examine these thoughts together.",
                "follow_up": "What evidence supports or contradicts",
                "challenge": "Is there another way to look at this?",
                "reframe": "How might you reframe this thought?"
            },
            "validation": {
                "opener": "It's completely understandable to feel this way.",
                "follow_up": "Anyone in your situation would",
                "support": "You're not alone in this.",
                "acknowledgment": "Your experience is valid and important."
            },
            "problem_solving": {
                "opener": "Let's work on finding some solutions.",
                "follow_up": "What options do you see",
                "action": "What's one small step you could take?",
                "planning": "How can we break this down into manageable parts?"
            },
            "mindfulness": {
                "opener": "Let's take a moment to be present.",
                "follow_up": "Notice what you're feeling right now",
                "guide": "Focus on your breath for a moment.",
                "awareness": "What do you notice in this moment?"
            },
            "motivational": {
                "opener": "You have the strength to handle this.",
                "follow_up": "What has helped you before",
                "encourage": "I believe in your ability to",
                "strength": "You've overcome challenges before."
            },
            "psychoeducation": {
                "opener": "Let me explain what might be happening.",
                "follow_up": "This is a common response to",
                "educate": "Understanding this can help because",
                "information": "Here's what we know about this."
            },
            "supportive": {
                "opener": "I'm here to support you.",
                "follow_up": "How can I best help you right now",
                "comfort": "It's okay to feel this way.",
                "presence": "You don't have to go through this alone."
            }
        }
    
    def process(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process conversation state and select strategy
        
        Args:
            state: Current conversation state
            context: Additional context (persona, history, etc.)
        
        Returns:
            Response with selected strategy and action
        """
        try:
            # Encode state
            state_vector = state.get("state_vector")
            if state_vector is None:
                # Create state vector from components
                from src.utils.state_manager import ConversationState
                conv_state = ConversationState(
                    emotional_state=np.array(state.get("emotional_state", [0.5]*5)),
                    engagement_level=state.get("engagement_level", 0.7),
                    conversation_history=context.get("conversation_history", []),
                    session_duration=state.get("session_duration", 0),
                    time_since_last=state.get("time_since_last", 0),
                    risk_level=state.get("risk_level", 0.0)
                )
                state_vector = conv_state.to_vector()
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            
            # Add exploration for diversity
            exploration = context.get("exploration", 0.3)
            temperature = 1.0 + exploration  # Higher temperature = more random
            
            # Get action from policy with controlled randomness
            with torch.no_grad():
                # Use temperature-controlled sampling for variety
                strategy_probs, resource_probs, tone_probs = self.policy_net.get_action_probs(
                    state_tensor,
                    temperature=temperature
                )
                
                # Sample from distributions
                if np.random.random() < exploration:
                    # Exploration: sample from distribution
                    strategy_idx = torch.multinomial(strategy_probs, 1).item()
                    tone_idx = torch.multinomial(tone_probs, 1).item()
                else:
                    # Exploitation: choose most likely
                    strategy_idx = torch.argmax(strategy_probs).item()
                    tone_idx = torch.argmax(tone_probs).item()
                
                # For logging
                action = torch.tensor([[strategy_idx, 0, tone_idx]])
                log_prob = torch.log(strategy_probs[0, strategy_idx] * tone_probs[0, tone_idx])
            
            # Convert indices to strategy and tone
            strategy = AGENT_CONFIG["conversation_strategies"][strategy_idx]
            tone = AGENT_CONFIG["response_tones"][tone_idx]
            
            # Generate response using selected strategy and tone
            if self.use_llm_enhancement:
                # Use LLM-enhanced response generation
                response_text = self.response_generator.generate_response(
                    strategy=strategy,
                    tone=tone,
                    context=context,
                    use_llm=True  # Use OpenAI for better responses
                )
            else:
                # Use template-based response
                response_text = self._generate_response_template(
                    strategy, tone, context
                )
            
            response = {
                "strategy": strategy,
                "tone": tone,
                "response_template": response_text,
                "confidence": float(torch.exp(log_prob).item()),
                "action": [strategy_idx, 0, tone_idx],
                "agent": self.name,
                "llm_enhanced": self.use_llm_enhancement
            }
            
            # Log interaction
            self.log_interaction(success=True, rating=0.7)
            
            logger.debug(f"Generated response with strategy: {strategy}, tone: {tone}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in ConversationAgent.process: {e}")
            # Return fallback response
            fallback_response = "I'm here to listen and support you. Could you tell me more about what you're experiencing?"
            
            # Try to use LLM for fallback if available
            if self.use_llm_enhancement:
                try:
                    fallback_response = self.response_generator.generate_response(
                        strategy="supportive",
                        tone="gentle",
                        context=context,
                        use_llm=True
                    )
                except:
                    pass  # Keep original fallback
            
            return {
                "strategy": "supportive",
                "tone": "gentle",
                "response_template": fallback_response,
                "confidence": 0.5,
                "action": [7, 0, 2],  # Supportive strategy
                "agent": self.name,
                "llm_enhanced": False
            }
    
    def _generate_response_template(self, strategy: str, tone: str, context: Dict[str, Any]) -> str:
        """Generate response template based on strategy and tone (fallback method)"""
        templates = self.strategy_templates.get(strategy, self.strategy_templates["supportive"])
        
        # Get conversation history
        conversation_history = context.get("conversation_history", [])
        history_length = len(conversation_history)
        
        # Dynamic template selection based on conversation stage
        if history_length == 0:
            template_key = "opener"
        elif history_length < 3:
            # Early conversation - use follow-up or validation
            template_key = np.random.choice(["follow_up", "validation", "support"])
        else:
            # Later conversation - use any template for variety
            template_keys = list(templates.keys())
            template_key = np.random.choice(template_keys)
        
        base_template = templates.get(template_key, "I understand.")
        
        # Add tone-specific adjustments
        tone_adjustments = {
            "supportive": [
                " I'm here for you.",
                " You're not alone in this.",
                " I want to help you through this."
            ],
            "encouraging": [
                " You're doing great by talking about this.",
                " I can see your strength.",
                " Every step forward matters."
            ],
            "gentle": [
                " Take your time.",
                " There's no rush.",
                " It's okay to feel this way."
            ],
            "direct": [
                " Let's focus on this.",
                " What's most important right now?",
                " Let's be specific."
            ],
            "challenging": [
                " Consider this perspective.",
                " What if we looked at it differently?",
                " Let's explore this further."
            ]
        }
        
        # Add random tone adjustment for variety
        tone_additions = tone_adjustments.get(tone, [""])
        adjusted = base_template + " " + np.random.choice(tone_additions)
        
        # Add contextual elements for more natural flow
        if history_length > 0 and np.random.random() < 0.3:
            contextual_additions = [
                " How does that resonate with you?",
                " What are your thoughts on this?",
                " Does this feel helpful?",
                " Tell me more about that.",
                " I'd like to understand better."
            ]
            adjusted += np.random.choice(contextual_additions)
        
        return adjusted.strip()
    
    def set_llm_enhancement(self, use_llm: bool):
        """Toggle LLM enhancement on/off"""
        self.use_llm_enhancement = use_llm
        logger.info(f"LLM enhancement set to: {use_llm}")
    
    def update(self, feedback: Dict[str, Any]):
        """
        Update agent based on feedback
        
        Args:
            feedback: Feedback containing reward, success metrics
        """
        # In production, this would update the policy network
        # For now, just log the feedback
        reward = feedback.get("reward", 0.5)
        success = feedback.get("success", True)
        
        self.log_interaction(success=success, rating=reward)
        
        logger.debug(f"ConversationAgent received feedback: reward={reward:.2f}")
    
    def load_model(self, model_path: Path):
        """Load trained PPO model"""
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            if "policy_state_dict" in checkpoint:
                self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
            else:
                self.policy_net.load_state_dict(checkpoint)
            
            logger.info(f"Loaded PPO model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def get_strategy_distribution(self, state_vector: np.ndarray) -> Dict[str, float]:
        """Get probability distribution over strategies"""
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
        
        with torch.no_grad():
            strategy_probs, _, _ = self.policy_net.get_action_probs(state_tensor)
        
        probs = strategy_probs.cpu().numpy()[0]
        
        distribution = {
            strategy: float(probs[i])
            for i, strategy in enumerate(AGENT_CONFIG["conversation_strategies"])
        }
        
        return distribution