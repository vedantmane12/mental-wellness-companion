"""
Orchestrator for coordinating multiple agents
Central control system for the mental wellness companion
"""
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.agents.monitoring_agent import MonitoringAgent
from src.utils.logger import logger
from src.utils.state_manager import ConversationState
from config.settings import SAFETY_CONFIG, AGENT_CONFIG


class Orchestrator:
    """
    Central orchestrator for coordinating all agents
    """
    
    def __init__(self):
        """Initialize orchestrator with all agents"""
        logger.info("Initializing Orchestrator...")
        
        # Initialize agents
        self.conversation_agent = ConversationAgent()
        self.recommendation_agent = RecommendationAgent()
        self.monitoring_agent = MonitoringAgent()
        
        # Session management
        self.active_sessions = {}
        self.session_states = {}
        self.recent_strategies = {}  # Track recent strategies per session
        
        # Orchestration statistics
        self.stats = {
            "total_interactions": 0,
            "successful_sessions": 0,
            "crisis_interventions": 0,
            "agent_utilization": {
                "conversation": 0,
                "recommendation": 0,
                "monitoring": 0
            }
        }
        
        logger.info("Orchestrator initialized with all agents")
    
    def process_interaction(
        self,
        session_id: str,
        user_message: str,
        persona: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a user interaction through all agents
        
        Args:
            session_id: Session identifier
            user_message: User's message
            persona: User persona (optional)
            context: Additional context (optional)
        
        Returns:
            Orchestrated response with all agent outputs
        """
        self.stats["total_interactions"] += 1
        
        # Get or create session state
        session_state = self._get_session_state(session_id, persona)
        
        # Update conversation history
        session_state["conversation_history"].append({
            "role": "user",
            "content": user_message
        })
        
        # Create current state representation
        current_state = self._create_state_representation(session_state)
        
        agent_context = {
            "conversation_history": session_state["conversation_history"],
            "training": False,
            "exploration": 0.3  # Increased for more variety
        }

        if context:
            agent_context.update(context)
                
        # Step 1: Monitor conversation health
        monitoring_result = self.monitoring_agent.process(
            current_state, 
            agent_context
        )
        self.stats["agent_utilization"]["monitoring"] += 1
        
        # Step 2: Check if intervention needed
        if monitoring_result["needs_intervention"]:
            response = self._handle_intervention(monitoring_result, session_state)
            self.stats["crisis_interventions"] += 1
        else:
            # Step 3: Get conversation strategy
            conversation_result = self.conversation_agent.process(
                current_state,
                agent_context
            )
            self.stats["agent_utilization"]["conversation"] += 1
            
            # Step 4: Get resource recommendation
            recommendation_result = self.recommendation_agent.process(
                current_state,
                agent_context
            )
            self.stats["agent_utilization"]["recommendation"] += 1
            
            # Step 5: Combine agent outputs with diversity
            response = self._combine_agent_outputs(
                conversation_result,
                recommendation_result,
                monitoring_result,
                session_id
            )
        
        # Update session state
        self._update_session_state(session_id, current_state, response)
        
        # Add response to history
        session_state["conversation_history"].append({
            "role": "assistant",
            "content": response["message"],
            "metadata": {
                "strategy": response.get("strategy"),
                "resource": response.get("resource_type"),
                "risk_level": monitoring_result["risk_level"]
            }
        })
        
        # Create final response
        final_response = {
            "session_id": session_id,
            "message": response["message"],
            "strategy": response.get("strategy"),
            "tone": response.get("tone"),
            "resource": response.get("resource"),
            "monitoring": {
                "risk_level": monitoring_result["risk_level"],
                "health_score": monitoring_result["health_score"],
                "recommendations": monitoring_result["recommendations"]
            },
            "continue_conversation": not monitoring_result["crisis_detected"],
            "metadata": {
                "turn": session_state["turn_count"],
                "agents_used": ["monitoring", "conversation", "recommendation"]
            }
        }
        
        logger.info(f"Orchestrated interaction for session {session_id}: "
                   f"risk={monitoring_result['risk_level']:.2f}, "
                   f"strategy={response.get('strategy')}")
        
        return final_response
    
    def _get_session_state(
        self,
        session_id: str,
        persona: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get or create session state"""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "session_id": session_id,
                "persona": persona or {},
                "conversation_history": [],
                "emotional_state": {
                    "anxiety": 0.5,
                    "depression": 0.5,
                    "stress": 0.5,
                    "anger": 0.2,
                    "happiness": 0.3
                },
                "engagement_level": 0.7,
                "turn_count": 0,
                "resources_recommended": [],
                "strategies_used": []
            }
        
        self.session_states[session_id]["turn_count"] += 1
        return self.session_states[session_id]
    
    def _create_state_representation(self, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create state representation for agents"""
        # Create ConversationState object
        conv_state = ConversationState(
            emotional_state=np.array(list(session_state["emotional_state"].values())),
            engagement_level=session_state["engagement_level"],
            conversation_history=session_state["conversation_history"],
            session_duration=session_state["turn_count"] * 2.0,  # Estimate minutes
            time_since_last=0.0,
            risk_level=0.0  # Will be updated by monitoring
        )
        
        return {
            "state_vector": conv_state.to_vector(),
            "emotional_state": session_state["emotional_state"],
            "engagement_level": session_state["engagement_level"],
            "risk_level": 0.0  # Updated by monitoring
        }
    
    def _handle_intervention(
        self,
        monitoring_result: Dict[str, Any],
        session_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle crisis intervention"""
        logger.warning(f"Handling intervention for session {session_state['session_id']}")
        
        # Use crisis response
        message = SAFETY_CONFIG["crisis_response"]
        
        return {
            "message": message,
            "strategy": "crisis_intervention",
            "tone": "supportive",
            "resource": {
                "type": "professional_referral",
                "urgent": True,
                "resources": [
                    "National Suicide Prevention Lifeline: 988",
                    "Crisis Text Line: Text HOME to 741741"
                ]
            }
        }
    
    def _combine_agent_outputs(
        self,
        conversation_result: Dict[str, Any],
        recommendation_result: Dict[str, Any],
        monitoring_result: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """Combine outputs from all agents with diversity management"""
        
        # Initialize recent strategies tracking for this session
        if session_id not in self.recent_strategies:
            self.recent_strategies[session_id] = []
        
        recent = self.recent_strategies[session_id]
        current_strategy = conversation_result["strategy"]
        
        # Check if strategy was used recently and force diversity
        if len(recent) >= 2 and current_strategy in recent[-2:]:
            # Get alternative strategies
            all_strategies = AGENT_CONFIG["conversation_strategies"]
            available = [s for s in all_strategies if s not in recent[-2:]]
            
            if available:
                # Choose a different strategy
                current_strategy = np.random.choice(available)
                conversation_result["strategy"] = current_strategy
                logger.debug(f"Forcing strategy diversity: switched to {current_strategy}")
        
        # Track this strategy
        recent.append(current_strategy)
        if len(recent) > 5:
            recent.pop(0)
        
        # Start with conversation agent's response
        message = conversation_result["response_template"]
        
        # Add variety to the message
        if np.random.random() < 0.3:  # 30% chance to add variation
            variations = [
                " How does that resonate with you?",
                " What are your thoughts on this?",
                " Does this feel helpful?",
                " I'm here to support you through this.",
                " Take your time to process this."
            ]
            message += np.random.choice(variations)
        
        # Add resource recommendation if appropriate
        if recommendation_result["confidence"] > 0.6:
            resource_addon = f"\n\n{recommendation_result['reason']}"
            message += resource_addon
        
        # Adjust based on monitoring recommendations
        if "Use validation and empathy" in monitoring_result["recommendations"]:
            message = self._add_empathy(message)
        
        return {
            "message": message,
            "strategy": current_strategy,
            "tone": conversation_result["tone"],
            "resource": {
                "type": recommendation_result["resource_type"],
                "metadata": recommendation_result["resource_metadata"],
                "timing": recommendation_result["timing_hours"]
            }
        }
    
    def _add_empathy(self, message: str) -> str:
        """Add empathetic elements to message"""
        empathy_additions = [
            "I really hear you. ",
            "That sounds really difficult. ",
            "Your feelings are valid. ",
            "I understand this is challenging. ",
            "Thank you for sharing this with me. "
        ]
        return np.random.choice(empathy_additions) + message
    
    def _update_session_state(
        self,
        session_id: str,
        current_state: Dict[str, Any],
        response: Dict[str, Any]
    ):
        """Update session state after interaction"""
        if session_id in self.session_states:
            session = self.session_states[session_id]
            
            # Track strategies used
            if "strategy" in response:
                session["strategies_used"].append(response["strategy"])
            
            # Track resources recommended
            if "resource" in response:
                session["resources_recommended"].append(response["resource"]["type"])
            
            # Update emotional state (simplified - would use user feedback in production)
            # Slight improvement from interaction
            for emotion in session["emotional_state"]:
                if emotion == "happiness":
                    session["emotional_state"][emotion] = min(1.0, session["emotional_state"][emotion] + 0.02)
                else:
                    session["emotional_state"][emotion] = max(0.0, session["emotional_state"][emotion] - 0.01)
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session and get summary"""
        if session_id not in self.session_states:
            return {"error": "Session not found"}
        
        session = self.session_states[session_id]
        
        # Calculate session metrics
        summary = {
            "session_id": session_id,
            "total_turns": session["turn_count"],
            "strategies_used": list(set(session["strategies_used"])),
            "resources_recommended": list(set(session["resources_recommended"])),
            "final_emotional_state": session["emotional_state"],
            "engagement_maintained": session["engagement_level"] > 0.5
        }
        
        # Update success metrics
        if session["turn_count"] > 3 and session["engagement_level"] > 0.5:
            self.stats["successful_sessions"] += 1
        
        # Clean up session
        del self.session_states[session_id]
        
        # Clean up recent strategies
        if session_id in self.recent_strategies:
            del self.recent_strategies[session_id]
        
        logger.info(f"Ended session {session_id}: {session['turn_count']} turns")
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return {
            "orchestrator_stats": self.stats,
            "conversation_agent": self.conversation_agent.get_metrics(),
            "recommendation_agent": self.recommendation_agent.get_resource_statistics(),
            "monitoring_agent": self.monitoring_agent.get_metrics(),
            "active_sessions": len(self.session_states)
        }