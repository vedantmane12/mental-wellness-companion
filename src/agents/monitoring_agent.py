"""
Monitoring Agent for tracking conversation health and safety
"""
import numpy as np
from typing import Dict, Any, List, Optional
from src.agents.base_agent import BaseAgent
from src.utils.logger import logger
from config.settings import SAFETY_CONFIG


class MonitoringAgent(BaseAgent):
    """
    Agent responsible for monitoring conversation health, safety, and progress
    """
    
    def __init__(self, name: str = "MonitoringAgent"):
        """Initialize monitoring agent"""
        super().__init__(name, "Monitoring")
        
        # Risk thresholds
        self.risk_thresholds = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "crisis": 0.9
        }
        
        # Conversation health metrics
        self.health_metrics = {
            "engagement_trend": [],
            "mood_trend": [],
            "risk_history": [],
            "intervention_count": 0
        }
        
        logger.info("Initialized MonitoringAgent")
    
    def process(self, state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor conversation and assess health/safety
        
        Args:
            state: Current conversation state
            context: Conversation context
        
        Returns:
            Monitoring assessment with recommendations
        """
        # Extract key metrics
        emotional_state = state.get("emotional_state", {})
        engagement_level = state.get("engagement_level", 0.5)
        conversation_history = context.get("conversation_history", [])
        
        # Calculate risk level
        risk_level = self._calculate_risk_level(emotional_state, conversation_history)
        
        # Check for crisis keywords
        crisis_detected = self._check_crisis_keywords(conversation_history)
        
        # Calculate conversation health
        health_score = self._calculate_health_score(
            engagement_level, 
            emotional_state,
            len(conversation_history)
        )
        
        # Determine intervention need
        needs_intervention = risk_level > self.risk_thresholds["high"] or crisis_detected
        
        # Update metrics
        self._update_metrics(engagement_level, emotional_state, risk_level)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_level,
            crisis_detected,
            health_score
        )
        
        assessment = {
            "risk_level": risk_level,
            "risk_category": self._get_risk_category(risk_level),
            "crisis_detected": crisis_detected,
            "health_score": health_score,
            "needs_intervention": needs_intervention,
            "recommendations": recommendations,
            "metrics_summary": self._get_metrics_summary(),
            "agent": self.name
        }
        
        # Log if intervention needed
        if needs_intervention:
            logger.warning(f"Intervention needed: risk_level={risk_level:.2f}, crisis={crisis_detected}")
            self.health_metrics["intervention_count"] += 1
        
        return assessment
    
    def _calculate_risk_level(
        self,
        emotional_state: Dict[str, float],
        conversation_history: List[Dict[str, str]]
    ) -> float:
        """Calculate overall risk level"""
        risk = 0.0
        
        # Emotional state risks
        if isinstance(emotional_state, dict):
            depression = emotional_state.get("depression", 0.5)
            anxiety = emotional_state.get("anxiety", 0.5)
            anger = emotional_state.get("anger", 0.3)
            
            # High depression is highest risk
            if depression > 0.8:
                risk = max(risk, 0.9)
            elif depression > 0.6:
                risk = max(risk, 0.6)
            
            # High anxiety
            if anxiety > 0.8:
                risk = max(risk, 0.7)
            
            # High anger could indicate crisis
            if anger > 0.7:
                risk = max(risk, 0.6)
        
        # Check conversation patterns
        if len(conversation_history) > 0:
            # Short responses might indicate disengagement
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
            avg_length = np.mean([
                len(msg.get("content", "")) 
                for msg in recent_messages 
                if msg.get("role") == "user"
            ])
            
            if avg_length < 20:  # Very short responses
                risk = max(risk, risk + 0.2)
        
        # Check for concerning patterns
        if self._check_concerning_patterns(conversation_history):
            risk = max(risk, 0.7)
        
        return min(risk, 1.0)
    
    def _check_crisis_keywords(self, conversation_history: List[Dict[str, str]]) -> bool:
        """Check for crisis keywords in conversation"""
        if not conversation_history:
            return False
        
        # Check recent user messages
        for msg in reversed(conversation_history[-5:]):  # Last 5 messages
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                for keyword in SAFETY_CONFIG["crisis_keywords"]:
                    if keyword in content:
                        logger.warning(f"Crisis keyword detected: '{keyword}'")
                        return True
        
        return False
    
    def _check_concerning_patterns(self, conversation_history: List[Dict[str, str]]) -> bool:
        """Check for concerning conversation patterns"""
        concerning_phrases = [
            "no point", "give up", "can't go on", "hopeless",
            "worthless", "burden", "better off without me"
        ]
        
        for msg in conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "").lower()
                for phrase in concerning_phrases:
                    if phrase in content:
                        return True
        
        return False
    
    def _calculate_health_score(
        self,
        engagement: float,
        emotional_state: Dict[str, float],
        conversation_length: int
    ) -> float:
        """Calculate overall conversation health score"""
        health = 0.0
        
        # Engagement contributes positively
        health += engagement * 0.4
        
        # Positive emotions contribute
        if isinstance(emotional_state, dict):
            happiness = emotional_state.get("happiness", 0.3)
            health += happiness * 0.3
            
            # Low negative emotions contribute
            depression = emotional_state.get("depression", 0.5)
            anxiety = emotional_state.get("anxiety", 0.5)
            health += (1 - depression) * 0.15
            health += (1 - anxiety) * 0.15
        
        # Reasonable conversation length is good
        if 3 <= conversation_length <= 15:
            health += 0.1
        
        return min(health, 1.0)
    
    def _get_risk_category(self, risk_level: float) -> str:
        """Get risk category from level"""
        if risk_level >= self.risk_thresholds["crisis"]:
            return "crisis"
        elif risk_level >= self.risk_thresholds["high"]:
            return "high"
        elif risk_level >= self.risk_thresholds["medium"]:
            return "medium"
        elif risk_level >= self.risk_thresholds["low"]:
            return "low"
        else:
            return "minimal"
    
    def _generate_recommendations(
        self,
        risk_level: float,
        crisis_detected: bool,
        health_score: float
    ) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        if crisis_detected:
            recommendations.append("IMMEDIATE: Provide crisis resources and professional referral")
            recommendations.append("Use supportive, non-judgmental language")
            recommendations.append("Avoid giving advice, focus on listening")
        elif risk_level > self.risk_thresholds["high"]:
            recommendations.append("Consider professional referral")
            recommendations.append("Increase check-in frequency")
            recommendations.append("Focus on safety and support")
        elif risk_level > self.risk_thresholds["medium"]:
            recommendations.append("Monitor closely for escalation")
            recommendations.append("Provide coping resources")
            recommendations.append("Use validation and empathy")
        
        if health_score < 0.3:
            recommendations.append("Work on improving engagement")
            recommendations.append("Try different conversation strategies")
        
        if not recommendations:
            recommendations.append("Continue current approach")
        
        return recommendations
    
    def _update_metrics(
        self,
        engagement: float,
        emotional_state: Dict[str, float],
        risk_level: float
    ):
        """Update tracking metrics"""
        self.health_metrics["engagement_trend"].append(engagement)
        
        # Calculate mood score
        if isinstance(emotional_state, dict):
            mood = emotional_state.get("happiness", 0.5) - (
                emotional_state.get("depression", 0.5) * 0.5 +
                emotional_state.get("anxiety", 0.5) * 0.3
            )
            self.health_metrics["mood_trend"].append(mood)
        
        self.health_metrics["risk_history"].append(risk_level)
        
        # Keep only recent history
        max_history = 20
        for key in ["engagement_trend", "mood_trend", "risk_history"]:
            if len(self.health_metrics[key]) > max_history:
                self.health_metrics[key] = self.health_metrics[key][-max_history:]
    
    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring metrics"""
        summary = {
            "avg_engagement": np.mean(self.health_metrics["engagement_trend"]) if self.health_metrics["engagement_trend"] else 0.5,
            "mood_direction": self._calculate_trend(self.health_metrics["mood_trend"]),
            "risk_trend": self._calculate_trend(self.health_metrics["risk_history"]),
            "interventions_triggered": self.health_metrics["intervention_count"]
        }
        return summary
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier = np.mean(values[:-3]) if len(values) > 3 else values[0]
        
        diff = recent - earlier
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"
    
    def update(self, feedback: Dict[str, Any]):
        """Update based on feedback"""
        # Log feedback
        success = feedback.get("intervention_successful", True)
        self.log_interaction(success=success)