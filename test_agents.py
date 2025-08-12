"""Test script for agents"""
import numpy as np
from src.agents.conversation_agent import ConversationAgent
from src.agents.recommendation_agent import RecommendationAgent

print("Testing Conversation Agent...")
conv_agent = ConversationAgent()
state = {
    "emotional_state": [0.7, 0.5, 0.6, 0.2, 0.3],
    "engagement_level": 0.75,
    "risk_level": 0.3
}
context = {"conversation_history": [], "training": False}
response = conv_agent.process(state, context)
print(f"Strategy: {response['strategy']}")
print(f"Tone: {response['tone']}")
print(f"Template: {response['response_template']}")

print("\nTesting Recommendation Agent...")
rec_agent = RecommendationAgent()
recommendation = rec_agent.process(state, context)
print(f"Resource: {recommendation['resource_type']}")
print(f"Timing: {recommendation['timing_hours']} hours ({recommendation['timing_category']})")
print(f"Reason: {recommendation['reason']}")

print("\nAll agents tested successfully!")