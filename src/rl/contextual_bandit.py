"""
Contextual Bandit implementation using Thompson Sampling
For optimizing resource recommendations and timing
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
from pathlib import Path

from src.utils.logger import logger
from src.utils.helpers import save_json, load_json
from config.settings import DATA_DIR, AGENT_CONFIG


@dataclass
class BanditArm:
    """Represents a bandit arm (resource type)"""
    name: str
    alpha: float = 1.0  # Beta distribution alpha (successes + 1)
    beta: float = 1.0   # Beta distribution beta (failures + 1)
    total_pulls: int = 0
    total_reward: float = 0.0
    
    def update(self, reward: float):
        """Update arm statistics with reward (0-1 scale)"""
        self.total_pulls += 1
        self.total_reward += reward
        
        # Update Beta distribution parameters
        # Treat reward as success probability
        if reward > 0.5:  # Success threshold
            self.alpha += reward
        else:
            self.beta += (1 - reward)
    
    def sample_value(self) -> float:
        """Sample from Beta distribution (Thompson Sampling)"""
        return np.random.beta(self.alpha, self.beta)
    
    def get_mean_reward(self) -> float:
        """Get mean reward for this arm"""
        if self.total_pulls == 0:
            return 0.5  # Optimistic initialization
        return self.total_reward / self.total_pulls
    
    def get_ucb(self, total_pulls: int, c: float = 2.0) -> float:
        """Calculate Upper Confidence Bound"""
        if self.total_pulls == 0:
            return float('inf')
        
        mean = self.get_mean_reward()
        exploration_term = c * np.sqrt(np.log(total_pulls) / self.total_pulls)
        return mean + exploration_term


class ContextualBandit:
    """
    Contextual Multi-Armed Bandit for resource recommendation
    Uses Thompson Sampling for exploration-exploitation balance
    """
    
    def __init__(self, context_dim: int = 256):
        """
        Initialize contextual bandit
        
        Args:
            context_dim: Dimension of context vector (state)
        """
        self.context_dim = context_dim
        
        # Initialize arms for each resource type
        self.arms = {
            resource: BanditArm(name=resource)
            for resource in AGENT_CONFIG["resource_types"]
        }
        
        # Context-specific models (simplified linear model)
        self.context_weights = {
            resource: np.random.randn(context_dim) * 0.01
            for resource in AGENT_CONFIG["resource_types"]
        }
        
        # Timing optimization (for check-in intervals)
        self.timing_arms = {
            "immediate": BanditArm(name="immediate"),  # 0-2 hours
            "soon": BanditArm(name="soon"),            # 2-8 hours
            "daily": BanditArm(name="daily"),          # 8-24 hours
            "weekly": BanditArm(name="weekly")         # 24-168 hours
        }
        
        # Statistics
        self.total_selections = 0
        self.context_history = []
        self.reward_history = []
        
        logger.info(f"Initialized ContextualBandit with {len(self.arms)} resource arms")
    
    def select_resource(
        self,
        context: np.ndarray,
        exploration_mode: str = "thompson"
    ) -> Tuple[str, float]:
        """
        Select a resource based on context
        
        Args:
            context: Context vector (state)
            exploration_mode: "thompson", "ucb", or "greedy"
        
        Returns:
            Tuple of (selected_resource, confidence_score)
        """
        self.total_selections += 1
        
        if exploration_mode == "thompson":
            # Thompson Sampling
            scores = {}
            for resource, arm in self.arms.items():
                # Sample from posterior
                base_score = arm.sample_value()
                
                # Add context-based adjustment
                context_score = self._compute_context_score(context, resource)
                scores[resource] = base_score + 0.3 * context_score
            
            selected = max(scores, key=scores.get)
            confidence = scores[selected]
            
        elif exploration_mode == "ucb":
            # Upper Confidence Bound
            scores = {}
            for resource, arm in self.arms.items():
                ucb_score = arm.get_ucb(self.total_selections)
                context_score = self._compute_context_score(context, resource)
                scores[resource] = ucb_score + 0.3 * context_score
            
            selected = max(scores, key=scores.get)
            confidence = min(scores[selected], 1.0)
            
        else:  # greedy
            # Exploit best known arm
            scores = {}
            for resource, arm in self.arms.items():
                mean_reward = arm.get_mean_reward()
                context_score = self._compute_context_score(context, resource)
                scores[resource] = mean_reward + 0.3 * context_score
            
            selected = max(scores, key=scores.get)
            confidence = scores[selected]
        
        logger.debug(f"Selected resource: {selected} (confidence: {confidence:.3f})")
        return selected, confidence
    
    def _compute_context_score(self, context: np.ndarray, resource: str) -> float:
        """
        Compute context-based score for a resource
        
        Args:
            context: Context vector
            resource: Resource type
        
        Returns:
            Context score (-1 to 1)
        """
        weights = self.context_weights[resource]
        score = np.dot(context, weights)
        # Normalize with tanh
        return np.tanh(score)
    
    def update_resource(
        self,
        resource: str,
        context: np.ndarray,
        reward: float
    ):
        """
        Update bandit with feedback
        
        Args:
            resource: Selected resource
            context: Context when selected
            reward: Reward received (0-1 scale)
        """
        # Update arm statistics
        self.arms[resource].update(reward)
        
        # Update context weights (simple gradient update)
        learning_rate = 0.01
        prediction = self._compute_context_score(context, resource)
        error = reward - (prediction + 1) / 2  # Normalize prediction to 0-1
        
        # Gradient update
        self.context_weights[resource] += learning_rate * error * context
        
        # Store history
        self.context_history.append(context)
        self.reward_history.append(reward)
        
        logger.debug(f"Updated {resource}: reward={reward:.3f}, "
                    f"total_pulls={self.arms[resource].total_pulls}")
    
    def select_timing(
        self,
        context: np.ndarray,
        user_preference: Optional[str] = None
    ) -> Tuple[int, str]:
        """
        Select check-in timing
        
        Args:
            context: Current context
            user_preference: User's stated preference
        
        Returns:
            Tuple of (hours_until_checkin, timing_category)
        """
        if user_preference == "daily":
            # Respect user preference but optimize within range
            hours = np.random.randint(20, 28)
            category = "daily"
        elif user_preference == "weekly":
            hours = np.random.randint(144, 192)
            category = "weekly"
        else:
            # Use Thompson Sampling for timing
            scores = {}
            for timing, arm in self.timing_arms.items():
                scores[timing] = arm.sample_value()
            
            category = max(scores, key=scores.get)
            
            # Convert category to hours
            timing_ranges = {
                "immediate": (1, 2),
                "soon": (2, 8),
                "daily": (20, 28),
                "weekly": (144, 192)
            }
            
            min_hours, max_hours = timing_ranges[category]
            hours = np.random.randint(min_hours, max_hours)
        
        logger.debug(f"Selected timing: {hours} hours ({category})")
        return hours, category
    
    def update_timing(self, category: str, engagement_maintained: bool):
        """
        Update timing arm with feedback
        
        Args:
            category: Timing category used
            engagement_maintained: Whether user remained engaged
        """
        reward = 1.0 if engagement_maintained else 0.0
        self.timing_arms[category].update(reward)
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics"""
        stats = {
            "total_selections": self.total_selections,
            "resource_stats": {},
            "timing_stats": {},
            "best_resources": [],
            "exploration_rate": 0.0
        }
        
        # Resource statistics
        for resource, arm in self.arms.items():
            stats["resource_stats"][resource] = {
                "pulls": arm.total_pulls,
                "mean_reward": arm.get_mean_reward(),
                "alpha": arm.alpha,
                "beta": arm.beta
            }
        
        # Timing statistics
        for timing, arm in self.timing_arms.items():
            stats["timing_stats"][timing] = {
                "pulls": arm.total_pulls,
                "mean_reward": arm.get_mean_reward()
            }
        
        # Best performing resources
        sorted_resources = sorted(
            self.arms.items(),
            key=lambda x: x[1].get_mean_reward(),
            reverse=True
        )
        stats["best_resources"] = [
            (name, arm.get_mean_reward())
            for name, arm in sorted_resources[:3]
        ]
        
        # Calculate exploration rate
        if self.total_selections > 0:
            max_pulls = max(arm.total_pulls for arm in self.arms.values())
            total_pulls = sum(arm.total_pulls for arm in self.arms.values())
            stats["exploration_rate"] = 1.0 - (max_pulls / total_pulls)
        
        return stats
    
    def save(self, filepath: Optional[Path] = None):
        """Save bandit state"""
        if filepath is None:
            filepath = DATA_DIR / "models" / "contextual_bandit.json"
        
        state = {
            "arms": {
                name: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "total_pulls": arm.total_pulls,
                    "total_reward": arm.total_reward
                }
                for name, arm in self.arms.items()
            },
            "timing_arms": {
                name: {
                    "alpha": arm.alpha,
                    "beta": arm.beta,
                    "total_pulls": arm.total_pulls,
                    "total_reward": arm.total_reward
                }
                for name, arm in self.timing_arms.items()
            },
            "context_weights": {
                name: weights.tolist()
                for name, weights in self.context_weights.items()
            },
            "total_selections": self.total_selections
        }
        
        save_json(state, filepath)
        logger.info(f"Saved contextual bandit to {filepath}")
    
    def load(self, filepath: Optional[Path] = None):
        """Load bandit state"""
        if filepath is None:
            filepath = DATA_DIR / "models" / "contextual_bandit.json"
        
        if not filepath.exists():
            logger.warning(f"No saved bandit found at {filepath}")
            return
        
        state = load_json(filepath)
        
        # Restore arms
        for name, arm_data in state["arms"].items():
            if name in self.arms:
                self.arms[name].alpha = arm_data["alpha"]
                self.arms[name].beta = arm_data["beta"]
                self.arms[name].total_pulls = arm_data["total_pulls"]
                self.arms[name].total_reward = arm_data["total_reward"]
        
        # Restore timing arms
        for name, arm_data in state["timing_arms"].items():
            if name in self.timing_arms:
                self.timing_arms[name].alpha = arm_data["alpha"]
                self.timing_arms[name].beta = arm_data["beta"]
                self.timing_arms[name].total_pulls = arm_data["total_pulls"]
                self.timing_arms[name].total_reward = arm_data["total_reward"]
        
        # Restore context weights
        for name, weights in state["context_weights"].items():
            if name in self.context_weights:
                self.context_weights[name] = np.array(weights)
        
        self.total_selections = state["total_selections"]
        
        logger.info(f"Loaded contextual bandit from {filepath}")


# Test function
def test_contextual_bandit():
    """Test contextual bandit"""
    bandit = ContextualBandit()
    
    # Simulate some selections and updates
    for i in range(20):
        # Random context
        context = np.random.randn(256) * 0.1
        
        # Select resource
        resource, confidence = bandit.select_resource(context)
        
        # Simulate reward (random for testing)
        reward = np.random.random()
        
        # Update bandit
        bandit.update_resource(resource, context, reward)
        
        # Select timing
        hours, timing_cat = bandit.select_timing(context)
        
        # Update timing (random engagement for testing)
        engagement = np.random.random() > 0.3
        bandit.update_timing(timing_cat, engagement)
        
        if (i + 1) % 5 == 0:
            logger.info(f"Iteration {i+1}: Selected {resource} (confidence: {confidence:.3f})")
    
    # Get statistics
    stats = bandit.get_statistics()
    logger.info(f"Bandit statistics:")
    logger.info(f"  Total selections: {stats['total_selections']}")
    logger.info(f"  Best resources: {stats['best_resources']}")
    logger.info(f"  Exploration rate: {stats['exploration_rate']:.2%}")
    
    # Save and load test
    bandit.save()
    bandit2 = ContextualBandit()
    bandit2.load()
    logger.info(f"Successfully saved and loaded bandit state")
    
    return bandit


if __name__ == "__main__":
    test_contextual_bandit()