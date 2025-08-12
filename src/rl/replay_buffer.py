"""
Experience Replay Buffer for PPO training
Stores trajectories and samples batches for training
"""
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

from src.utils.logger import logger


@dataclass
class Experience:
    """Single experience tuple"""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


class ReplayBuffer:
    """
    Replay buffer for PPO training
    """
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
        
        logger.info(f"Initialized ReplayBuffer with capacity {capacity}")
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        log_prob: float,
        value: float = 0.0
    ):
        """
        Add experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode done flag
            log_prob: Log probability of action
            value: Value estimate
        """
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=log_prob,
            value=value
        )
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def compute_returns_and_advantages(
        self,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """
        Compute returns and advantages using GAE
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        if len(self.buffer) == 0:
            return
        
        # Convert to numpy for efficiency
        rewards = np.array([exp.reward for exp in self.buffer])
        values = np.array([exp.value for exp in self.buffer])
        dones = np.array([exp.done for exp in self.buffer])
        
        # Calculate returns and advantages
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # Work backwards through the buffer
        last_advantage = 0
        last_return = 0
        
        for t in reversed(range(len(self.buffer))):
            if t == len(self.buffer) - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1]
            
            # TD error
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            
            # GAE
            advantages[t] = last_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
            
            # Returns
            returns[t] = last_return = rewards[t] + gamma * (1 - dones[t]) * last_return
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update buffer with computed values
        for i, exp in enumerate(self.buffer):
            exp.advantage = advantages[i]
            exp.return_ = returns[i]
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of experiences
        
        Args:
            batch_size: Batch size
        
        Returns:
            Dictionary of batched tensors
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Random sampling
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        # Collect batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        log_probs = []
        advantages = []
        returns = []
        
        for idx in indices:
            exp = self.buffer[idx]
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            next_states.append(exp.next_state)
            dones.append(exp.done)
            log_probs.append(exp.log_prob)
            advantages.append(exp.advantage)
            returns.append(exp.return_)
        
        # Convert to tensors
        batch = {
            "states": torch.FloatTensor(np.array(states)),
            "actions": torch.LongTensor(np.array(actions)),
            "rewards": torch.FloatTensor(rewards),
            "next_states": torch.FloatTensor(np.array(next_states)),
            "dones": torch.FloatTensor(dones),
            "log_probs": torch.FloatTensor(log_probs),
            "advantages": torch.FloatTensor(advantages),
            "returns": torch.FloatTensor(returns)
        }
        
        return batch
    
    def get_all_batches(self, batch_size: int) -> List[Dict[str, torch.Tensor]]:
        """
        Get all experiences in batches
        
        Args:
            batch_size: Batch size
        
        Returns:
            List of batches
        """
        batches = []
        indices = np.arange(len(self.buffer))
        np.random.shuffle(indices)
        
        for start_idx in range(0, len(self.buffer), batch_size):
            end_idx = min(start_idx + batch_size, len(self.buffer))
            batch_indices = indices[start_idx:end_idx]
            
            # Collect batch
            states = []
            actions = []
            log_probs = []
            advantages = []
            returns = []
            
            for idx in batch_indices:
                exp = self.buffer[idx]
                states.append(exp.state)
                actions.append(exp.action)
                log_probs.append(exp.log_prob)
                advantages.append(exp.advantage)
                returns.append(exp.return_)
            
            batch = {
                "states": torch.FloatTensor(np.array(states)),
                "actions": torch.LongTensor(np.array(actions)),
                "old_log_probs": torch.FloatTensor(log_probs),
                "advantages": torch.FloatTensor(advantages),
                "returns": torch.FloatTensor(returns)
            }
            
            batches.append(batch)
        
        return batches
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        """Get buffer size"""
        return len(self.buffer)