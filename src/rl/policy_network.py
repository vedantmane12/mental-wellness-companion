"""
Policy Network for PPO
Outputs action probabilities for conversation strategies
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

from config.model_config import model_config
from src.utils.logger import logger


class PolicyNetwork(nn.Module):
    """
    Policy network for selecting actions in mental wellness conversations
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = None,
        output_dim: int = 19,  # 8 strategies + 6 resources + 5 tones
        dropout_rate: float = 0.1
    ):
        """
        Initialize policy network
        
        Args:
            input_dim: State vector dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Total action dimensions
            dropout_rate: Dropout probability
        """
        super(PolicyNetwork, self).__init__()
        
        # Use config if no custom dims provided
        if hidden_dims is None:
            hidden_dims = model_config.policy_network["hidden_layers"]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Separate heads for different action types
        self.strategy_head = nn.Linear(prev_dim, 8)  # Conversation strategies
        self.resource_head = nn.Linear(prev_dim, 6)  # Resource types
        self.tone_head = nn.Linear(prev_dim, 5)      # Response tones
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized PolicyNetwork with architecture: {input_dim} -> {hidden_dims} -> {output_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Tuple of (strategy_logits, resource_logits, tone_logits)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get logits for each action type
        strategy_logits = self.strategy_head(features)
        resource_logits = self.resource_head(features)
        tone_logits = self.tone_head(features)
        
        return strategy_logits, resource_logits, tone_logits
    
    def get_action_probs(
        self,
        state: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action probabilities
        
        Args:
            state: State tensor
            temperature: Temperature for softmax (lower = more deterministic)
        
        Returns:
            Tuple of probability distributions
        """
        strategy_logits, resource_logits, tone_logits = self.forward(state)
        
        # Apply temperature scaling
        strategy_probs = F.softmax(strategy_logits / temperature, dim=-1)
        resource_probs = F.softmax(resource_logits / temperature, dim=-1)
        tone_probs = F.softmax(tone_logits / temperature, dim=-1)
        
        return strategy_probs, resource_probs, tone_probs
    
    def sample_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            state: State tensor
            deterministic: If True, select most likely action
        
        Returns:
            Tuple of (action, log_prob)
        """
        strategy_probs, resource_probs, tone_probs = self.get_action_probs(state)
        
        if deterministic:
            # Select most likely actions
            strategy_action = torch.argmax(strategy_probs, dim=-1)
            resource_action = torch.argmax(resource_probs, dim=-1)
            tone_action = torch.argmax(tone_probs, dim=-1)
        else:
            # Sample from distributions
            strategy_dist = torch.distributions.Categorical(strategy_probs)
            resource_dist = torch.distributions.Categorical(resource_probs)
            tone_dist = torch.distributions.Categorical(tone_probs)
            
            strategy_action = strategy_dist.sample()
            resource_action = resource_dist.sample()
            tone_action = tone_dist.sample()
        
        # Stack actions
        action = torch.stack([strategy_action, resource_action, tone_action], dim=-1)
        
        # Calculate log probability
        log_prob = (
            torch.log(strategy_probs.gather(-1, strategy_action.unsqueeze(-1)).squeeze(-1)) +
            torch.log(resource_probs.gather(-1, resource_action.unsqueeze(-1)).squeeze(-1)) +
            torch.log(tone_probs.gather(-1, tone_action.unsqueeze(-1)).squeeze(-1))
        )
        
        return action, log_prob
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy of given actions
        
        Args:
            states: Batch of states
            actions: Batch of actions taken
        
        Returns:
            Tuple of (log_probs, entropy)
        """
        strategy_probs, resource_probs, tone_probs = self.get_action_probs(states)
        
        # Split actions
        strategy_actions = actions[:, 0].long()
        resource_actions = actions[:, 1].long()
        tone_actions = actions[:, 2].long()
        
        # Calculate log probabilities
        strategy_log_probs = torch.log(strategy_probs.gather(1, strategy_actions.unsqueeze(1)).squeeze(1))
        resource_log_probs = torch.log(resource_probs.gather(1, resource_actions.unsqueeze(1)).squeeze(1))
        tone_log_probs = torch.log(tone_probs.gather(1, tone_actions.unsqueeze(1)).squeeze(1))
        
        log_probs = strategy_log_probs + resource_log_probs + tone_log_probs
        
        # Calculate entropy for exploration bonus
        strategy_entropy = -(strategy_probs * torch.log(strategy_probs + 1e-8)).sum(dim=-1)
        resource_entropy = -(resource_probs * torch.log(resource_probs + 1e-8)).sum(dim=-1)
        tone_entropy = -(tone_probs * torch.log(tone_probs + 1e-8)).sum(dim=-1)
        
        entropy = strategy_entropy + resource_entropy + tone_entropy
        
        return log_probs, entropy.mean()