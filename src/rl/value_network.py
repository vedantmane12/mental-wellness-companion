"""
Value Network for PPO
Estimates state value for advantage calculation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.model_config import model_config
from src.utils.logger import logger


class ValueNetwork(nn.Module):
    """
    Value network for estimating state values
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dims: list = None,
        dropout_rate: float = 0.1
    ):
        """
        Initialize value network
        
        Args:
            input_dim: State vector dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout probability
        """
        super(ValueNetwork, self).__init__()
        
        # Use config if no custom dims provided
        if hidden_dims is None:
            hidden_dims = model_config.value_network["hidden_layers"]
        
        self.input_dim = input_dim
        
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
        
        # Add final layer for value output
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized ValueNetwork with architecture: {input_dim} -> {hidden_dims} -> 1")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            State value estimate [batch_size, 1]
        """
        return self.network(state)
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get state value (convenience method)
        
        Args:
            state: State tensor
        
        Returns:
            State value
        """
        return self.forward(state).squeeze(-1)