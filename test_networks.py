"""Test script for neural networks"""
import torch
import numpy as np
from src.rl.policy_network import PolicyNetwork
from src.rl.value_network import ValueNetwork
from src.rl.replay_buffer import ReplayBuffer

# Test Policy Network
print("Testing Policy Network...")
policy_net = PolicyNetwork()
state = torch.randn(1, 256)  # Batch size 1, state dim 256
action, log_prob = policy_net.sample_action(state)
print(f"Action shape: {action.shape}, Log prob: {log_prob.item():.4f}")

# Test Value Network
print("\nTesting Value Network...")
value_net = ValueNetwork()
value = value_net.get_value(state)
print(f"Value estimate: {value.item():.4f}")

# Test Replay Buffer
print("\nTesting Replay Buffer...")
buffer = ReplayBuffer(capacity=100)
for i in range(10):
    buffer.add(
        state=np.random.randn(256),
        action=np.array([0, 1, 2]),
        reward=np.random.randn(),
        next_state=np.random.randn(256),
        done=False,
        log_prob=-1.0,
        value=0.5
    )
buffer.compute_returns_and_advantages()
batch = buffer.sample_batch(5)
print(f"Batch keys: {batch.keys()}")
print(f"States shape: {batch['states'].shape}")

print("\nAll tests passed!")