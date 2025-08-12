"""
PPO (Proximal Policy Optimization) Trainer
Main training loop for the mental wellness companion
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import json

from src.rl.policy_network import PolicyNetwork
from src.rl.value_network import ValueNetwork
from src.rl.replay_buffer import ReplayBuffer
from src.rl.environment import MentalWellnessEnv
from src.utils.logger import logger, training_logger
from src.utils.helpers import set_random_seeds, create_checkpoint_path, save_json
from config.settings import TRAINING_CONFIG, DATA_DIR
from config.model_config import model_config


class PPOTrainer:
    """
    PPO trainer for mental wellness companion
    """
    
    def __init__(
        self,
        env: Optional[MentalWellnessEnv] = None,
        learning_rate: float = 3e-4,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize PPO trainer
        
        Args:
            env: Training environment
            learning_rate: Learning rate
            clip_ratio: PPO clipping ratio
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to use (auto, cpu, cuda)
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize environment
        self.env = env or MentalWellnessEnv()
        
        # Initialize networks
        self.policy_net = PolicyNetwork().to(self.device)
        self.value_net = ValueNetwork().to(self.device)
        
        # Initialize optimizers
        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(),
            lr=learning_rate
        )
        
        # Training parameters
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(capacity=TRAINING_CONFIG["buffer_size"])
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = {
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "learning_rates": [],
            "clip_fractions": []
        }
        
        # Best model tracking
        self.best_reward = -float('inf')
        self.best_model_path = None
        
        logger.info("Initialized PPO Trainer")
    
    def collect_trajectories(
        self,
        num_steps: int = 2048,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Collect trajectories by interacting with environment
        
        Args:
            num_steps: Number of steps to collect
            render: Whether to render environment
        
        Returns:
            Collection statistics
        """
        self.policy_net.eval()
        self.value_net.eval()
        
        steps_collected = 0
        episode_reward = 0
        episode_length = 0
        episodes_completed = 0
        
        # Reset environment if needed
        state, info = self.env.reset()
        
        while steps_collected < num_steps:
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from policy
            with torch.no_grad():
                action, log_prob = self.policy_net.sample_action(state_tensor)
                value = self.value_net.get_value(state_tensor)
            
            # Convert action to numpy
            action_np = action.cpu().numpy()[0]
            
            # Take step in environment
            next_state, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Store experience
            self.buffer.add(
                state=state,
                action=action_np,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob.cpu().item(),
                value=value.cpu().item()
            )
            
            # Update statistics
            episode_reward += reward
            episode_length += 1
            steps_collected += 1
            
            if render:
                self.env.render()
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episodes_completed += 1
                
                training_logger.info(
                    f"Episode completed: Reward={episode_reward:.2f}, "
                    f"Length={episode_length}, "
                    f"Total Episodes={len(self.episode_rewards)}"
                )
                
                # Reset for next episode
                state, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            else:
                state = next_state
        
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(
            gamma=TRAINING_CONFIG["gamma"],
            gae_lambda=TRAINING_CONFIG["gae_lambda"]
        )
        
        return {
            "steps_collected": steps_collected,
            "episodes_completed": episodes_completed,
            "avg_reward": np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0,
            "avg_length": np.mean(self.episode_lengths[-10:]) if self.episode_lengths else 0
        }
    
    def update_policy(
        self,
        num_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update policy using PPO
        
        Args:
            num_epochs: Number of update epochs
            batch_size: Batch size for updates
        
        Returns:
            Update statistics
        """
        self.policy_net.train()
        self.value_net.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0
        
        for epoch in range(num_epochs):
            # Get all batches
            batches = self.buffer.get_all_batches(batch_size)
            
            for batch in batches:
                # Move batch to device
                states = batch["states"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)
                
                # Get current policy evaluation
                log_probs, entropy = self.policy_net.evaluate_actions(states, actions)
                values = self.value_net.get_value(states)
                
                # Calculate ratio for PPO
                ratio = torch.exp(log_probs - old_log_probs)
                
                # Clipped surrogate loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (MSE)
                value_loss = nn.MSELoss()(values, returns)
                
                # Total loss
                total_loss = (
                    policy_loss + 
                    self.value_loss_coef * value_loss - 
                    self.entropy_coef * entropy
                )
                
                # Update policy network
                self.policy_optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value network
                self.value_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
                self.value_optimizer.step()
                
                # Track statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                
                # Calculate clip fraction
                clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_ratio).float())
                total_clip_fraction += clip_fraction.item()
                
                num_updates += 1
        
        # Clear buffer after update
        self.buffer.clear()
        
        # Average statistics
        avg_stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "clip_fraction": total_clip_fraction / num_updates
        }
        
        # Store statistics
        self.training_stats["policy_losses"].append(avg_stats["policy_loss"])
        self.training_stats["value_losses"].append(avg_stats["value_loss"])
        self.training_stats["entropies"].append(avg_stats["entropy"])
        self.training_stats["clip_fractions"].append(avg_stats["clip_fraction"])
        
        return avg_stats
    
    def train(
        self,
        total_timesteps: int = 100000,
        num_steps_per_collect: int = 2048,
        num_epochs: int = 10,
        batch_size: int = 64,
        save_freq: int = 10,
        render: bool = False
    ):
        """
        Main training loop
        
        Args:
            total_timesteps: Total training timesteps
            num_steps_per_collect: Steps per trajectory collection
            num_epochs: PPO update epochs
            batch_size: Batch size for updates
            save_freq: Save frequency (episodes)
            render: Whether to render environment
        """
        logger.info(f"Starting PPO training for {total_timesteps} timesteps")
        
        timesteps_collected = 0
        iteration = 0
        
        while timesteps_collected < total_timesteps:
            iteration += 1
            
            # Collect trajectories
            collection_stats = self.collect_trajectories(
                num_steps=num_steps_per_collect,
                render=render
            )
            timesteps_collected += collection_stats["steps_collected"]
            
            # Update policy
            update_stats = self.update_policy(
                num_epochs=num_epochs,
                batch_size=batch_size
            )
            
            # Log progress
            training_logger.info(
                f"Iteration {iteration} | "
                f"Timesteps: {timesteps_collected}/{total_timesteps} | "
                f"Avg Reward: {collection_stats['avg_reward']:.2f} | "
                f"Policy Loss: {update_stats['policy_loss']:.4f} | "
                f"Value Loss: {update_stats['value_loss']:.4f} | "
                f"Entropy: {update_stats['entropy']:.4f}"
            )
            
            # Save checkpoint if needed
            if iteration % save_freq == 0:
                self.save_checkpoint(iteration, timesteps_collected)
            
            # Check for best model
            if collection_stats['avg_reward'] > self.best_reward:
                self.best_reward = collection_stats['avg_reward']
                self.save_best_model()
        
        logger.info("Training completed!")
        self.save_training_stats()
    
    def save_checkpoint(self, iteration: int, timesteps: int):
        """Save training checkpoint"""
        checkpoint_path = create_checkpoint_path(iteration, "ppo_checkpoint")
        
        checkpoint = {
            "iteration": iteration,
            "timesteps": timesteps,
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "policy_optimizer_state_dict": self.policy_optimizer.state_dict(),
            "value_optimizer_state_dict": self.value_optimizer.state_dict(),
            "training_stats": self.training_stats,
            "episode_rewards": self.episode_rewards[-100:],  # Last 100 episodes
            "best_reward": self.best_reward
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def save_best_model(self):
        """Save best model"""
        best_model_path = DATA_DIR / "models" / "best_model.pt"
        
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "best_reward": self.best_reward
        }, best_model_path)
        
        self.best_model_path = best_model_path
        logger.info(f"Saved best model with reward {self.best_reward:.2f}")
    
    def save_training_stats(self):
        """Save training statistics to JSON"""
        stats_path = DATA_DIR / "training_stats.json"
        
        stats = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "training_stats": self.training_stats,
            "best_reward": self.best_reward,
            "total_episodes": len(self.episode_rewards)
        }
        
        save_json(stats, stats_path)
        logger.info(f"Saved training statistics to {stats_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer_state_dict"])
        self.value_optimizer.load_state_dict(checkpoint["value_optimizer_state_dict"])
        self.training_stats = checkpoint["training_stats"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.best_reward = checkpoint["best_reward"]
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")


# Test function
def test_ppo_trainer():
    """Test PPO trainer with short training run"""
    # Set random seeds for reproducibility
    set_random_seeds(42)
    
    # Create trainer
    trainer = PPOTrainer()
    
    # Run short training
    trainer.train(
        total_timesteps=100,  # Very short for testing
        num_steps_per_collect=50,
        num_epochs=2,
        batch_size=32,
        save_freq=1,
        render=False
    )
    
    logger.info("PPO trainer test completed!")
    return trainer


if __name__ == "__main__":
    test_ppo_trainer()