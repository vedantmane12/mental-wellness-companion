"""
Improved training script with diversity rewards and better exploration
"""
import numpy as np
import torch
from pathlib import Path
import sys
import json
from typing import Dict, Any, List, Optional  # Add this import!

sys.path.append(str(Path(__file__).parent.parent))

from src.rl.ppo_trainer import PPOTrainer
from src.rl.environment import MentalWellnessEnv
from src.simulation.persona_generator import PersonaGenerator
from src.utils.logger import logger, training_logger
from src.utils.helpers import set_random_seeds
from config.settings import TRAINING_CONFIG, DATA_DIR, AGENT_CONFIG  # Add AGENT_CONFIG


class ImprovedMentalWellnessEnv(MentalWellnessEnv):
    """Enhanced environment with diversity rewards"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recent_strategies = []
        self.strategy_counts = {strategy: 0 for strategy in AGENT_CONFIG["conversation_strategies"]}
    
    def _calculate_reward(self, metrics: Dict[str, float], action_dict: Dict[str, Any]) -> float:
        """Enhanced reward calculation with diversity bonus"""
        # Base reward from parent class
        reward = super()._calculate_reward(metrics, action_dict)
        
        # Add diversity bonus
        strategy = action_dict["conversation_strategy"]
        
        # Bonus for using underused strategies
        total_uses = sum(self.strategy_counts.values())
        if total_uses > 0:
            strategy_frequency = self.strategy_counts[strategy] / total_uses
            # Higher bonus for less frequently used strategies
            diversity_bonus = 0.2 * (1 - strategy_frequency)
            reward += diversity_bonus
        
        # Penalty for repeating recent strategies
        if strategy in self.recent_strategies[-3:]:
            reward -= 0.1
        
        # Track strategy usage
        self.recent_strategies.append(strategy)
        if len(self.recent_strategies) > 10:
            self.recent_strategies.pop(0)
        self.strategy_counts[strategy] += 1
        
        # Context-appropriate strategy bonus
        if self._is_appropriate_strategy(strategy, metrics):
            reward += 0.15
        
        return reward
    
    def _is_appropriate_strategy(self, strategy: str, metrics: Dict[str, Any]) -> bool:
        """Check if strategy is appropriate for context"""
        mood_change = metrics.get("mood_change", 0)
        engagement = metrics.get("engagement", 0.5)
        
        # Simple heuristics for appropriate strategies
        if strategy == "empathetic_listening" and engagement < 0.3:
            return True
        if strategy == "problem_solving" and mood_change > 0.1:
            return True
        if strategy == "validation" and mood_change < -0.1:
            return True
        if strategy == "motivational" and engagement > 0.7:
            return True
        if strategy == "cognitive_behavioral" and 0.3 < engagement < 0.7:
            return True
        
        return False


class ImprovedPPOTrainer(PPOTrainer):
    """Enhanced PPO trainer with better exploration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Start with higher entropy for more exploration
        self.entropy_coef = 0.05  # Increased from 0.01
        self.entropy_decay = 0.995  # Gradually reduce entropy
        self.current_episode = 0  # Add this to track episodes
    
    def collect_trajectories(self, num_steps: int = 2048, render: bool = False) -> Dict[str, float]:
        """Collect trajectories with adaptive exploration"""
        # Temperature for exploration (starts high, decays)
        temperature = 1.5 * (0.95 ** (self.current_episode / 10))
        
        # Collect trajectories with parent method
        stats = super().collect_trajectories(num_steps, render)
        
        # Update episode counter
        self.current_episode += stats.get("episodes_completed", 0)
        
        # Decay entropy coefficient
        self.entropy_coef *= self.entropy_decay
        self.entropy_coef = max(self.entropy_coef, 0.001)  # Minimum entropy
        
        return stats
    
    def update_policy(self, num_epochs: int = 10, batch_size: int = 64) -> Dict[str, float]:
        """Update policy with diversity-aware loss"""
        stats = super().update_policy(num_epochs, batch_size)
        
        # Log diversity metrics
        training_logger.info(f"Entropy coefficient: {self.entropy_coef:.4f}")
        
        return stats


def train_with_diversity():
    """Train with improved diversity"""
    set_random_seeds(42)
    
    logger.info("="*60)
    logger.info("IMPROVED TRAINING WITH DIVERSITY REWARDS")
    logger.info("="*60)
    
    # Generate diverse personas
    logger.info("Generating diverse training personas...")
    persona_gen = PersonaGenerator()
    
    # Create personas with different mental health profiles
    persona_types = [
        "anxious", "depressed", "stressed", "angry", 
        "confused", "lonely", "overwhelmed", "grieving"
    ]
    
    personas = []
    for i, p_type in enumerate(persona_types * 3):  # 24 personas
        try:
            # Generate persona with specific type
            prompt_persona = persona_gen.client.generate_persona(i, p_type)
            personas.append(prompt_persona)
        except:
            # Fallback to default generation
            persona = persona_gen._generate_fallback_persona(i, p_type)
            personas.append(persona)
    
    logger.info(f"Generated {len(personas)} diverse personas")
    
    # Create improved environment
    env = ImprovedMentalWellnessEnv(
        personas=personas,
        max_episode_length=8,  # Shorter episodes for faster learning
        training_mode=True
    )
    
    # Create improved trainer
    trainer = ImprovedPPOTrainer(
        env=env,
        learning_rate=5e-4,  # Slightly higher LR
        clip_ratio=0.2,
        value_loss_coef=0.5,
        entropy_coef=0.05,  # Higher initial entropy
        max_grad_norm=0.5
    )
    
    # Training parameters
    total_episodes = 200  # More episodes
    steps_per_episode = 8
    total_timesteps = total_episodes * steps_per_episode
    
    logger.info(f"Starting training for {total_episodes} episodes...")
    
    # Train with curriculum learning
    for phase in range(3):
        logger.info(f"\n{'='*40}")
        logger.info(f"Training Phase {phase + 1}/3")
        logger.info(f"{'='*40}")
        
        # Adjust training focus
        if phase == 0:
            # Phase 1: Learn basic responses
            trainer.entropy_coef = 0.05
            logger.info("Focus: Learning basic conversation strategies")
        elif phase == 1:
            # Phase 2: Improve strategy selection
            trainer.entropy_coef = 0.02
            logger.info("Focus: Refining strategy selection")
        else:
            # Phase 3: Fine-tune
            trainer.entropy_coef = 0.01
            logger.info("Focus: Fine-tuning responses")
        
        # Train for this phase
        trainer.train(
            total_timesteps=total_timesteps // 3,
            num_steps_per_collect=50,
            num_epochs=10,
            batch_size=32,
            save_freq=20,
            render=False
        )
        
        # Evaluate diversity
        evaluate_strategy_diversity(trainer)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    
    # Save final model
    trainer.save_best_model()
    
    # Print final statistics
    print_training_summary(trainer)
    
    return trainer


def evaluate_strategy_diversity(trainer):
    """Evaluate how diverse the learned strategies are"""
    logger.info("\nEvaluating strategy diversity...")
    
    strategy_counts = {strategy: 0 for strategy in AGENT_CONFIG["conversation_strategies"]}
    
    # Test on different emotional states
    test_states = [
        {"anxiety": 0.8, "depression": 0.2, "stress": 0.5, "anger": 0.1, "happiness": 0.2},
        {"anxiety": 0.2, "depression": 0.8, "stress": 0.3, "anger": 0.1, "happiness": 0.1},
        {"anxiety": 0.5, "depression": 0.5, "stress": 0.8, "anger": 0.3, "happiness": 0.2},
        {"anxiety": 0.3, "depression": 0.3, "stress": 0.3, "anger": 0.7, "happiness": 0.2},
    ]
    
    for emotional_state in test_states:
        # Create state vector
        from src.utils.state_manager import ConversationState
        state = ConversationState(
            emotional_state=np.array(list(emotional_state.values())),
            engagement_level=0.5,
            conversation_history=[],
            session_duration=0,
            time_since_last=0
        )
        
        state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0)
        
        # Get action from policy
        with torch.no_grad():
            action, _ = trainer.policy_net.sample_action(state_tensor, deterministic=True)
            strategy_idx = action[0, 0].item()
            strategy = AGENT_CONFIG["conversation_strategies"][strategy_idx]
            strategy_counts[strategy] += 1
    
    # Print diversity metrics
    logger.info("Strategy distribution:")
    for strategy, count in strategy_counts.items():
        percentage = (count / len(test_states)) * 100
        logger.info(f"  {strategy}: {percentage:.1f}%")
    
    # Calculate entropy as diversity measure
    total = sum(strategy_counts.values())
    entropy = -sum((c/total) * np.log(c/total + 1e-10) for c in strategy_counts.values() if c > 0)
    max_entropy = np.log(len(strategy_counts))
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0
    
    logger.info(f"Diversity score: {diversity_score:.2f} (1.0 = perfect diversity)")
    
    return diversity_score


def print_training_summary(trainer):
    """Print comprehensive training summary"""
    if not trainer.episode_rewards:
        return
    
    logger.info("\nTraining Summary:")
    logger.info(f"  Total episodes: {len(trainer.episode_rewards)}")
    logger.info(f"  Best reward: {trainer.best_reward:.2f}")
    logger.info(f"  Average reward (last 50): {np.mean(trainer.episode_rewards[-50:]):.2f}")
    
    if trainer.training_stats['policy_losses']:
        logger.info(f"  Final policy loss: {trainer.training_stats['policy_losses'][-1]:.4f}")
    if trainer.training_stats['value_losses']:
        logger.info(f"  Final value loss: {trainer.training_stats['value_losses'][-1]:.4f}")
    
    # Save enhanced statistics
    stats = {
        "total_episodes": len(trainer.episode_rewards),
        "best_reward": float(trainer.best_reward),
        "final_avg_reward": float(np.mean(trainer.episode_rewards[-50:])) if len(trainer.episode_rewards) >= 50 else float(np.mean(trainer.episode_rewards)),
        "episode_rewards": [float(r) for r in trainer.episode_rewards],
        "training_stats": {
            "policy_losses": [float(l) for l in trainer.training_stats['policy_losses']],
            "value_losses": [float(l) for l in trainer.training_stats['value_losses']],
            "entropies": [float(e) for e in trainer.training_stats['entropies']],
            "clip_fractions": [float(c) for c in trainer.training_stats['clip_fractions']]
        }
    }
    
    stats_path = DATA_DIR / "improved_training_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    trainer = train_with_diversity()