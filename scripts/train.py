"""
Main training script for Mental Wellness Companion
Combines PPO and Contextual Bandit training
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.rl.ppo_trainer import PPOTrainer
from src.rl.contextual_bandit import ContextualBandit
from src.simulation.persona_generator import PersonaGenerator
from src.utils.logger import logger
from src.utils.helpers import set_random_seeds
from config.settings import TRAINING_CONFIG, DATA_DIR


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Mental Wellness Companion")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--personas", type=int, default=100, help="Number of personas to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--load-checkpoint", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    
    args = parser.parse_args()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    logger.info("="*50)
    logger.info("MENTAL WELLNESS COMPANION TRAINING")
    logger.info("="*50)
    
    # Step 1: Generate or load personas
    logger.info(f"\n📊 Generating {args.personas} training personas...")
    persona_generator = PersonaGenerator()
    
    # Check if personas already exist
    existing_personas = persona_generator.load_personas("training")
    if len(existing_personas) >= args.personas:
        logger.info(f"Using {len(existing_personas)} existing personas")
        personas = existing_personas[:args.personas]
    else:
        logger.info(f"Generating new personas...")
        personas = persona_generator.generate_batch(
            batch_size=args.personas,
            prefix="training"
        )
    
    # Step 2: Initialize PPO trainer
    logger.info("\n🤖 Initializing PPO trainer...")
    ppo_trainer = PPOTrainer()
    
    if args.load_checkpoint:
        checkpoint_path = Path(args.load_checkpoint)
        if checkpoint_path.exists():
            ppo_trainer.load_checkpoint(checkpoint_path)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Step 3: Train PPO
    logger.info("\n🎯 Starting PPO training...")
    logger.info(f"Episodes: {args.episodes}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Calculate timesteps (assuming average 5 steps per episode)
    total_timesteps = args.episodes * 5
    
    ppo_trainer.train(
        total_timesteps=total_timesteps,
        num_steps_per_collect=100,
        num_epochs=TRAINING_CONFIG["ppo_epochs"],
        batch_size=args.batch_size,
        save_freq=10,
        render=args.render
    )
    
    # Step 4: Initialize and train contextual bandit
    logger.info("\n🎰 Training contextual bandit...")
    bandit = ContextualBandit()
    
    # Simulate bandit training with collected data
    if ppo_trainer.episode_rewards:
        logger.info("Training bandit with collected interaction data...")
        # In production, would use actual interaction data
        # For now, simulate some training
        for _ in range(100):
            context = ppo_trainer.env.observation_space.sample()
            resource, confidence = bandit.select_resource(context)
            # Simulate reward based on PPO performance
            reward = min(1.0, max(0.0, ppo_trainer.best_reward / 10.0))
            bandit.update_resource(resource, context, reward)
        
        bandit.save()
        logger.info("Contextual bandit trained and saved")
    
    # Step 5: Save final statistics
    logger.info("\n📈 Training completed! Saving statistics...")
    
    final_stats = {
        "episodes_completed": len(ppo_trainer.episode_rewards),
        "best_reward": ppo_trainer.best_reward,
        "average_reward": sum(ppo_trainer.episode_rewards[-100:]) / min(100, len(ppo_trainer.episode_rewards)) if ppo_trainer.episode_rewards else 0,
        "final_policy_loss": ppo_trainer.training_stats["policy_losses"][-1] if ppo_trainer.training_stats["policy_losses"] else 0,
        "final_value_loss": ppo_trainer.training_stats["value_losses"][-1] if ppo_trainer.training_stats["value_losses"] else 0,
        "bandit_statistics": bandit.get_statistics()
    }
    
    stats_path = DATA_DIR / "training_summary.json"
    with open(stats_path, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    logger.info(f"\n✅ Training Summary:")
    logger.info(f"  Episodes: {final_stats['episodes_completed']}")
    logger.info(f"  Best Reward: {final_stats['best_reward']:.2f}")
    logger.info(f"  Average Reward (last 100): {final_stats['average_reward']:.2f}")
    logger.info(f"  Best Resources: {final_stats['bandit_statistics']['best_resources']}")
    
    logger.info(f"\n📁 Models saved to: {DATA_DIR / 'models'}")
    logger.info("Training complete! 🎉")


if __name__ == "__main__":
    main()