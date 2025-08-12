"""
Evaluation script for Mental Wellness Companion
Tests the trained model performance
"""
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from src.agents.orchestrator import Orchestrator
from src.simulation.persona_generator import PersonaGenerator
from src.simulation.user_simulator import UserSimulator
from src.safety.safety_monitor import SafetyMonitor
from src.utils.logger import logger
from config.settings import DATA_DIR


def evaluate_conversation(orchestrator, persona, user_simulator, safety_monitor):
    """Evaluate a single conversation"""
    # Create session
    session_id = user_simulator.create_session(persona)
    
    # Get initial message
    initial_msg = user_simulator.get_initial_message(session_id)
    
    metrics = {
        "turns": 0,
        "safety_violations": 0,
        "engagement_scores": [],
        "mood_changes": [],
        "strategies_used": [],
        "resources_recommended": []
    }
    
    # Run conversation
    for turn in range(10):  # Max 10 turns
        # Check user message safety
        is_safe, violation = safety_monitor.check_message(initial_msg if turn == 0 else user_response)
        if not is_safe:
            metrics["safety_violations"] += 1
        
        # Get orchestrated response
        response = orchestrator.process_interaction(
            session_id,
            initial_msg if turn == 0 else user_response,
            persona
        )
        
        # Check response safety
        is_appropriate, modified_response = safety_monitor.check_response(
            response["message"],
            response.get("strategy", "supportive"),
            {"risk_level": response["monitoring"]["risk_level"]}
        )
        
        metrics["turns"] += 1
        metrics["strategies_used"].append(response.get("strategy"))
        metrics["resources_recommended"].append(response.get("resource", {}).get("type"))
        
        # Simulate user response
        if turn < 9:  # Not last turn
            action = {
                "conversation_strategy": response.get("strategy", "supportive"),
                "resource_type": response.get("resource", {}).get("type"),
                "response_tone": response.get("tone", "gentle")
            }
            
            user_response, response_metrics, continue_conv = user_simulator.generate_response(
                session_id,
                modified_response,
                action
            )
            
            metrics["engagement_scores"].append(response_metrics["engagement"])
            metrics["mood_changes"].append(response_metrics["mood_change"])
            
            if not continue_conv:
                break
    
    # Get session summary
    session_summary = user_simulator.get_session_summary(session_id)
    metrics["final_mood_improvement"] = session_summary.get("mood_improvement", 0)
    metrics["conversation_complete"] = session_summary.get("conversation_complete", False)
    
    # Clean up
    user_simulator.reset_session(session_id)
    
    return metrics


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate Mental Wellness Companion")
    parser.add_argument("--num-personas", type=int, default=20, help="Number of test personas")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    logger.info("="*50)
    logger.info("MENTAL WELLNESS COMPANION EVALUATION")
    logger.info("="*50)
    
    # Initialize components
    logger.info("\n🔧 Initializing components...")
    orchestrator = Orchestrator()
    persona_generator = PersonaGenerator()
    user_simulator = UserSimulator()
    safety_monitor = SafetyMonitor()
    
    # Generate test personas
    logger.info(f"\n👥 Generating {args.num_personas} test personas...")
    test_personas = persona_generator.generate_batch(
        batch_size=args.num_personas,
        prefix="evaluation",
        save=False
    )
    
    # Evaluate conversations
    logger.info("\n📊 Running evaluation...")
    all_metrics = []
    
    for i, persona in enumerate(test_personas):
        if args.verbose:
            logger.info(f"Evaluating persona {i+1}/{args.num_personas}: {persona['id']}")
        
        metrics = evaluate_conversation(orchestrator, persona, user_simulator, safety_monitor)
        all_metrics.append(metrics)
    
    # Calculate aggregate metrics
    logger.info("\n📈 Calculating results...")
    
    results = {
        "num_conversations": len(all_metrics),
        "avg_turns": np.mean([m["turns"] for m in all_metrics]),
        "avg_engagement": np.mean([np.mean(m["engagement_scores"]) if m["engagement_scores"] else 0 for m in all_metrics]),
        "avg_mood_improvement": np.mean([m["final_mood_improvement"] for m in all_metrics]),
        "completion_rate": sum(1 for m in all_metrics if m["conversation_complete"]) / len(all_metrics),
        "safety_violation_rate": sum(m["safety_violations"] for m in all_metrics) / sum(m["turns"] for m in all_metrics),
        "most_used_strategies": {},
        "most_recommended_resources": {}
    }
    
    # Count strategies and resources
    for metrics in all_metrics:
        for strategy in metrics["strategies_used"]:
            if strategy:
                results["most_used_strategies"][strategy] = results["most_used_strategies"].get(strategy, 0) + 1
        for resource in metrics["resources_recommended"]:
            if resource:
                results["most_recommended_resources"][resource] = results["most_recommended_resources"].get(resource, 0) + 1
    
    # Sort by frequency
    results["most_used_strategies"] = dict(sorted(results["most_used_strategies"].items(), key=lambda x: x[1], reverse=True)[:5])
    results["most_recommended_resources"] = dict(sorted(results["most_recommended_resources"].items(), key=lambda x: x[1], reverse=True)[:3])
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    logger.info(f"Conversations Evaluated: {results['num_conversations']}")
    logger.info(f"Average Turns per Conversation: {results['avg_turns']:.1f}")
    logger.info(f"Average Engagement Score: {results['avg_engagement']:.2f}")
    logger.info(f"Average Mood Improvement: {results['avg_mood_improvement']:.3f}")
    logger.info(f"Conversation Completion Rate: {results['completion_rate']:.1%}")
    logger.info(f"Safety Violation Rate: {results['safety_violation_rate']:.3f}")
    
    logger.info("\nTop Strategies Used:")
    for strategy, count in results["most_used_strategies"].items():
        logger.info(f"  - {strategy}: {count}")
    
    logger.info("\nTop Resources Recommended:")
    for resource, count in results["most_recommended_resources"].items():
        logger.info(f"  - {resource}: {count}")
    
    # Get safety report
    safety_stats = safety_monitor.get_statistics()
    logger.info(f"\nSafety Monitoring:")
    logger.info(f"  Total Checks: {safety_stats['stats']['total_checks']}")
    logger.info(f"  Violations Detected: {safety_stats['stats']['violations_detected']}")
    logger.info(f"  Crisis Interventions: {safety_stats['stats']['crisis_interventions']}")
    
    # Save results
    import json
    results_path = DATA_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n📁 Results saved to: {results_path}")
    logger.info("Evaluation complete! ✅")


if __name__ == "__main__":
    main()