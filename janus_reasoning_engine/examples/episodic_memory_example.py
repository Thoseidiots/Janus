"""
Example usage of the Episodic Memory System.

Demonstrates storing experiences, retrieving similar experiences,
analyzing performance, and using experience replay for learning.

**Validates: Requirements REQ-6.1, REQ-6.4**
"""

import logging
from janus_reasoning_engine.memory import (
    UnifiedMemory,
    EpisodicMemory,
    OutcomeType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate episodic memory system."""
    
    # Initialize unified memory
    logger.info("Initializing unified memory...")
    unified_memory = UnifiedMemory(
        hbm_dimension=10000,
        hbm_sparsity=0.1,
        sqlite_path="janus_reasoning.db",
        artifacts_dir="janus_artifacts",
    )
    unified_memory.initialize()
    
    # Create episodic memory system
    logger.info("Creating episodic memory system...")
    episodic_memory = EpisodicMemory(unified_memory)
    
    # Example 1: Store a successful job experience
    logger.info("\n=== Example 1: Storing a successful experience ===")
    success_id = episodic_memory.store_experience(
        context={
            "platform": "upwork",
            "job_title": "Python Web Development",
            "description": "Build a Django REST API for e-commerce",
            "budget": 500,
            "competition": "medium",
        },
        action={
            "type": "submit_proposal",
            "bid_amount": 500,
            "proposal_text": "I have 5 years of Django experience...",
            "estimated_time": "2 weeks",
        },
        outcome={
            "result": "hired",
            "final_earnings": 500,
            "client_rating": 5.0,
            "client_feedback": "Excellent work, very professional",
        },
        skills=["python", "django", "rest-api", "postgresql"],
        platforms=["upwork"],
        outcome_type=OutcomeType.SUCCESS,
        earnings=500.0,
        time_spent=20.0,  # hours
        difficulty=0.6,
        learning_value=0.3,
        tags=["web-dev", "backend", "e-commerce"],
    )
    logger.info(f"Stored successful experience: {success_id}")
    
    # Example 2: Store a failure experience
    logger.info("\n=== Example 2: Storing a failure experience ===")
    failure_id = episodic_memory.store_experience(
        context={
            "platform": "upwork",
            "job_title": "Machine Learning Model Development",
            "description": "Build a recommendation system",
            "budget": 2000,
            "competition": "high",
        },
        action={
            "type": "submit_proposal",
            "bid_amount": 1800,
            "proposal_text": "I can build this recommendation system...",
        },
        outcome={
            "result": "rejected",
            "reason": "Client chose someone with more ML experience",
        },
        skills=["machine-learning", "python"],
        platforms=["upwork"],
        outcome_type=OutcomeType.FAILURE,
        learning_value=0.8,  # High learning value - need to improve ML skills
        tags=["ml", "rejected", "skill-gap"],
    )
    logger.info(f"Stored failure experience: {failure_id}")
    
    # Example 3: Store more experiences for analysis
    logger.info("\n=== Example 3: Storing multiple experiences ===")
    
    # More Python/Django successes
    for i in range(3):
        episodic_memory.store_experience(
            context={"platform": "upwork", "job_title": f"Django Project {i+1}"},
            action={"type": "submit_proposal", "bid_amount": 300 + i*100},
            outcome={"result": "hired", "final_earnings": 300 + i*100},
            skills=["python", "django"],
            platforms=["upwork"],
            outcome_type=OutcomeType.SUCCESS,
            earnings=300.0 + i*100,
            time_spent=15.0 + i*5,
        )
    
    # Some JavaScript experiences
    episodic_memory.store_experience(
        context={"platform": "fiverr", "job_title": "React Component Development"},
        action={"type": "submit_proposal", "bid_amount": 200},
        outcome={"result": "hired", "final_earnings": 200},
        skills=["javascript", "react"],
        platforms=["fiverr"],
        outcome_type=OutcomeType.SUCCESS,
        earnings=200.0,
        time_spent=10.0,
    )
    
    logger.info("Stored 5 additional experiences")
    
    # Example 4: Retrieve similar experiences
    logger.info("\n=== Example 4: Retrieving similar experiences ===")
    similar = episodic_memory.retrieve_similar_experiences(
        "Django web development project",
        limit=5,
        similarity_threshold=0.0,
    )
    logger.info(f"Found {len(similar)} similar experiences")
    for exp in similar[:3]:  # Show first 3
        logger.info(f"  - {exp.context.get('job_title', 'Unknown')} "
                   f"({exp.outcome_type.value})")
    
    # Example 5: Retrieve by skill
    logger.info("\n=== Example 5: Retrieving by skill ===")
    python_exp = episodic_memory.retrieve_by_skill("python", limit=10)
    logger.info(f"Found {len(python_exp)} Python experiences")
    
    django_exp = episodic_memory.retrieve_by_skill("django", limit=10)
    logger.info(f"Found {len(django_exp)} Django experiences")
    
    # Example 6: Retrieve by platform
    logger.info("\n=== Example 6: Retrieving by platform ===")
    upwork_exp = episodic_memory.retrieve_by_platform("upwork", limit=10)
    logger.info(f"Found {len(upwork_exp)} Upwork experiences")
    
    fiverr_exp = episodic_memory.retrieve_by_platform("fiverr", limit=10)
    logger.info(f"Found {len(fiverr_exp)} Fiverr experiences")
    
    # Example 7: Retrieve successful experiences
    logger.info("\n=== Example 7: Retrieving successful experiences ===")
    successes = episodic_memory.retrieve_successful_experiences(limit=10)
    logger.info(f"Found {len(successes)} successful experiences")
    
    high_earning = episodic_memory.retrieve_successful_experiences(
        min_earnings=400.0
    )
    logger.info(f"Found {len(high_earning)} high-earning experiences (>$400)")
    
    # Example 8: Experience replay for learning
    logger.info("\n=== Example 8: Experience replay batch ===")
    replay_batch = episodic_memory.get_experience_replay_batch(
        batch_size=5,
        prioritize_failures=True,
    )
    logger.info(f"Got replay batch of {len(replay_batch)} experiences")
    logger.info("Batch composition:")
    for exp in replay_batch:
        logger.info(f"  - {exp.outcome_type.value}: "
                   f"{exp.context.get('job_title', 'Unknown')}")
    
    # Example 9: Analyze skill performance
    logger.info("\n=== Example 9: Analyzing skill performance ===")
    python_stats = episodic_memory.analyze_skill_performance("python")
    logger.info(f"Python skill statistics:")
    logger.info(f"  Total experiences: {python_stats['total_experiences']}")
    logger.info(f"  Success rate: {python_stats['success_rate']:.2%}")
    logger.info(f"  Average earnings: ${python_stats['average_earnings']:.2f}")
    logger.info(f"  Total earnings: ${python_stats['total_earnings']:.2f}")
    
    django_stats = episodic_memory.analyze_skill_performance("django")
    logger.info(f"\nDjango skill statistics:")
    logger.info(f"  Total experiences: {django_stats['total_experiences']}")
    logger.info(f"  Success rate: {django_stats['success_rate']:.2%}")
    logger.info(f"  Average earnings: ${django_stats['average_earnings']:.2f}")
    
    ml_stats = episodic_memory.analyze_skill_performance("machine-learning")
    logger.info(f"\nMachine Learning skill statistics:")
    logger.info(f"  Total experiences: {ml_stats['total_experiences']}")
    logger.info(f"  Success rate: {ml_stats['success_rate']:.2%}")
    logger.info(f"  Average earnings: ${ml_stats['average_earnings']:.2f}")
    
    # Example 10: Analyze platform performance
    logger.info("\n=== Example 10: Analyzing platform performance ===")
    upwork_stats = episodic_memory.analyze_platform_performance("upwork")
    logger.info(f"Upwork platform statistics:")
    logger.info(f"  Total experiences: {upwork_stats['total_experiences']}")
    logger.info(f"  Success rate: {upwork_stats['success_rate']:.2%}")
    logger.info(f"  Average earnings: ${upwork_stats['average_earnings']:.2f}")
    logger.info(f"  Total earnings: ${upwork_stats['total_earnings']:.2f}")
    
    fiverr_stats = episodic_memory.analyze_platform_performance("fiverr")
    logger.info(f"\nFiverr platform statistics:")
    logger.info(f"  Total experiences: {fiverr_stats['total_experiences']}")
    logger.info(f"  Success rate: {fiverr_stats['success_rate']:.2%}")
    logger.info(f"  Average earnings: ${fiverr_stats['average_earnings']:.2f}")
    
    # Example 11: Overall statistics
    logger.info("\n=== Example 11: Overall statistics ===")
    overall_stats = episodic_memory.get_statistics()
    logger.info(f"Overall episodic memory statistics:")
    logger.info(f"  Total experiences: {overall_stats['total_experiences']}")
    logger.info(f"  Successes: {overall_stats['success_count']}")
    logger.info(f"  Failures: {overall_stats['failure_count']}")
    logger.info(f"  Success rate: {overall_stats['success_rate']:.2%}")
    
    # Example 12: Learning insights
    logger.info("\n=== Example 12: Learning insights ===")
    logger.info("Key insights from episodic memory:")
    
    # Identify strongest skills
    skills_to_analyze = ["python", "django", "javascript", "react", "machine-learning"]
    skill_stats = []
    for skill in skills_to_analyze:
        stats = episodic_memory.analyze_skill_performance(skill)
        if stats["total_experiences"] > 0:
            skill_stats.append((skill, stats))
    
    # Sort by success rate
    skill_stats.sort(key=lambda x: x[1]["success_rate"], reverse=True)
    
    logger.info("\nSkills ranked by success rate:")
    for skill, stats in skill_stats:
        if stats["total_experiences"] > 0:
            logger.info(f"  {skill}: {stats['success_rate']:.2%} "
                       f"({stats['total_experiences']} experiences)")
    
    # Identify skill gaps (low success rate)
    logger.info("\nSkills needing improvement:")
    for skill, stats in skill_stats:
        if stats["total_experiences"] > 0 and stats["success_rate"] < 0.5:
            logger.info(f"  {skill}: {stats['success_rate']:.2%} success rate "
                       f"- consider learning more")
    
    # Shutdown
    logger.info("\n=== Shutting down ===")
    unified_memory.shutdown()
    logger.info("Done!")


if __name__ == "__main__":
    main()
