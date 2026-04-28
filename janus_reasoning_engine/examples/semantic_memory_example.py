"""
Example usage of the Semantic Memory System.

Demonstrates:
- Adding and managing skills with proficiency tracking
- Storing procedures as step-by-step knowledge
- Organizing knowledge hierarchically
- Retrieving knowledge by topic/skill
- Updating and refining knowledge
"""

import logging
from janus_reasoning_engine.memory import (
    UnifiedMemory,
    SemanticMemory,
    SkillLevel,
    KnowledgeType,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Demonstrate semantic memory capabilities."""
    
    # Initialize unified memory
    logger.info("Initializing unified memory...")
    unified_memory = UnifiedMemory(
        hbm_dimension=10000,
        hbm_sparsity=0.1,
        sqlite_path="janus_semantic_demo.db",
        artifacts_dir="janus_artifacts_demo",
    )
    unified_memory.initialize()
    
    # Create semantic memory
    semantic_memory = SemanticMemory(unified_memory)
    
    # ==================== Skill Management ====================
    logger.info("\n=== Skill Management ===")
    
    # Add programming skills
    python_id = semantic_memory.add_skill(
        name="Python Programming",
        description="Ability to write Python code for various applications",
        level=SkillLevel.ADVANCED,
        confidence=0.85,
        tags=["programming", "python", "backend"],
        domains=["software-development"],
    )
    logger.info(f"Added Python skill: {python_id}")
    
    javascript_id = semantic_memory.add_skill(
        name="JavaScript",
        description="Frontend and backend JavaScript development",
        level=SkillLevel.INTERMEDIATE,
        confidence=0.7,
        tags=["programming", "javascript", "frontend"],
        domains=["software-development"],
    )
    logger.info(f"Added JavaScript skill: {javascript_id}")
    
    # Add a parent-child skill hierarchy
    web_dev_id = semantic_memory.add_skill(
        name="Web Development",
        description="Building web applications",
        level=SkillLevel.ADVANCED,
        confidence=0.8,
        domains=["software-development"],
    )
    
    frontend_id = semantic_memory.add_skill(
        name="Frontend Development",
        description="Building user interfaces with React and Vue",
        level=SkillLevel.INTERMEDIATE,
        confidence=0.75,
        parent_skill=web_dev_id,
        tags=["frontend", "react", "vue"],
        domains=["software-development"],
    )
    logger.info(f"Added Frontend skill as child of Web Development")
    
    # Update skill after successful use
    semantic_memory.update_skill(
        python_id,
        level=SkillLevel.EXPERT,
        confidence=0.9,
        increment_use=True,
        increment_success=True,
    )
    logger.info("Updated Python skill to Expert level")
    
    # Get skills by domain
    programming_skills = semantic_memory.get_skills_by_domain("software-development")
    logger.info(f"Found {len(programming_skills)} programming skills")
    for skill in programming_skills:
        logger.info(f"  - {skill.name} ({skill.level.value})")
    
    # Get skill hierarchy
    hierarchy = semantic_memory.get_skill_hierarchy(web_dev_id)
    logger.info(f"Web Development hierarchy has {len(hierarchy['children'])} children")
    
    # ==================== Procedure Management ====================
    logger.info("\n=== Procedure Management ===")
    
    # Add a procedure for deploying applications
    deploy_proc_id = semantic_memory.add_procedure(
        name="Deploy Python App to AWS",
        description="Complete procedure for deploying a Python application to AWS EC2",
        steps=[
            {
                "action": "prepare",
                "description": "Prepare application for deployment",
                "details": "Run tests, build artifacts, update dependencies",
            },
            {
                "action": "package",
                "description": "Package application",
                "details": "Create deployment package with all dependencies",
            },
            {
                "action": "upload",
                "description": "Upload to AWS S3",
                "details": "Upload package to S3 bucket for deployment",
            },
            {
                "action": "deploy",
                "description": "Deploy to EC2",
                "details": "Launch EC2 instance and deploy application",
            },
            {
                "action": "verify",
                "description": "Verify deployment",
                "details": "Run health checks and verify application is running",
            },
        ],
        required_skills=[python_id],
        tags=["deployment", "aws", "devops"],
        domains=["software-development", "devops"],
    )
    logger.info(f"Added deployment procedure: {deploy_proc_id}")
    
    # Update procedure usage
    semantic_memory.update_procedure_usage(deploy_proc_id, success=True)
    logger.info("Updated procedure usage statistics")
    
    # Retrieve procedure
    procedure = semantic_memory.get_procedure(deploy_proc_id)
    logger.info(f"Procedure '{procedure.name}' has {len(procedure.steps)} steps")
    logger.info(f"  Success rate: {procedure.success_count}/{procedure.use_count}")
    
    # ==================== Knowledge Management ====================
    logger.info("\n=== Knowledge Management ===")
    
    # Add facts
    fact_id = semantic_memory.add_knowledge(
        knowledge_type=KnowledgeType.FACT,
        name="Python is interpreted",
        content={
            "fact": "Python is an interpreted, high-level programming language",
            "details": "Python code is executed line by line by the interpreter",
            "implications": ["Slower than compiled languages", "Easier to debug", "More flexible"],
        },
        confidence=0.95,
        source="Python documentation",
        tags=["python", "programming-languages"],
        domains=["software-development"],
    )
    logger.info(f"Added fact: {fact_id}")
    
    # Add concepts
    concept_id = semantic_memory.add_knowledge(
        knowledge_type=KnowledgeType.CONCEPT,
        name="RESTful API Design",
        content={
            "definition": "Architectural style for designing networked applications",
            "principles": [
                "Stateless communication",
                "Client-server architecture",
                "Cacheable responses",
                "Uniform interface",
            ],
            "http_methods": {
                "GET": "Retrieve resource",
                "POST": "Create resource",
                "PUT": "Update resource",
                "DELETE": "Delete resource",
            },
        },
        confidence=0.9,
        tags=["api", "rest", "architecture"],
        domains=["software-development"],
    )
    logger.info(f"Added concept: {concept_id}")
    
    # Add rules
    rule_id = semantic_memory.add_knowledge(
        knowledge_type=KnowledgeType.RULE,
        name="Always validate user input",
        content={
            "rule": "Never trust user input without validation",
            "rationale": "Prevents security vulnerabilities like SQL injection and XSS",
            "examples": [
                "Validate email format",
                "Sanitize HTML input",
                "Use parameterized queries",
                "Implement rate limiting",
            ],
            "severity": "critical",
        },
        confidence=1.0,
        tags=["security", "best-practices", "validation"],
        domains=["software-development", "security"],
    )
    logger.info(f"Added rule: {rule_id}")
    
    # Update knowledge
    semantic_memory.update_knowledge(
        concept_id,
        content={
            "definition": "Architectural style for designing networked applications",
            "principles": [
                "Stateless communication",
                "Client-server architecture",
                "Cacheable responses",
                "Uniform interface",
                "Layered system",  # Added new principle
            ],
            "http_methods": {
                "GET": "Retrieve resource",
                "POST": "Create resource",
                "PUT": "Update resource",
                "DELETE": "Delete resource",
                "PATCH": "Partial update",  # Added new method
            },
        },
        confidence=0.95,
    )
    logger.info("Updated REST API concept with additional information")
    
    # ==================== Skill Inventory ====================
    logger.info("\n=== Skill Inventory ===")
    
    inventory = semantic_memory.get_skill_inventory()
    logger.info(f"Total skills: {inventory['total_skills']}")
    logger.info("Skills by level:")
    for level, skills in inventory['by_level'].items():
        if skills:
            logger.info(f"  {level}: {', '.join(skills)}")
    
    logger.info("Skills by domain:")
    for domain, skills in inventory['by_domain'].items():
        logger.info(f"  {domain}: {', '.join(skills)}")
    
    # ==================== Statistics ====================
    logger.info("\n=== Statistics ===")
    
    stats = semantic_memory.get_statistics()
    logger.info(f"Total skills: {stats['total_skills']}")
    logger.info(f"Total procedures: {stats['total_procedures']}")
    logger.info(f"Total knowledge: {stats['total_knowledge']}")
    logger.info(f"Cached items: {stats['cached_skills']} skills, "
                f"{stats['cached_procedures']} procedures, "
                f"{stats['cached_knowledge']} knowledge")
    
    # Cleanup
    logger.info("\n=== Cleanup ===")
    unified_memory.shutdown()
    logger.info("Semantic memory demo complete!")


if __name__ == "__main__":
    main()
