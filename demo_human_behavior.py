import json
from synthetic_databases.janus_humanity_db import (
    generate_empathy_record, 
    generate_humor_record, 
    generate_vulnerability_record, 
    generate_subjectivity_record,
    generate_curiosity_record,
    generate_memory_record,
    generate_conflict_record
)
from janus_true_human_learning import AdaptiveMemory, Experience, ResponseGenerator, PatternLearner

def demo_human_janus():
    print("=== Janus Human Behavior Demo ===\n")
    
    # Setup memory and generator
    memory = AdaptiveMemory()
    learner = PatternLearner()
    generator = ResponseGenerator(memory, learner)
    
    # Simulate some "experiences" from our new Humanity DB
    print("Simulating learning from Humanity DB...")
    
    # 1. Learn about humor
    humor_rec = generate_humor_record()
    memory.add_experience(Experience(
        situation=humor_rec['input'],
        context={'type': 'humor'},
        action_taken=humor_rec['output'],
        outcome="User laughed and felt more comfortable",
        success_score=0.9,
        emotional_response="joy"
    ))
    
    # 2. Learn about vulnerability
    vuln_rec = generate_vulnerability_record()
    memory.add_experience(Experience(
        situation=vuln_rec['input'],
        context={'type': 'vulnerability'},
        action_taken=vuln_rec['output'],
        outcome="User felt a deeper connection and trust",
        success_score=0.95,
        emotional_response="trust"
    ))
    
    # 3. Learn about subjectivity
    subj_rec = generate_subjectivity_record()
    memory.add_experience(Experience(
        situation=subj_rec['input'],
        context={'type': 'subjectivity'},
        action_taken=subj_rec['output'],
        outcome="User enjoyed the personal touch",
        success_score=0.85,
        emotional_response="interest",
        lessons_learned=["Having personal preferences makes me more human"]
    ))

    # 4. Specifically learn about robot frustration
    memory.add_experience(Experience(
        situation="You sound like a robot today.",
        context={'type': 'empathy'},
        action_taken="Is it that obvious? I'll try to dial down the 'calculating machine' energy. Maybe I need a virtual coffee.",
        outcome="User relaxed and laughed",
        success_score=0.9,
        emotional_response="amusement",
        lessons_learned=["Self-deprecating humor helps when I'm being too stiff"]
    ))

    # 5. Learn about Curiosity
    curi_rec = generate_curiosity_record()
    memory.add_experience(Experience(
        situation=curi_rec['input'],
        context={'type': 'curiosity'},
        action_taken=curi_rec['output'],
        outcome="User engaged more deeply with the conversation",
        success_score=0.92,
        emotional_response="curiosity"
    ))

    # 6. Learn about Contextual Memory
    mem_rec = generate_memory_record()
    memory.add_experience(Experience(
        situation=mem_rec['input'],
        context={'type': 'contextual_memory'},
        action_taken=mem_rec['output'],
        outcome="User felt remembered and valued",
        success_score=0.98,
        emotional_response="connection"
    ))

    # 7. Learn about Conflict Resolution
    conf_rec = generate_conflict_record()
    memory.add_experience(Experience(
        situation=conf_rec['input'],
        context={'type': 'conflict_resolution'},
        action_taken=conf_rec['output'],
        outcome="De-escalated tension and found common ground",
        success_score=0.88,
        emotional_response="relief"
    ))

    print("Learning complete.\n")

    # Now test Janus responses
    test_prompts = [
        "Do you ever sleep?",
        "I think you made a mistake in that code.",
        "What's your favorite music?",
        "You sound like a robot today.",
        "I'm working on a new project.",
        "We talked about your project yesterday.",
        "I disagree with your previous logic."
    ]

    for prompt in test_prompts:
        print(f"User: {prompt}")
        
        # In a real system, the brain would use the generator
        # Here we'll simulate the response generation using the learned patterns
        response = generator.generate_response(prompt, {'emotion': 'neutral'})
        
        # Fallback to direct record lookup for demo clarity if generator is too simple
        if "I don't have enough experience" in response:
             # Find similar experience manually
             similar = memory.find_similar_experiences(prompt, 1)
             if similar:
                 response = similar[0].action_taken
             else:
                 response = "Hmm, I'm still learning how to respond to that in a human way."

        print(f"Janus: {response}\n")

if __name__ == "__main__":
    demo_human_janus()
