from model import GroundedModel
from curriculum import GradePreK, Grade1, Grade2, Grade3, Grade4, Grade5, Grade6, Grade7, Grade8, Grade9, Grade10, Grade11, Grade12
from game_curriculum import GameAIDataset, GameAIQuestDataset

def main():
    """Main function to run the training task with game data."""
    print("Game AI Training script initialized.")
    # 1. Initialize the model
    model = GroundedModel()
    print("\nModel initialized.")
    # 2. Define the curriculum stages
    # (We can run a subset or the whole thing, but let's focus on the game data)
    curriculum_stages = [GradePreK, Grade1, Grade2, Grade3, Grade4, Grade5, Grade6, Grade7, Grade8, Grade9, Grade10, Grade11, Grade12]
    # 3. Add game-specific datasets as extracurricular or final stages
    game_stages = [GameAIDataset, GameAIQuestDataset]
    # 4. Sequentially process each stage
    for stage_class in curriculum_stages + game_stages:
        stage_instance = stage_class()
        stage_name = stage_instance.__class__.__name__
        print(f"\n--- Starting {stage_name} Stage ---")
        dataset = stage_instance.generate_dataset(samples=20)
        print(f"Generated {len(dataset)} samples for {stage_name}:")
        for i, (prompt, answer) in enumerate(dataset):
            print(f"  Sample {i+1}: Prompt: {prompt} -> Answer: {answer[:100]}...")
    print("\n--- Game AI Training Complete ---")

if __name__ == "__main__":
    main()
