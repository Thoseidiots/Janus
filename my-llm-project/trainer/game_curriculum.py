import os
import random
import glob

class GameAIDataset:
    """Dataset class for Game AI training data from the training_data folder."""
    def __init__(self, data_dir="training_data"):
        self.data_dir = data_dir
        self.files = glob.glob(os.path.join(self.data_dir, "*.md"))
        self.content = []
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.content.append((os.path.basename(file_path), f.read()))

    def generate_dataset(self, samples=10):
        dataset = []
        if not self.content:
            return [("No game training data found.", "Please check the training_data directory.")]
            
        for _ in range(samples):
            filename, text = random.choice(self.content)
            # Create a prompt based on the filename or a snippet
            topic = filename.replace('.md', '').replace('_', ' ').title()
            # Simple prompt-answer pair for fine-tuning
            prompt = f"Explain the key concepts of {topic} in AAA game development."
            # Use a portion of the text as the answer (in a real scenario, this would be more sophisticated)
            answer = text[:2000] + "..." if len(text) > 2000 else text
            dataset.append((prompt, answer))
        return dataset

class GameAIQuestDataset:
    """Dataset class for generating game quests for training."""
    def __init__(self):
        # Simplified version of the QuestGenerator logic
        self.themes = ['heroic', 'mysterious', 'personal', 'political', 'economic', 'exploration']
        self.objectives = ['kill', 'collect', 'deliver', 'escort', 'investigate', 'craft']
        self.rewards = ['gold', 'experience', 'item', 'reputation', 'title']

    def generate_dataset(self, samples=10):
        dataset = []
        for _ in range(samples):
            theme = random.choice(self.themes)
            obj = random.choice(self.objectives)
            reward = random.choice(self.rewards)
            prompt = f"Generate a {theme} quest for a AAA game with a '{obj}' objective."
            answer = f"Quest Title: The {theme.capitalize()} {obj.capitalize()}\nObjective: {obj.capitalize()} the target.\nReward: {reward.capitalize()}."
            dataset.append((prompt, answer))
        return dataset
