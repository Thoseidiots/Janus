import random

def generate_pre_k_dataset(samples=1000):
    """Generates a dataset for the Pre-K curriculum stage."""
    dataset = []
    items = ["keys", "phone", "wallet", "toy", "apple"]
    containers_small = ["blue box", "small bag", "wooden crate"]
    containers_large = ["red suitcase", "backpack", "cardboard box"]
    locations = ["car", "park", "kitchen", "garage"]
    for _ in range(samples):
        item, c1, c2, loc = random.choice(items), random.choice(containers_small), random.choice(containers_large), random.choice(locations)
        prompt = f"I put the {item} in the {c1}. I put the {c1} inside the {c2}. I move the {c2} to the {loc}. Where is the {item}?"
        answer = f"The {item} is in the {c1}, which is inside the {c2} in the {loc}."
        dataset.append((prompt, answer))
    return dataset

def generate_elementary_school_dataset(samples=1000):
    """Generates a dataset for the Elementary School curriculum stage."""
    dataset = []
    items = ["marble", "coin", "block", "bead", "stone"]
    containers = ["green box", "yellow bag", "blue bowl", "purple cup"]
    locations = ["on the table", "on the floor", "on the shelf"]
    for _ in range(samples):
        item = random.choice(items)
        c1, c2 = random.sample(containers, 2)
        loc1, loc2 = random.sample(locations, 2)

        prompt = f"The {item} is in the {c1}. The {c1} is {loc1}. I move the {item} from the {c1} to the {c2}. The {c2} is {loc2}. Where is the {item}?"
        answer = f"The {item} is in the {c2}."
        dataset.append((prompt, answer))
    return dataset
