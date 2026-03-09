import random
import string

class GradePreK:
    """Grade Pre-K: Simple spatial reasoning, repetition, state change, and rhymes."""
    def __init__(self):
        self.items = ["keys", "phone", "wallet", "toy", "apple"]
        self.containers_small = ["blue box", "small bag", "wooden crate"]
        self.containers_large = ["red suitcase", "backpack", "cardboard box"]
        self.locations = ["car", "park", "kitchen", "garage"]
        self.animals = ["cat", "dog", "bird", "bunny"]
        self.animal_states = ["playing", "awake", "running"]
        self.teacher_phrases = ["the ball is red", "I see a big dog", "let's count to three", "the sky is blue"]
        self.rhymes = {
            "Twinkle, twinkle, little": "star",
            "Baa, baa, black sheep, have you any": "wool",
            "The wheels on the bus go": "round and round",
            "Row, row, row your": "boat",
        }
        self.task_generators = [
            self._generate_spatial_task,
            self._generate_repetition_task,
            self._generate_state_change_task,
            self._generate_rhyme_task,
            self._generate_counting_task,
            self._generate_alphabet_repetition_task,
        ]

    def _generate_spatial_task(self):
        item = random.choice(self.items)
        c1 = random.choice(self.containers_small)
        c2 = random.choice(self.containers_large)
        loc = random.choice(self.locations)
        prompt = f"I put the {item} in the {c1}. I put the {c1} inside the {c2}. I move the {c2} to the {loc}. Where is the {item}?"
        answer = f"The {item} is in the {c1}, which is inside the {c2} in the {loc}."
        return prompt, answer

    def _generate_repetition_task(self):
        phrase = random.choice(self.teacher_phrases)
        prompt = f"The teacher says '{phrase}'. You repeat. What do you say?"
        answer = f"{phrase}"
        return prompt, answer

    def _generate_state_change_task(self):
        animal = random.choice(self.animals)
        state = random.choice(self.animal_states)
        prompt = f"The {animal} is {state}. The {animal} takes a nap. Is the {animal} {state}?"
        answer = "No."
        return prompt, answer

    def _generate_rhyme_task(self):
        start, end = random.choice(list(self.rhymes.items()))
        prompt = f"{start} ____."
        answer = f"{end}"
        return prompt, answer

    def _generate_counting_task(self):
        count_to = random.randint(3, 10)
        prompt = f"Let's count to {count_to}."
        answer = " ".join(map(str, range(1, count_to + 1)))
        return prompt, answer

    def _generate_alphabet_repetition_task(self):
        alphabet = string.ascii_uppercase
        start = random.randint(0, len(alphabet) - 4)
        end = start + random.randint(3, 5)
        segment = " ".join(alphabet[start:end])
        prompt = f"The teacher says '{segment}'. You repeat. What do you say?"
        answer = segment
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade1:
    """Grade 1: Basic state change by moving an object."""
    def __init__(self):
        self.items = ["marble", "coin", "block", "bead", "stone"]
        self.containers = ["green box", "yellow bag", "blue bowl", "purple cup"]

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            item = random.choice(self.items)
            c1, c2 = random.sample(self.containers, 2)
            prompt = f"The {item} is in the {c1}. I move the {item} from the {c1} to the {c2}. Where is the {item}?"
            answer = f"The {item} is in the {c2}."
            dataset.append((prompt, answer))
        return dataset

class Grade2:
    """Grade 2: Simple arithmetic (addition and subtraction)."""
    def __init__(self):
        self.max_number = 10
        self.addition_phrases = [
            "What is {n1} plus {n2}?",
            "What is {n1} + {n2}?",
            "If you have {n1} apples and get {n2} more, how many apples do you have?",
        ]
        self.subtraction_phrases = [
            "What is {n1} minus {n2}?",
            "What is {n1} - {n2}?",
            "If you have {n1} cookies and you eat {n2}, how many are left?",
        ]
        self.task_generators = [
            self._generate_addition_task,
            self._generate_subtraction_task,
        ]

    def _generate_addition_task(self):
        n1 = random.randint(1, self.max_number)
        n2 = random.randint(1, self.max_number)
        prompt = random.choice(self.addition_phrases).format(n1=n1, n2=n2)
        answer = str(n1 + n2)
        return prompt, answer

    def _generate_subtraction_task(self):
        n1 = random.randint(1, self.max_number)
        n2 = random.randint(1, n1)  # Ensure the result is not negative
        prompt = random.choice(self.subtraction_phrases).format(n1=n1, n2=n2)
        answer = str(n1 - n2)
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade3:
    """Grade 3: State tracking combined with arithmetic."""
    def __init__(self):
        self.items = ["crayons", "pencils", "stickers", "cookies", "marbles"]
        self.containers = ["box", "jar", "bag", "drawer"]
        self.max_initial_items = 15
        self.max_change = 10
        self.task_generators = [
            self._generate_addition_task,
            self._generate_subtraction_task,
        ]

    def _generate_addition_task(self):
        item = random.choice(self.items)
        container = random.choice(self.containers)
        n1 = random.randint(1, self.max_initial_items)
        n2 = random.randint(1, self.max_change)
        prompt = f"The {container} has {n1} {item}. You put {n2} more {item} in the {container}. How many {item} are in the {container} now?"
        answer = str(n1 + n2)
        return prompt, answer

    def _generate_subtraction_task(self):
        item = random.choice(self.items)
        container = random.choice(self.containers)
        n1 = random.randint(1, self.max_initial_items)
        n2 = random.randint(1, n1) # Ensure we don't go negative
        prompt = f"There are {n1} {item} in the {container}. You take out {n2} {item}. How many {item} are left in the {container}?"
        answer = str(n1 - n2)
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade4:
    """Grade 4: Simple list and sequence manipulation."""
    def __init__(self):
        self.colors = ["red", "green", "blue", "yellow", "purple", "orange"]
        self.animals = ["cat", "dog", "fish", "bird", "horse", "snake"]
        self.numbers = list(range(1, 21))
        self.task_generators = [
            self._generate_find_by_index_task,
            self._generate_reverse_list_task,
            self._generate_list_length_task,
            self._generate_append_item_task,
        ]

    def _generate_find_by_index_task(self):
        items = random.sample(self.colors, random.randint(3, 5))
        index = random.randint(0, len(items) - 1)
        index_words = ["first", "second", "third", "fourth", "last"]
        index_word = index_words[index] if index < 4 else index_words[-1]
        if index_word == "last":
            index = len(items) - 1

        list_str = ", ".join(items)
        prompt = f"Given the list of colors: {list_str}. What is the {index_word} color?"
        answer = items[index]
        return prompt, answer

    def _generate_reverse_list_task(self):
        items = [str(n) for n in random.sample(self.numbers, random.randint(3, 5))]
        list_str = ", ".join(items)
        reversed_list_str = ", ".join(reversed(items))
        prompt = f"Reverse this list: {list_str}."
        answer = reversed_list_str
        return prompt, answer

    def _generate_list_length_task(self):
        items = random.sample(self.animals, random.randint(2, 6))
        prompt = f"How many animals are in this list: {', '.join(items)}?"
        answer = str(len(items))
        return prompt, answer

    def _generate_append_item_task(self):
        items = random.sample(self.colors, random.randint(2, 4))
        new_item = random.choice(self.colors)
        prompt = f"If you add '{new_item}' to the end of the list [{', '.join(items)}], what is the new list?"
        answer = f"[{', '.join(items + [new_item])}]"
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade5:
    """Grade 5: Social awareness, etiquette, and rule-following."""
    def __init__(self,):
        # Vocabulary for dynamic templates
        self.people = ["friend", "classmate", "teacher", "brother", "sister"]
        self.actions_negative = ["falls down", "is crying", "is sad", "dropped their books"]
        self.actions_positive = ["won an award", "got a new puppy", "is happy"]
        self.locations = ["the library", "the classroom", "the museum", "the hallway"]
        self.rules_forbidden = {"running": "run", "shouting": "shout", "eating": "eat"}
        self.items_polite = ["cookie", "turn on the swing", "pencil"]

        # Task generators
        self.task_generators = [
            self._generate_politeness_task,
            self._generate_rule_following_task,
            self._generate_social_awareness_task,
        ]

    def _generate_politeness_task(self):
        templates = [
            ("Someone gives you a gift. What do you say?", "Thank you."),
            ("You accidentally bump into someone. What do you say?", "Excuse me."),
            (f"You want to ask for a {random.choice(self.items_polite)}. How do you ask politely?", f"May I please have a {self.items_polite[-1]}?"),
            ("Someone says 'Thank you'. What can you say back?", "You're welcome.")
        ]
        return random.choice(templates)

    def _generate_rule_following_task(self):
        forbidden_action_noun, forbidden_action_verb = random.choice(list(self.rules_forbidden.items()))
        location = random.choice(self.locations)
        prompt = f"The rule in {location} is 'No {forbidden_action_noun}'. You are in {location}. Should you {forbidden_action_verb}?"
        answer = "No."
        return prompt, answer

    def _generate_social_awareness_task(self):
        person = random.choice(self.people)
        if random.choice([True, False]):
            action = random.choice(self.actions_negative)
            prompt = f"Your {person} {action}. What is a good thing to do?"
            answer = "Ask if they are okay."
        else:
            action = random.choice(self.actions_positive)
            prompt = f"Your {person} {action}. What is a nice thing to say?"
            answer = "Congratulations!"
        return prompt, answer

    def generate_dataset(self, samples=1000):
        """Generates a dataset of social reasoning questions."""
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade6:
    """Grade 6: Simple conditional logic."""
    def __init__(self):
        self.items = ["report card", "textbook", "pencil case", "lunchbox"]
        self.containers = ["locker", "desk", "bookshelf"]

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            item = random.choice(self.items)
            c1, c2 = random.sample(self.containers, 2)
            condition = random.choice([True, False])
            if condition:
                prompt = f"The {item} was in the {c1}. If the bell has rung, I move the {item} to the {c2}. The bell has rung. Where is the {item}?"
                answer = f"The {item} is in the {c2}."
            else:
                prompt = f"The {item} was in the {c1}. If the bell has rung, I move the {item} to the {c2}. The bell has not rung. Where is the {item}?"
                answer = f"The {item} is in the {c1}."
            dataset.append((prompt, answer))
        return dataset

class Grade7:
    """Grade 7: Conditional Arithmetic."""
    def __init__(self):
        self.items = ["points", "gold coins", "tickets", "tokens"]
        self.conditions = ["it is a sunny day", "you are wearing a red shirt", "it is Tuesday"]
        self.max_initial = 20
        self.max_change = 10

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            item = random.choice(self.items)
            condition_text = random.choice(self.conditions)
            n1 = random.randint(5, self.max_initial)
            n2 = random.randint(1, self.max_change)
            
            # Decide if the condition is met
            condition_met = random.choice([True, False])

            if condition_met:
                prompt = f"You have {n1} {item}. If {condition_text}, you get {n2} more. It is true that {condition_text}. How many {item} do you have?"
                answer = str(n1 + n2)
            else:
                prompt = f"You have {n1} {item}. If {condition_text}, you get {n2} more. It is not true that {condition_text}. How many {item} do you have?"
                answer = str(n1)
            dataset.append((prompt, answer))
        return dataset

class Grade8:
    """Grade 8: Simple temporal reasoning with one object and two steps."""
    def __init__(self):
        self.items = ["ball", "key", "remote", "book"]
        self.locations = ["box", "drawer", "shelf", "table", "bag", "cupboard"]

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            item = random.choice(self.items)
            # Sample three unique locations
            loc1, loc2, loc3 = random.sample(self.locations, 3)

            # Define the sequence of actions
            initial_state = f"The {item} is in the {loc1}."
            action1 = f"I move the {item} from the {loc1} to the {loc2}."
            action2 = f"I move the {item} from the {loc2} to the {loc3}."

            # Randomly decide which location to ask about
            if random.choice([True, False]):
                prompt = f"{initial_state} First, {action1}. Then, {action2}. Where is the {item} now?"
                answer = f"The {item} is in the {loc3}."
            else:
                # A slightly harder variation asking about an intermediate state
                prompt = f"{initial_state} First, {action1}. After the first step, where is the {item}?"
                answer = f"The {item} is in the {loc2}."
            dataset.append((prompt, answer))
        return dataset

class Grade9:
    """Grade 9: Temporal and stateful reasoning with two objects and two actions."""
    def __init__(self):
        self.items = ["notebook", "calculator", "glasses", "pen"]
        self.containers = ["backpack", "drawer", "locker", "binder"]

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            item1, item2 = random.sample(self.items, 2)
            c1, c2, c3 = random.sample(self.containers, 3)
            initial_state = f"The {item1} is in the {c1}, and the {item2} is in the {c2}."
            action_move = f"I move the {item1} from the {c1} to the {c3}"
            action_swap = f"I swap the contents of the {c1} and the {c2}"

            if random.choice([True, False]):
                prompt = f"{initial_state} First, {action_move}. Then, {action_swap}. Where is the {item2}?"
                answer = f"The {item2} is in the {c1}."
            else:
                prompt = f"{initial_state} First, {action_swap}. Then, {action_move}. Where is the {item2}?"
                answer = f"The {item2} is in the {c1}."
            dataset.append((prompt, answer))
        return dataset

class Grade10:
    """Grade 10: Logical deduction and transitivity."""
    def __init__(self):
        self.entities = ['A', 'B', 'C', 'D', 'E', 'F']
        self.relations = [
            "is taller than",
            "is faster than",
            "is heavier than",
            "is older than",
            "is stronger than",
        ]

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            # Sample 3 unique entities for the chain
            e1, e2, e3 = random.sample(self.entities, 3)
            relation = random.choice(self.relations)

            # Create the premises for a transitive relation (e1 > e2 > e3)
            premise1 = f"{e1} {relation} {e2}"
            premise2 = f"{e2} {relation} {e3}"

            # Randomly generate a "Yes" or "No" question
            if random.choice([True, False]):
                # Ask a question where the answer is "Yes"
                prompt = f"If {premise1} and {premise2}, is it true that {e1} {relation} {e3}?"
                answer = "Yes."
            else:
                # Ask a question where the answer is "No"
                prompt = f"If {premise1} and {premise2}, is it true that {e3} {relation} {e1}?"
                answer = "No."
            dataset.append((prompt, answer))
        return dataset

class Grade11:
    """Grade 11: Common-sense reasoning about the physical world."""
    def __init__(self):
        self.gravity_objects = ["ball", "rock", "apple", "book"]
        self.buoyant_objects = ["beach ball", "empty bottle", "wood log"]
        self.heavy_objects = ["rock", "bowling ball", "anvil"]
        self.light_objects = ["feather", "leaf", "balloon"]
        self.breakable_objects = ["glass cup", "egg", "plate"]
        self.task_generators = [
            self._generate_gravity_task,
            self._generate_buoyancy_task,
            self._generate_relative_weight_task,
            self._generate_fragility_task,
        ]

    def _generate_gravity_task(self):
        item = random.choice(self.gravity_objects)
        prompt = f"If you are holding a {item} and let go, will it fall to the ground or float to the sky?"
        answer = "Fall to the ground."
        return prompt, answer

    def _generate_buoyancy_task(self):
        item = random.choice(self.buoyant_objects)
        prompt = f"If you place a {item} in water, will it sink or float?"
        answer = "Float."
        return prompt, answer

    def _generate_relative_weight_task(self):
        heavy_item = random.choice(self.heavy_objects)
        light_item = random.choice(self.light_objects)
        prompt = f"Which is heavier, a {heavy_item} or a {light_item}?"
        answer = f"The {heavy_item}."
        return prompt, answer

    def _generate_fragility_task(self):
        item = random.choice(self.breakable_objects)
        prompt = f"If you drop a {item} from a high shelf onto a hard floor, is it likely to break?"
        answer = "Yes."
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class Grade12:
    """Grade 12: Multi-step logical puzzles (riddles)."""
    def __init__(self):
        self.names = ["Alex", "Ben", "Chris"]
        self.attributes = {
            "pet": ["cat", "dog", "fish"],
            "color": ["red", "blue", "green"],
            "city": ["Paris", "Tokyo", "Cairo"],
        }

    def generate_dataset(self, samples=1000):
        dataset = []
        for _ in range(samples):
            # Choose an attribute type for the puzzle
            attr_type, attr_list = random.choice(list(self.attributes.items()))
            
            # Shuffle names and attributes to create a random ground truth
            random.shuffle(self.names)
            random.shuffle(attr_list)
            solution = dict(zip(self.names, attr_list))
            
            # Generate clues based on the solution
            # Clue 1: A direct positive link for the first person
            clue1 = f"{self.names[0]}'s {attr_type} is the {solution[self.names[0]]}."
            
            # Clue 2: A direct negative link for the second person
            clue2 = f"{self.names[1]}'s {attr_type} is not the {solution[self.names[0]]}."

            # The question is about the third person
            question_person = self.names[2]
            
            prompt = f"Consider these three people: {', '.join(self.names)}. Here are the clues: 1. {clue1} 2. {clue2}. Based on these clues, what is {question_person}'s {attr_type}?"
            answer = f"The {solution[question_person]}."
            dataset.append((prompt, answer))
        return dataset

class ExtracurricularPhysics101:
    """Extracurricular: Basic common-sense physics and properties of matter."""
    def __init__(self):
        self.gravity_scenarios = {
            "If you drop a ball, will it go up or down?": "Down.",
            "An apple falls from a tree. Where does it land?": "On the ground.",
        }
        self.material_properties = [
            (("a feather", "a rock"), "a rock"),
            (("a balloon filled with air", "a bowling ball"), "a bowling ball"),
        ]
        self.states_of_matter = {
            "If you heat up water, does it become ice or steam?": "Steam.",
            "If you freeze juice, does it become a liquid or a solid?": "A solid.",
        }
        self.task_generators = [
            lambda: random.choice(list(self.gravity_scenarios.items())),
            self._generate_material_properties_task,
            lambda: random.choice(list(self.states_of_matter.items())),
        ]

    def _generate_material_properties_task(self):
        (item1, item2), heavier_item = random.choice(self.material_properties)
        prompt = f"Which is heavier, {item1} or {item2}?"
        answer = heavier_item
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]

class ExtracurricularCreativeWriting101:
    """Extracurricular: Creative writing and storytelling."""
    def __init__(self):
        self.characters = ["a brave knight", "a curious astronaut", "a friendly dragon", "a clever detective"]
        self.settings = ["a mysterious forest", "a bustling futuristic city", "a quiet, forgotten library"]
        self.story_openers = [
            "Once upon a time, in a land filled with magic, there was...",
            "The old spaceship creaked as it landed on the alien planet. The first thing the captain saw was...",
            "It was a dark and stormy night when the detective heard a knock on the door. It was...",
        ]
        self.task_generators = [
            self._generate_story_starter_task,
        ]

    def _generate_story_starter_task(self):
        """Generates a prompt to start a story."""
        task_type = random.choice(["character", "setting", "opener"])
        if task_type == "character":
            prompt = f"Write a short story about {random.choice(self.characters)}."
        elif task_type == "setting":
            prompt = f"Write a short story that takes place in {random.choice(self.settings)}."
        else: # opener
            prompt = f"Continue this story: '{random.choice(self.story_openers)}'"
        answer = "(An open-ended creative response is expected here.)"
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [self._generate_story_starter_task() for _ in range(samples)]

class ExtracurricularMusicTheory101:
    """Extracurricular: Basic music theory concepts."""
    def __init__(self):
        self.notes = ["C", "D", "E", "F", "G", "A", "B"]
        self.rhythms = {
            "whole note": "4",
            "half note": "2",
            "quarter note": "1",
        }
        self.dynamics = {
            "piano": "softly",
            "forte": "loudly",
            "crescendo": "to gradually get louder",
        }
        self.task_generators = [
            self._generate_note_sequence_task,
            self._generate_rhythm_task,
            self._generate_dynamics_task,
        ]

    def _generate_note_sequence_task(self):
        index = random.randint(0, len(self.notes) - 2)
        prompt = f"In a C major scale, what note comes after {self.notes[index]}?"
        answer = self.notes[index + 1]
        return prompt, answer

    def _generate_rhythm_task(self):
        note_type, beats = random.choice(list(self.rhythms.items()))
        prompt = f"In 4/4 time, how many beats is a {note_type} worth?"
        answer = beats
        return prompt, answer

    def _generate_dynamics_task(self):
        term, meaning = random.choice(list(self.dynamics.items()))
        prompt = f"In music, what does the term '{term}' mean?"
        answer = f"To play {meaning}."
        return prompt, answer

    def generate_dataset(self, samples=1000):
        return [random.choice(self.task_generators)() for _ in range(samples)]
