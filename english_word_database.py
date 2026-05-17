"""
english_word_database.py
========================
Comprehensive English word database with procedural description generation.

Creates a structured database of English words with procedurally generated
descriptions, semantic relationships, and contextual information to improve
AI coherence and understanding.

Features:
- Word categorization by part of speech, domain, and complexity
- Procedural description generation using semantic templates
- Word relationships (synonyms, antonyms, hypernyms, hyponyms)
- Contextual usage examples
- Etymology and word origin information
- Difficulty/complexity ratings
"""

import sqlite3
import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum
from pathlib import Path
import re
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

DATABASE_PATH = Path("english_words.db")


class PartOfSpeech(Enum):
    """Parts of speech for word classification."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    INTERJECTION = "interjection"


class WordComplexity(Enum):
    """Complexity levels for words."""
    BASIC = 1      # Common, everyday words
    INTERMEDIATE = 2  # Moderate vocabulary
    ADVANCED = 3   # Academic/professional
    EXPERT = 4     # Specialized/technical
    ARCHAIC = 5    # Rare/historical


class Domain(Enum):
    """Domains/categories for words."""
    GENERAL = "general"
    SCIENCE = "science"
    TECHNOLOGY = "technology"
    ARTS = "arts"
    BUSINESS = "business"
    PHILOSOPHY = "philosophy"
    MEDICINE = "medicine"
    LAW = "law"
    LITERATURE = "literature"
    MATHEMATICS = "mathematics"
    URBAN = "urban"
    MUSIC = "music"
    SPORTS = "sports"
    GAMING = "gaming"
    SOCIAL_MEDIA = "social_media"
    FOOD = "food"
    TRAVEL = "travel"
    FASHION = "fashion"
    AUTOMOTIVE = "automotive"
    POLITICS = "politics"
    RELIGION = "religion"
    NATURE = "nature"
    SPACE = "space"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class WordEntry:
    """Complete entry for a word in the database."""
    word: str
    part_of_speech: PartOfSpeech
    complexity: WordComplexity
    domain: Domain
    definition: str
    procedural_description: str = ""
    synonyms: List[str] = field(default_factory=list)
    antonyms: List[str] = field(default_factory=list)
    hypernyms: List[str] = field(default_factory=list)  # "is a type of"
    hyponyms: List[str] = field(default_factory=list)  # "types include"
    examples: List[str] = field(default_factory=list)
    etymology: str = ""
    frequency: float = 0.0  # 0.0 to 1.0
    syllable_count: int = 1
    phonetic: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['part_of_speech'] = self.part_of_speech.value
        data['complexity'] = self.complexity.value
        data['domain'] = self.domain.value
        return data


@dataclass
class SemanticRelation:
    """Represents a relationship between words."""
    word1: str
    word2: str
    relation_type: str  # synonym, antonym, hypernym, hyponym, related, etc.
    strength: float  # 0.0 to 1.0


# ── Procedural Description Generator ──────────────────────────────────────────

class ProceduralDescriptionGenerator:
    """
    Generates rich, coherent descriptions of words using procedural templates
    and semantic knowledge graphs.
    """
    
    # Description templates by part of speech
    NOUN_TEMPLATES = [
        "{word} is a {complexity} {domain} noun that refers to {definition}. "
        "It is characterized by {characteristics} and often associated with {associations}.",
        
        "As a {complexity} term in {domain}, {word} denotes {definition}. "
        "It encompasses {hyponyms} and is a type of {hypernyms}. "
        "Common contexts include {contexts}.",
        
        "{word} represents a {complexity} concept in {domain}. "
        "It is defined as {definition} and typically appears in {contexts}. "
        "Related concepts include {related_concepts}."
    ]
    
    VERB_TEMPLATES = [
        "{word} is a {complexity} {domain} verb meaning {definition}. "
        "It is typically used to describe actions involving {action_context}. "
        "Common objects of this action include {objects}.",
        
        "To {word} means to {definition}. "
        "This {complexity} {domain} action involves {action_context} "
        "and is often followed by {consequences}.",
        
        "{word} represents a {complexity} action in {domain}. "
        "It describes the process of {definition} and is commonly used "
        "in contexts involving {contexts}."
    ]
    
    ADJECTIVE_TEMPLATES = [
        "{word} is a {complexity} {domain} adjective describing {definition}. "
        "It is used to characterize nouns that are {characteristics} "
        "and often appears in {contexts}.",
        
        "Something described as {word} is {definition}. "
        "This {complexity} {domain} quality implies {implications} "
        "and is typically associated with {associations}.",
        
        "As a {complexity} descriptor in {domain}, {word} indicates {definition}. "
        "It suggests {implications} and is commonly applied to {typical_subjects}."
    ]
    
    # Semantic fillers
    CHARACTERISTICS = [
        "distinctive properties", "notable features", "defining attributes",
        "characteristic qualities", "essential aspects", "fundamental nature"
    ]
    
    ASSOCIATIONS = [
        "related concepts", "connected ideas", "associated phenomena",
        "linked elements", "corresponding notions", "parallel concepts"
    ]
    
    CONTEXTS = [
        "formal discourse", "everyday conversation", "technical writing",
        "academic contexts", "professional settings", "casual communication"
    ]
    
    ACTION_CONTEXT = [
        "intentional activity", "purposeful behavior", "deliberate action",
        "conscious effort", "planned activity", "voluntary movement"
    ]
    
    IMPLICATIONS = [
        "specific qualities", "particular characteristics", "distinct attributes",
        "notable properties", "defining features", "essential qualities"
    ]
    
    def __init__(self, word_db: 'WordDatabase'):
        self.word_db = word_db
    
    def generate_description(self, word: str) -> str:
        """
        Generate a procedural description for a word.
        
        Args:
            word: The word to describe
            
        Returns:
            Procedurally generated description
        """
        entry = self.word_db.get_word(word)
        if not entry:
            return f"No entry found for '{word}'."
        
        # Select appropriate template based on part of speech
        if entry.part_of_speech == PartOfSpeech.NOUN:
            template = random.choice(self.NOUN_TEMPLATES)
        elif entry.part_of_speech == PartOfSpeech.VERB:
            template = random.choice(self.VERB_TEMPLATES)
        elif entry.part_of_speech == PartOfSpeech.ADJECTIVE:
            template = random.choice(self.ADJECTIVE_TEMPLATES)
        else:
            template = "{word} is a {complexity} {domain} {pos} meaning {definition}."
        
        # Fill in template
        description = template.format(
            word=entry.word,
            complexity=entry.complexity.name.lower(),
            domain=entry.domain.value,
            pos=entry.part_of_speech.value,
            definition=entry.definition,
            characteristics=random.choice(self.CHARACTERISTICS),
            associations=random.choice(self.ASSOCIATIONS),
            hyponyms=self._format_list(entry.hyponyms[:3]),
            hypernyms=self._format_list(entry.hypernyms[:2]),
            contexts=random.choice(self.CONTEXTS),
            related_concepts=self._format_list(entry.synonyms[:3]),
            action_context=random.choice(self.ACTION_CONTEXT),
            objects=self._format_list(entry.hyponyms[:3]),
            consequences=random.choice(self.ASSOCIATIONS),
            implications=random.choice(self.IMPLICATIONS),
            typical_subjects=random.choice(["entities", "objects", "concepts", "phenomena"])
        )
        
        return description
    
    def _format_list(self, items: List[str]) -> str:
        """Format a list of items for natural language."""
        if not items:
            return "various elements"
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return f"{', '.join(items[:-1])}, and {items[-1]}"


# ── Word Database ─────────────────────────────────────────────────────────────

class WordDatabase:
    """
    Comprehensive database of English words with semantic relationships
    and procedural description generation.
    """
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self.conn = None
        self.description_generator = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database schema."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        
        cursor = self.conn.cursor()
        
        # Main words table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                part_of_speech TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                domain TEXT NOT NULL,
                definition TEXT NOT NULL,
                procedural_description TEXT,
                frequency REAL DEFAULT 0.0,
                syllable_count INTEGER DEFAULT 1,
                phonetic TEXT,
                etymology TEXT
            )
        """)
        
        # Word relationships table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word1 TEXT NOT NULL,
                word2 TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                UNIQUE(word1, word2, relation_type)
            )
        """)
        
        # Examples table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT NOT NULL,
                example_text TEXT NOT NULL,
                context TEXT
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_word ON words(word)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pos ON words(part_of_speech)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_complexity ON words(complexity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_domain ON words(domain)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_word1 ON relationships(word1)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_word2 ON relationships(word2)")
        
        self.conn.commit()
        
        # Initialize description generator
        self.description_generator = ProceduralDescriptionGenerator(self)
    
    def add_word(self, entry: WordEntry) -> bool:
        """
        Add a word entry to the database.
        
        Args:
            entry: WordEntry object with word information
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            
            # Insert main word entry
            cursor.execute("""
                INSERT OR REPLACE INTO words 
                (word, part_of_speech, complexity, domain, definition, 
                 procedural_description, frequency, syllable_count, phonetic, etymology)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.word,
                entry.part_of_speech.value,
                entry.complexity.value,
                entry.domain.value,
                entry.definition,
                entry.procedural_description,
                entry.frequency,
                entry.syllable_count,
                entry.phonetic,
                entry.etymology
            ))
            
            # Add relationships
            for synonym in entry.synonyms:
                self._add_relationship(entry.word, synonym, "synonym", 0.9)
            
            for antonym in entry.antonyms:
                self._add_relationship(entry.word, antonym, "antonym", 0.95)
            
            for hypernym in entry.hypernyms:
                self._add_relationship(entry.word, hypernym, "hypernym", 0.8)
            
            for hyponym in entry.hyponyms:
                self._add_relationship(entry.word, hyponym, "hyponym", 0.8)
            
            # Add examples
            for example in entry.examples:
                cursor.execute("""
                    INSERT INTO examples (word, example_text, context)
                    VALUES (?, ?, ?)
                """, (entry.word, example, "general"))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            print(f"Error adding word '{entry.word}': {e}")
            return False
    
    def _add_relationship(self, word1: str, word2: str, relation_type: str, strength: float):
        """Add a semantic relationship between two words."""
        cursor = self.conn.cursor()
        try:
            cursor.execute("""
                INSERT OR IGNORE INTO relationships (word1, word2, relation_type, strength)
                VALUES (?, ?, ?, ?)
            """, (word1, word2, relation_type, strength))
        except Exception:
            pass  # Ignore duplicate relationships
    
    def get_word(self, word: str) -> Optional[WordEntry]:
        """
        Retrieve a word entry from the database.
        
        Args:
            word: The word to retrieve
            
        Returns:
            WordEntry object or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM words WHERE word = ?", (word.lower(),))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Get relationships
        synonyms = self._get_relationships(word, "synonym")
        antonyms = self._get_relationships(word, "antonym")
        hypernyms = self._get_relationships(word, "hypernym")
        hyponyms = self._get_relationships(word, "hyponym")
        
        # Get examples
        examples = self._get_examples(word)
        
        return WordEntry(
            word=row['word'],
            part_of_speech=PartOfSpeech(row['part_of_speech']),
            complexity=WordComplexity(row['complexity']),
            domain=Domain(row['domain']),
            definition=row['definition'],
            procedural_description=row['procedural_description'] or "",
            synonyms=synonyms,
            antonyms=antonyms,
            hypernyms=hypernyms,
            hyponyms=hyponyms,
            examples=examples,
            etymology=row['etymology'] or "",
            frequency=row['frequency'],
            syllable_count=row['syllable_count'],
            phonetic=row['phonetic'] or ""
        )
    
    def _get_relationships(self, word: str, relation_type: str) -> List[str]:
        """Get all related words of a specific type."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT word2 FROM relationships WHERE word1 = ? AND relation_type = ?",
            (word.lower(), relation_type)
        )
        return [row['word2'] for row in cursor.fetchall()]
    
    def _get_examples(self, word: str) -> List[str]:
        """Get example sentences for a word."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT example_text FROM examples WHERE word = ?", (word.lower(),))
        return [row['example_text'] for row in cursor.fetchall()]
    
    def generate_description(self, word: str) -> str:
        """
        Generate a procedural description for a word.
        
        Args:
            word: The word to describe
            
        Returns:
            Generated description
        """
        entry = self.get_word(word)
        if not entry:
            return f"No entry found for '{word}'."
        
        # Generate description if not already cached
        if not entry.procedural_description:
            description = self.description_generator.generate_description(word)
            # Cache it
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE words SET procedural_description = ? WHERE word = ?",
                (description, word.lower())
            )
            self.conn.commit()
            return description
        
        return entry.procedural_description
    
    def search_words(
        self,
        part_of_speech: Optional[PartOfSpeech] = None,
        complexity: Optional[WordComplexity] = None,
        domain: Optional[Domain] = None,
        limit: int = 100
    ) -> List[WordEntry]:
        """
        Search for words with optional filters.
        
        Args:
            part_of_speech: Filter by part of speech
            complexity: Filter by complexity level
            domain: Filter by domain
            limit: Maximum number of results
            
        Returns:
            List of matching WordEntry objects
        """
        query = "SELECT * FROM words WHERE 1=1"
        params = []
        
        if part_of_speech:
            query += " AND part_of_speech = ?"
            params.append(part_of_speech.value)
        
        if complexity:
            query += " AND complexity = ?"
            params.append(complexity.value)
        
        if domain:
            query += " AND domain = ?"
            params.append(domain.value)
        
        query += f" LIMIT {limit}"
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        
        entries = []
        for row in cursor.fetchall():
            entry = self.get_word(row['word'])
            if entry:
                entries.append(entry)
        
        return entries
    
    def get_related_words(self, word: str, relation_type: str) -> List[str]:
        """
        Get words related to the given word by a specific relation type.
        
        Args:
            word: The source word
            relation_type: Type of relationship (synonym, antonym, etc.)
            
        Returns:
            List of related words
        """
        return self._get_relationships(word.lower(), relation_type)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        stats = {}
        
        cursor.execute("SELECT COUNT(*) FROM words")
        stats['total_words'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM relationships")
        stats['total_relationships'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM examples")
        stats['total_examples'] = cursor.fetchone()[0]
        
        # Count by part of speech
        cursor.execute("SELECT part_of_speech, COUNT(*) FROM words GROUP BY part_of_speech")
        stats['by_part_of_speech'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by complexity
        cursor.execute("SELECT complexity, COUNT(*) FROM words GROUP BY complexity")
        stats['by_complexity'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Count by domain
        cursor.execute("SELECT domain, COUNT(*) FROM words GROUP BY domain")
        stats['by_domain'] = {row[0]: row[1] for row in cursor.fetchall()}
        
        return stats
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()


# ── Sample Data Loader ─────────────────────────────────────────────────────────

class SampleDataLoader:
    """Loads sample word data to populate the database."""
    
    @staticmethod
    def load_common_words(db: WordDatabase):
        """Load a set of common English words with relationships."""
        
        # Nouns
        common_nouns = [
            ("time", PartOfSpeech.NOUN, WordComplexity.BASIC, Domain.GENERAL,
             "the indefinite continued progress of existence", ["moment", "duration", "period"],
             [], ["concept", "measurement"], ["second", "minute", "hour"]),
            
            ("person", PartOfSpeech.NOUN, WordComplexity.BASIC, Domain.GENERAL,
             "a human being regarded as an individual", ["individual", "human", "being"],
             [], ["entity", "organism"], ["man", "woman", "child"]),
            
            ("knowledge", PartOfSpeech.NOUN, WordComplexity.INTERMEDIATE, Domain.PHILOSOPHY,
             "facts, information, and skills acquired through experience or education",
             ["understanding", "wisdom", "insight"], ["ignorance"], ["cognition", "awareness"],
             ["expertise", "learning", "comprehension"]),
            
            ("algorithm", PartOfSpeech.NOUN, WordComplexity.ADVANCED, Domain.TECHNOLOGY,
             "a process or set of rules to be followed in calculations",
             ["procedure", "method", "process"], [], ["computing", "logic"],
             ["sorting algorithm", "search algorithm", "machine learning"]),
            
            ("consciousness", PartOfSpeech.NOUN, WordComplexity.EXPERT, Domain.PHILOSOPHY,
             "the state of being aware of and responsive to one's surroundings",
             ["awareness", "sentience", "cognition"], ["unconsciousness"],
             ["mental state", "phenomenon"], ["self-awareness", "mindfulness"]),
        ]
        
        # Verbs
        common_verbs = [
            ("think", PartOfSpeech.VERB, WordComplexity.BASIC, Domain.GENERAL,
             "to have a particular opinion, belief, or idea",
             ["believe", "consider", "reason"], ["ignore", "dismiss"],
             ["mental action", "cognitive process"], ["ponder", "reflect", "analyze"]),
            
            ("create", PartOfSpeech.VERB, WordComplexity.INTERMEDIATE, Domain.GENERAL,
             "to bring something into existence",
             ["make", "produce", "generate"], ["destroy", "eliminate"],
             ["action", "process"], ["design", "build", "construct"]),
            
            ("analyze", PartOfSpeech.VERB, WordComplexity.ADVANCED, Domain.SCIENCE,
             "to examine methodically and in detail the constitution or structure of something",
             ["examine", "investigate", "study"], ["ignore", "overlook"],
             ["cognitive process", "method"], ["evaluate", "assess", "inspect"]),
        ]
        
        # Adjectives
        common_adjectives = [
            ("good", PartOfSpeech.ADJECTIVE, WordComplexity.BASIC, Domain.GENERAL,
             "to be desired or approved of", ["excellent", "positive", "beneficial"],
             ["bad", "poor"], ["quality", "attribute"], ["great", "fine", "wonderful"]),
            
            ("complex", PartOfSpeech.ADJECTIVE, WordComplexity.INTERMEDIATE, Domain.GENERAL,
             "consisting of many different and connected parts",
             ["complicated", "intricate", "involved"], ["simple", "basic"],
             ["characteristic", "quality"], ["elaborate", "sophisticated"]),
            
            ("ephemeral", PartOfSpeech.ADJECTIVE, WordComplexity.EXPERT, Domain.LITERATURE,
             "lasting for a very short time", ["transient", "fleeting", "brief"],
             ["permanent", "eternal"], ["temporal quality", "characteristic"],
             ["momentary", "short-lived", "temporary"]),
        ]
        
        # Add all words
        for word_data in common_nouns + common_verbs + common_adjectives:
            word, pos, complexity, domain, definition, synonyms, antonyms, hypernyms, hyponyms = word_data
            
            entry = WordEntry(
                word=word,
                part_of_speech=pos,
                complexity=complexity,
                domain=domain,
                definition=definition,
                synonyms=synonyms,
                antonyms=antonyms,
                hypernyms=hypernyms,
                hyponyms=hyponyms,
                examples=[f"The {word} is essential."],
                frequency=0.5,
                syllable_count=len(word.split('-'))
            )
            
            db.add_word(entry)


# ── Main Interface ─────────────────────────────────────────────────────────────

def main():
    """Main function to demonstrate the word database."""
    print("English Word Database with Procedural Description Generation")
    print("=" * 60)
    
    # Initialize database
    db = WordDatabase()
    
    # Load sample data
    print("\nLoading sample words...")
    SampleDataLoader.load_common_words(db)
    
    # Show statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total words: {stats['total_words']}")
    print(f"  Total relationships: {stats['total_relationships']}")
    print(f"  Total examples: {stats['total_examples']}")
    print(f"  By part of speech: {stats['by_part_of_speech']}")
    print(f"  By complexity: {stats['by_complexity']}")
    print(f"  By domain: {stats['by_domain']}")
    
    # Generate procedural descriptions
    print("\n" + "=" * 60)
    print("Procedural Descriptions:")
    print("=" * 60)
    
    test_words = ["time", "knowledge", "algorithm", "consciousness", "ephemeral"]
    
    for word in test_words:
        print(f"\n{word.upper()}:")
        description = db.generate_description(word)
        print(f"  {description}")
        
        # Show related words
        synonyms = db.get_related_words(word, "synonym")
        if synonyms:
            print(f"  Synonyms: {', '.join(synonyms)}")
    
    # Search examples
    print("\n" + "=" * 60)
    print("Search Examples:")
    print("=" * 60)
    
    # Search for advanced technology words
    tech_words = db.search_words(
        part_of_speech=PartOfSpeech.NOUN,
        domain=Domain.TECHNOLOGY,
        complexity=WordComplexity.ADVANCED
    )
    print(f"\nAdvanced technology nouns: {[w.word for w in tech_words]}")
    
    # Search for basic verbs
    basic_verbs = db.search_words(
        part_of_speech=PartOfSpeech.VERB,
        complexity=WordComplexity.BASIC
    )
    print(f"Basic verbs: {[w.word for w in basic_verbs]}")
    
    # Close database
    db.close()
    
    print("\n" + "=" * 60)
    print("Database demonstration complete!")


if __name__ == "__main__":
    main()
