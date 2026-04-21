"""
janus_vocabulary_db.py
----------------------
Generates vocabulary and word knowledge training data for the Janus NLP pipeline.
Produces a SQLite .db file and a .jsonl file when run directly.

Categories:
  - word_definition
  - synonym_antonym
  - word_in_context
  - vocabulary_building
  - collocations
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "janus_vocabulary.db")
JSONL_PATH = os.path.join(os.path.dirname(__file__), "janus_vocabulary.jsonl")

random.seed(22)

# ---------------------------------------------------------------------------
# Word pools
# ---------------------------------------------------------------------------

WORDS_COMMON = [
    ("happy", "feeling or showing pleasure or contentment",
     "Old English hæp (luck, chance)", "She was happy to hear the good news."),
    ("run", "move at a speed faster than a walk",
     "Old English rinnan (to flow)", "He runs five miles every morning."),
    ("big", "of considerable size or extent",
     "Middle English, origin uncertain", "They live in a big house."),
    ("make", "form something by putting parts together",
     "Old English macian", "She will make a cake for the party."),
    ("go", "move from one place to another",
     "Old English gan", "We go to school by bus."),
    ("good", "to be desired or approved of",
     "Old English god", "This is a good opportunity."),
    ("new", "not existing before; made or introduced recently",
     "Old English niwe", "They bought a new car."),
    ("time", "the indefinite continued progress of existence",
     "Old English tima", "Time flies when you are having fun."),
    ("work", "activity involving mental or physical effort",
     "Old English weorc", "She works at a hospital."),
    ("know", "be aware of through observation or inquiry",
     "Old English cnawan", "Do you know the answer?"),
]

WORDS_INTERMEDIATE = [
    ("ambiguous", "open to more than one interpretation",
     "Latin ambiguus (doubtful)", "The contract contained several ambiguous clauses."),
    ("benevolent", "well meaning and kindly",
     "Latin bene (well) + volens (wishing)", "The benevolent donor funded the new library."),
    ("concise", "giving a lot of information clearly in few words",
     "Latin concisus (cut up)", "Please write a concise summary of the report."),
    ("diligent", "having or showing care in one's work",
     "Latin diligens (careful)", "She is a diligent student who never misses a deadline."),
    ("eloquent", "fluent or persuasive in speaking or writing",
     "Latin eloquens (speaking out)", "His eloquent speech moved the audience."),
    ("frugal", "sparing or economical with money or food",
     "Latin frugalis (virtuous)", "Living a frugal lifestyle helped him save money."),
    ("gregarious", "fond of company; sociable",
     "Latin gregarius (of a flock)", "She is gregarious and makes friends easily."),
    ("hinder", "make it difficult for someone to do something",
     "Old English hindrian", "Bad weather hindered the rescue operation."),
    ("imminent", "about to happen",
     "Latin imminere (to overhang)", "The storm was imminent, so they stayed indoors."),
    ("jeopardise", "put someone or something into a situation of risk",
     "Old French jeu parti (divided game)", "His reckless behaviour jeopardised the project."),
    ("keen", "having or showing eagerness or enthusiasm",
     "Old English cene (brave)", "She is keen to learn new skills."),
    ("lucid", "expressed clearly; easy to understand",
     "Latin lucidus (bright)", "The professor gave a lucid explanation of the theory."),
    ("meticulous", "showing great attention to detail",
     "Latin meticulosus (fearful)", "He is meticulous in his research."),
    ("novel", "new or unusual in an interesting way",
     "Latin novellus (new)", "The scientist proposed a novel approach to the problem."),
    ("obsolete", "no longer produced or used",
     "Latin obsoletus (grown old)", "Fax machines are becoming obsolete."),
]

WORDS_ADVANCED = [
    ("abstruse", "difficult to understand; obscure",
     "Latin abstrusus (concealed)", "The philosopher's abstruse arguments confused many readers."),
    ("acrimony", "bitterness or ill feeling",
     "Latin acrimonia (sharpness)", "The divorce proceedings were marked by acrimony."),
    ("bellicose", "demonstrating aggression and willingness to fight",
     "Latin bellicosus (warlike)", "The bellicose rhetoric alarmed neighbouring countries."),
    ("capricious", "given to sudden changes of mood or behaviour",
     "Italian capriccio (sudden start)", "The capricious weather ruined their outdoor plans."),
    ("deleterious", "causing harm or damage",
     "Greek deleterios (destructive)", "Smoking has deleterious effects on health."),
    ("ephemeral", "lasting for a very short time",
     "Greek ephemeros (lasting a day)", "Fame can be ephemeral in the age of social media."),
    ("fastidious", "very attentive to accuracy and detail",
     "Latin fastidiosus (disdainful)", "She is fastidious about the quality of her work."),
    ("garrulous", "excessively talkative",
     "Latin garrulus (chattering)", "The garrulous neighbour kept them talking for hours."),
    ("hegemony", "leadership or dominance of one country over others",
     "Greek hegemonia (leadership)", "The empire sought hegemony over the entire region."),
    ("iconoclast", "a person who attacks cherished beliefs",
     "Greek eikonoklastes (image breaker)", "The iconoclast challenged every established norm."),
]

WORDS_ACADEMIC = [
    ("paradigm", "a typical example or pattern of something",
     "Greek paradeigma (pattern)", "The discovery shifted the scientific paradigm."),
    ("empirical", "based on observation or experience rather than theory",
     "Greek empeirikos (experienced)", "The study relied on empirical evidence."),
    ("hypothesis", "a proposed explanation made on limited evidence",
     "Greek hypothesis (foundation)", "The researcher tested her hypothesis with experiments."),
    ("methodology", "a system of methods used in a field",
     "Greek methodos (pursuit of knowledge)", "The paper described the research methodology in detail."),
    ("synthesis", "the combination of ideas to form a theory",
     "Greek synthesis (composition)", "The essay required a synthesis of multiple sources."),
    ("discourse", "written or spoken communication or debate",
     "Latin discursus (running about)", "Academic discourse requires precise language."),
    ("pedagogy", "the method and practice of teaching",
     "Greek paidagogia (leading children)", "Modern pedagogy emphasises active learning."),
    ("epistemology", "the branch of philosophy concerned with knowledge",
     "Greek episteme (knowledge)", "Epistemology asks how we can know what we know."),
    ("ontology", "the branch of philosophy dealing with existence",
     "Greek ontos (being)", "Ontology explores the nature of reality."),
    ("heuristic", "enabling discovery through practical methods",
     "Greek heuriskein (to find)", "The teacher used a heuristic approach to problem-solving."),
]

ALL_WORDS = WORDS_COMMON + WORDS_INTERMEDIATE + WORDS_ADVANCED + WORDS_ACADEMIC

SYNONYM_ANTONYM_DATA = [
    ("happy", "joyful, content, pleased", "sad, miserable, unhappy",
     "'Joyful' implies active expression; 'content' suggests quiet satisfaction; 'pleased' is milder."),
    ("big", "large, enormous, vast", "small, tiny, miniature",
     "'Enormous' and 'vast' suggest extreme size; 'large' is more neutral."),
    ("fast", "quick, rapid, swift", "slow, sluggish, leisurely",
     "'Swift' often implies elegance; 'rapid' suggests urgency; 'quick' is most general."),
    ("smart", "intelligent, clever, astute", "foolish, ignorant, obtuse",
     "'Astute' implies shrewd judgment; 'clever' suggests resourcefulness; 'intelligent' is most neutral."),
    ("brave", "courageous, valiant, bold", "cowardly, timid, fearful",
     "'Valiant' has a heroic connotation; 'bold' can also mean daring in a non-dangerous sense."),
    ("old", "ancient, aged, elderly", "young, new, modern",
     "'Ancient' refers to very old things; 'elderly' is respectful for people; 'aged' is more neutral."),
    ("difficult", "challenging, arduous, complex", "easy, simple, straightforward",
     "'Arduous' implies physical effort; 'complex' implies many parts; 'challenging' is most neutral."),
    ("beautiful", "gorgeous, stunning, attractive", "ugly, hideous, plain",
     "'Gorgeous' is more intense than 'beautiful'; 'stunning' implies a striking effect."),
    ("angry", "furious, irate, irritated", "calm, peaceful, serene",
     "'Furious' is more intense than 'angry'; 'irate' is formal; 'irritated' is milder."),
    ("important", "significant, crucial, vital", "trivial, insignificant, minor",
     "'Crucial' and 'vital' imply necessity; 'significant' implies notable impact."),
    ("begin", "start, commence, initiate", "end, finish, conclude",
     "'Commence' is formal; 'initiate' implies being the first to act; 'start' is most common."),
    ("help", "assist, aid, support", "hinder, obstruct, impede",
     "'Aid' often implies emergency help; 'assist' is more general; 'support' implies ongoing help."),
    ("show", "display, demonstrate, exhibit", "hide, conceal, obscure",
     "'Exhibit' implies a formal display; 'demonstrate' implies showing how something works."),
    ("think", "consider, ponder, contemplate", "ignore, disregard, overlook",
     "'Ponder' implies deep thought; 'contemplate' suggests quiet reflection; 'consider' is most neutral."),
    ("walk", "stroll, march, stride", "run, sprint, dash",
     "'Stroll' implies a leisurely pace; 'march' implies purpose; 'stride' implies long steps."),
]

WORD_IN_CONTEXT_DATA = [
    ("bank", "She walked to the [bank] to deposit her cheque.",
     "In this context, 'bank' means a financial institution. It does not mean the side of a river, as there is no mention of water or geography."),
    ("light", "The [light] from the window woke him up.",
     "Here 'light' refers to natural illumination from the sun. It does not mean 'not heavy' (adjective) or 'to ignite' (verb)."),
    ("spring", "The flowers bloom every [spring].",
     "'Spring' here refers to the season between winter and summer. It does not mean a coiled device or to jump."),
    ("fair", "The judge was known for being [fair] in all her rulings.",
     "'Fair' here means impartial and just. It does not refer to a funfair or to light-coloured hair."),
    ("match", "He struck a [match] to light the candle.",
     "'Match' here is a small stick used to produce fire. It does not mean a sports contest or to correspond."),
    ("bark", "The dog's [bark] echoed through the empty street.",
     "'Bark' here is the sound a dog makes. It does not refer to the outer covering of a tree."),
    ("wave", "She gave a friendly [wave] as she passed.",
     "'Wave' here is a gesture of greeting. It does not refer to a wave of water or a wave of emotion."),
    ("pitch", "The sales team prepared their [pitch] for the investors.",
     "'Pitch' here means a persuasive presentation. It does not refer to a sports field or the pitch of a sound."),
    ("cool", "She remained [cool] under pressure during the interview.",
     "'Cool' here means calm and composed. It does not refer to temperature or to something fashionable."),
    ("draw", "The match ended in a [draw].",
     "'Draw' here means a tied result. It does not mean to sketch or to pull something."),
]

VOCABULARY_BUILDING_TOPICS = [
    ("medicine", [
        ("diagnosis", "the identification of a disease", "The doctor made a diagnosis after reviewing the test results."),
        ("prognosis", "the likely course of a disease", "The prognosis for recovery was positive."),
        ("chronic", "persisting for a long time", "She suffers from a chronic back condition."),
        ("acute", "severe and sudden in onset", "He was admitted for acute appendicitis."),
        ("benign", "not harmful; not cancerous", "The tumour was found to be benign."),
        ("malignant", "tending to invade and destroy tissue", "The malignant cells spread rapidly."),
        ("remission", "a temporary recovery from disease", "The patient went into remission after treatment."),
        ("symptom", "a sign of a condition experienced by the patient", "Fever is a common symptom of infection."),
        ("pathogen", "a bacterium or virus causing disease", "The pathogen was identified in the lab."),
        ("immunisation", "the process of making immune to infection", "Childhood immunisation prevents many diseases."),
    ]),
    ("technology", [
        ("algorithm", "a set of rules for solving a problem", "The search engine uses a complex algorithm."),
        ("bandwidth", "the capacity of a network connection", "Streaming video requires high bandwidth."),
        ("encryption", "converting data into a coded form", "All messages are protected by encryption."),
        ("latency", "the delay before data transfer begins", "Low latency is critical for online gaming."),
        ("scalability", "the ability to handle growing amounts of work", "The system was designed for scalability."),
        ("interface", "a point where two systems meet and interact", "The user interface was redesigned for simplicity."),
        ("protocol", "a set of rules governing data exchange", "HTTP is the protocol used for web communication."),
        ("cache", "stored data for quick access", "Clearing the cache can resolve many browser issues."),
        ("iteration", "repetition of a process to generate results", "Each iteration of the software improved performance."),
        ("deployment", "the process of making software available", "The deployment was completed without downtime."),
    ]),
    ("law", [
        ("jurisdiction", "the official power to make legal decisions", "The case fell outside the court's jurisdiction."),
        ("litigation", "the process of taking legal action", "The company faced costly litigation."),
        ("precedent", "an earlier event used as a guide", "The ruling set a precedent for future cases."),
        ("plaintiff", "the party who brings a case to court", "The plaintiff sought damages for negligence."),
        ("defendant", "the party accused in a legal case", "The defendant pleaded not guilty."),
        ("statute", "a written law passed by a legislature", "The statute was amended to reflect modern standards."),
        ("liability", "the state of being legally responsible", "The company admitted liability for the accident."),
        ("injunction", "a court order requiring action or restraint", "An injunction was granted to stop the publication."),
        ("affidavit", "a written statement confirmed by oath", "She submitted an affidavit supporting her claim."),
        ("arbitration", "settling a dispute outside court", "The parties agreed to resolve the matter through arbitration."),
    ]),
]

COLLOCATION_DATA = [
    ("make", [
        ("make a decision", "She needs to make a decision by Friday."),
        ("make progress", "The team is making good progress on the project."),
        ("make an effort", "He made a real effort to improve his grades."),
        ("make a mistake", "Everyone makes mistakes; what matters is learning from them."),
        ("make a difference", "Volunteering can make a real difference in people's lives."),
    ]),
    ("take", [
        ("take a break", "Let's take a break and come back to this later."),
        ("take responsibility", "She took responsibility for the error."),
        ("take action", "The government must take action on climate change."),
        ("take a risk", "Starting a business means taking a risk."),
        ("take advantage", "He took advantage of the opportunity to travel."),
    ]),
    ("strong", [
        ("strong evidence", "There is strong evidence to support the theory."),
        ("strong opinion", "She has a strong opinion on the matter."),
        ("strong coffee", "He prefers strong coffee in the morning."),
        ("strong wind", "A strong wind knocked over the fence."),
        ("strong argument", "The lawyer presented a strong argument."),
    ]),
    ("heavy", [
        ("heavy rain", "Heavy rain caused flooding in the city."),
        ("heavy traffic", "We were delayed by heavy traffic on the motorway."),
        ("heavy workload", "She struggled with a heavy workload this month."),
        ("heavy smoker", "He was a heavy smoker for twenty years."),
        ("heavy emphasis", "The course places heavy emphasis on practical skills."),
    ]),
    ("break", [
        ("break a record", "She broke the world record in the 100m sprint."),
        ("break the news", "He had to break the news of the accident to the family."),
        ("break a habit", "It is hard to break a bad habit."),
        ("break the ice", "A joke can help break the ice at a meeting."),
        ("break a promise", "He never breaks a promise."),
    ]),
]


# ---------------------------------------------------------------------------
# Record generators
# ---------------------------------------------------------------------------

def _make_id(text: str) -> str:
    """Generate a unique MD5-based record ID."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _now() -> str:
    """Return current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def _difficulty_for_word(word: str) -> str:
    """Infer difficulty level from which pool the word belongs to."""
    common_words = {w[0] for w in WORDS_COMMON}
    intermediate_words = {w[0] for w in WORDS_INTERMEDIATE}
    advanced_words = {w[0] for w in WORDS_ADVANCED}
    if word in common_words:
        return "common"
    if word in intermediate_words:
        return "intermediate"
    if word in advanced_words:
        return "advanced"
    return "academic"


def generate_word_definition_records(n: int) -> list[dict]:
    """Generate word_definition records."""
    records = []
    for i in range(n):
        entry = ALL_WORDS[i % len(ALL_WORDS)]
        word, definition, etymology, example = entry
        uid_src = f"wd_{i}_{word}"
        input_text = f"Define '{word}' in context: {example}"
        output_text = (
            f"Definition: {definition}. "
            f"Etymology: {etymology}. "
            f"Usage example: {example}"
        )
        records.append({
            "record_id": _make_id(uid_src),
            "category": "word_definition",
            "word": word,
            "difficulty": _difficulty_for_word(word),
            "input_text": input_text,
            "output_text": output_text,
            "created_at": _now(),
        })
    return records


def generate_synonym_antonym_records(n: int) -> list[dict]:
    """Generate synonym_antonym records."""
    records = []
    for i in range(n):
        entry = SYNONYM_ANTONYM_DATA[i % len(SYNONYM_ANTONYM_DATA)]
        word, synonyms, antonyms, notes = entry
        uid_src = f"sa_{i}_{word}"
        input_text = f"Provide synonyms and antonyms for '{word}'."
        output_text = (
            f"Synonyms: {synonyms}. "
            f"Antonyms: {antonyms}. "
            f"Usage notes: {notes}"
        )
        records.append({
            "record_id": _make_id(uid_src),
            "category": "synonym_antonym",
            "word": word,
            "difficulty": "intermediate",
            "input_text": input_text,
            "output_text": output_text,
            "created_at": _now(),
        })
    return records


def generate_word_in_context_records(n: int) -> list[dict]:
    """Generate word_in_context records."""
    records = []
    for i in range(n):
        entry = WORD_IN_CONTEXT_DATA[i % len(WORD_IN_CONTEXT_DATA)]
        word, paragraph, meaning = entry
        uid_src = f"wic_{i}_{word}"
        input_text = f"What does the bracketed word mean in this sentence? {paragraph}"
        output_text = meaning
        records.append({
            "record_id": _make_id(uid_src),
            "category": "word_in_context",
            "word": word,
            "difficulty": "intermediate",
            "input_text": input_text,
            "output_text": output_text,
            "created_at": _now(),
        })
    return records


def generate_vocabulary_building_records(n: int) -> list[dict]:
    """Generate vocabulary_building records."""
    records = []
    for i in range(n):
        topic_entry = VOCABULARY_BUILDING_TOPICS[i % len(VOCABULARY_BUILDING_TOPICS)]
        topic, word_list = topic_entry
        uid_src = f"vb_{i}_{topic}"
        input_text = f"Give me 10 key vocabulary words related to '{topic}' with definitions and examples."
        lines = []
        for word, defn, example in word_list:
            lines.append(f"{word}: {defn}. Example: {example}")
        output_text = "\n".join(lines)
        records.append({
            "record_id": _make_id(uid_src),
            "category": "vocabulary_building",
            "word": topic,
            "difficulty": "intermediate",
            "input_text": input_text,
            "output_text": output_text,
            "created_at": _now(),
        })
    return records


def generate_collocation_records(n: int) -> list[dict]:
    """Generate collocation records."""
    records = []
    for i in range(n):
        entry = COLLOCATION_DATA[i % len(COLLOCATION_DATA)]
        word, collocations = entry
        uid_src = f"col_{i}_{word}"
        input_text = f"Give 5 natural collocations for the word '{word}' with example sentences."
        lines = []
        for colloc, example in collocations:
            lines.append(f"'{colloc}': {example}")
        output_text = "\n".join(lines)
        records.append({
            "record_id": _make_id(uid_src),
            "category": "collocations",
            "word": word,
            "difficulty": "intermediate",
            "input_text": input_text,
            "output_text": output_text,
            "created_at": _now(),
        })
    return records


# ---------------------------------------------------------------------------
# Database and JSONL builders
# ---------------------------------------------------------------------------

def _create_table(conn: sqlite3.Connection) -> None:
    """Create the vocab_records table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS vocab_records (
            record_id  TEXT PRIMARY KEY,
            category   TEXT,
            word       TEXT,
            difficulty TEXT,
            input_text TEXT,
            output_text TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()


def _insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records into vocab_records, ignoring duplicates."""
    sql = """
        INSERT OR IGNORE INTO vocab_records
            (record_id, category, word, difficulty, input_text, output_text, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (r["record_id"], r["category"], r["word"], r["difficulty"],
         r["input_text"], r["output_text"], r["created_at"])
        for r in records
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def build_database(n_per_category: int = 300) -> None:
    """
    Build the vocabulary SQLite database and JSONL file.

    Parameters
    ----------
    n_per_category : int
        Number of records to generate per category (default 300).
    """
    random.seed(22)
    all_records: list[dict] = []
    all_records.extend(generate_word_definition_records(n_per_category))
    all_records.extend(generate_synonym_antonym_records(n_per_category))
    all_records.extend(generate_word_in_context_records(n_per_category))
    all_records.extend(generate_vocabulary_building_records(n_per_category))
    all_records.extend(generate_collocation_records(n_per_category))

    # --- SQLite ---
    conn = sqlite3.connect(DB_PATH)
    try:
        _create_table(conn)
        inserted = _insert_records(conn, all_records)
    finally:
        conn.close()

    # --- JSONL ---
    written = 0
    try:
        with open(JSONL_PATH, "w", encoding="utf-8") as fh:
            for r in all_records:
                line = {
                    "instruction": r["input_text"],
                    "response": r["output_text"],
                    "category": r["category"],
                    "word": r["word"],
                    "difficulty": r["difficulty"],
                    "source": "janus_vocabulary_v1",
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        print(f"[ERROR] Failed to write JSONL: {exc}")

    total = len(all_records)
    print(f"Vocabulary DB : {total} records inserted={inserted}")
    print(f"JSONL lines   : {written}")
    print(f"DB path       : {DB_PATH}")
    print(f"JSONL path    : {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=400)
