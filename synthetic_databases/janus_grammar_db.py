"""
janus_grammar_db.py
-------------------
Generates grammar and syntax training data for the Janus NLP pipeline.
Produces a SQLite .db file and a .jsonl file when run directly.

Categories:
  - sentence_correction
  - sentence_transformation
  - punctuation
  - word_choice
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "janus_grammar.db")
JSONL_PATH = os.path.join(os.path.dirname(__file__), "janus_grammar.jsonl")

random.seed(11)

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

SUBJECT_VERB_ERRORS = [
    ("The dogs barks loudly every morning.", "The dogs bark loudly every morning.",
     "Subject 'dogs' is plural; use plural verb 'bark'."),
    ("She don't like spicy food.", "She doesn't like spicy food.",
     "Third-person singular subject requires 'doesn't'."),
    ("The team are playing well today.", "The team is playing well today.",
     "Collective noun 'team' takes a singular verb in American English."),
    ("Neither the teacher nor the students was ready.", "Neither the teacher nor the students were ready.",
     "With 'neither…nor', the verb agrees with the nearer subject 'students' (plural)."),
    ("Everyone have finished the exam.", "Everyone has finished the exam.",
     "'Everyone' is singular and requires 'has'."),
    ("The list of items are on the table.", "The list of items is on the table.",
     "The subject is 'list' (singular), not 'items'."),
    ("My brother and sister is coming tonight.", "My brother and sister are coming tonight.",
     "Compound subject joined by 'and' takes a plural verb."),
    ("There was many reasons for the delay.", "There were many reasons for the delay.",
     "'Reasons' is plural; use 'were'."),
    ("The news are shocking.", "The news is shocking.",
     "'News' is an uncountable noun and takes a singular verb."),
    ("Each of the students have a textbook.", "Each of the students has a textbook.",
     "'Each' is singular and requires 'has'."),
    ("The committee have made their decision.", "The committee has made its decision.",
     "In American English, collective nouns like 'committee' take singular verbs."),
    ("He go to school every day.", "He goes to school every day.",
     "Third-person singular present tense requires '-es' ending."),
    ("The data shows a clear trend.", "The data show a clear trend.",
     "'Data' is the plural of 'datum'; it takes a plural verb in formal usage."),
    ("Neither of the options are acceptable.", "Neither of the options is acceptable.",
     "'Neither' as a pronoun is singular."),
    ("The number of errors are increasing.", "The number of errors is increasing.",
     "'The number' is singular; 'a number' would be plural."),
    ("Politics are a dirty game.", "Politics is a dirty game.",
     "Subjects ending in '-ics' referring to a field of study take singular verbs."),
    ("Five miles are a long walk.", "Five miles is a long walk.",
     "A distance treated as a single unit takes a singular verb."),
    ("The jury have reached a verdict.", "The jury has reached a verdict.",
     "Collective noun 'jury' takes singular verb in American English."),
    ("Bread and butter are my favourite breakfast.", "Bread and butter is my favourite breakfast.",
     "When two things are considered a single unit, use a singular verb."),
    ("The scissors is on the desk.", "The scissors are on the desk.",
     "'Scissors' is always plural."),
]

TENSE_ERRORS = [
    ("Yesterday, she walks to the market.", "Yesterday, she walked to the market.",
     "Past time marker 'yesterday' requires simple past tense."),
    ("By the time he arrived, she already left.", "By the time he arrived, she had already left.",
     "An action completed before another past action requires past perfect."),
    ("I am living here since 2010.", "I have been living here since 2010.",
     "'Since' with a duration requires present perfect continuous."),
    ("He will call you when he will arrive.", "He will call you when he arrives.",
     "In time clauses with 'when', use simple present for future meaning."),
    ("She was cooking dinner when he will come home.", "She was cooking dinner when he came home.",
     "Both clauses describing the same past moment should use past tense."),
    ("We have visited Paris last year.", "We visited Paris last year.",
     "Specific past time expressions like 'last year' require simple past, not present perfect."),
    ("If I will see him, I will tell him.", "If I see him, I will tell him.",
     "In first conditional 'if' clauses, use simple present."),
    ("The sun rises in the east and set in the west.", "The sun rises in the east and sets in the west.",
     "Tense must be consistent within a sentence describing a general truth."),
    ("She has been to London three times last year.", "She went to London three times last year.",
     "Present perfect cannot be used with specific past time expressions."),
    ("He was studying when the phone rings.", "He was studying when the phone rang.",
     "Both verbs should be in past tense for a past narrative."),
    ("I finish my homework before dinner yesterday.", "I had finished my homework before dinner yesterday.",
     "An action completed before another past event uses past perfect."),
    ("They are friends since childhood.", "They have been friends since childhood.",
     "A state continuing from the past to now uses present perfect."),
    ("She will be a doctor after she graduates.", "She will be a doctor after she graduates.",
     "Correct — but if written 'will graduate', that is an error; time clauses use simple present."),
    ("He told me that he will come.", "He told me that he would come.",
     "Reported speech shifts future 'will' to 'would'."),
    ("I am knowing the answer.", "I know the answer.",
     "Stative verbs like 'know' are not used in continuous tenses."),
]

DANGLING_MODIFIER_ERRORS = [
    ("Walking down the street, the trees were beautiful.", "Walking down the street, I found the trees beautiful.",
     "The participial phrase must modify the subject of the main clause."),
    ("Having finished the exam, the room was quiet.", "Having finished the exam, the students found the room quiet.",
     "The dangling participle 'having finished' needs a logical subject."),
    ("To improve your writing, practice is essential.", "To improve your writing, you must practice.",
     "The infinitive phrase implies a subject that should match the main clause subject."),
    ("Exhausted from the hike, the bed felt wonderful.", "Exhausted from the hike, she found the bed wonderful.",
     "The modifier 'exhausted' must refer to a person, not 'the bed'."),
    ("Running late, the bus was missed.", "Running late, she missed the bus.",
     "The participial phrase 'running late' needs a human subject."),
    ("After eating dinner, the dishes were washed.", "After eating dinner, she washed the dishes.",
     "The implied subject of 'eating' must match the subject of the main clause."),
    ("Being a rainy day, we stayed indoors.", "Because it was a rainy day, we stayed indoors.",
     "Absolute phrases with 'being' often create dangling modifiers; restructure the sentence."),
    ("Covered in mud, the bath was drawn immediately.", "Covered in mud, he drew a bath immediately.",
     "The modifier 'covered in mud' must refer to the person, not 'the bath'."),
    ("To bake a perfect cake, the oven must be preheated.", "To bake a perfect cake, you must preheat the oven.",
     "The infinitive phrase implies 'you' as the subject."),
    ("Driving through the mountains, the scenery was breathtaking.", "Driving through the mountains, we found the scenery breathtaking.",
     "The participial phrase needs a human subject in the main clause."),
]

COMMA_SPLICE_ERRORS = [
    ("I love coffee, it keeps me awake.", "I love coffee because it keeps me awake.",
     "A comma splice joins two independent clauses with only a comma; use a conjunction or semicolon."),
    ("She was tired, she went to bed early.", "She was tired, so she went to bed early.",
     "Add a coordinating conjunction after the comma to fix the splice."),
    ("The movie was long, we enjoyed it.", "The movie was long, but we enjoyed it.",
     "Use 'but' to show contrast between the two clauses."),
    ("He studied hard, he passed the exam.", "He studied hard; therefore, he passed the exam.",
     "Use a semicolon with a conjunctive adverb to connect related independent clauses."),
    ("It was raining, we cancelled the picnic.", "It was raining, so we cancelled the picnic.",
     "Add coordinating conjunction 'so' to show cause and effect."),
    ("The report is due tomorrow, I haven't started it.", "The report is due tomorrow, yet I haven't started it.",
     "Use 'yet' to show contrast."),
    ("She speaks French, she also speaks German.", "She speaks French, and she also speaks German.",
     "Use 'and' to join two related independent clauses."),
    ("The door was open, anyone could walk in.", "The door was open, so anyone could walk in.",
     "Add 'so' to show the logical consequence."),
    ("I called him, he didn't answer.", "I called him, but he didn't answer.",
     "Use 'but' to show contrast."),
    ("The sun set, the stars appeared.", "The sun set, and the stars appeared.",
     "Use 'and' to join two sequential events."),
]

RUN_ON_ERRORS = [
    ("I went to the store I bought milk.", "I went to the store, and I bought milk.",
     "Two independent clauses must be separated by a comma and conjunction, or a semicolon."),
    ("She loves reading she reads every night.", "She loves reading; she reads every night.",
     "Use a semicolon to join two closely related independent clauses."),
    ("The cat sat on the mat the dog lay on the floor.", "The cat sat on the mat, and the dog lay on the floor.",
     "Join two independent clauses with a comma and coordinating conjunction."),
    ("He was hungry he ate a sandwich.", "He was hungry, so he ate a sandwich.",
     "Add a comma and 'so' to show cause and effect."),
    ("We arrived late the show had already started.", "We arrived late; the show had already started.",
     "Use a semicolon between two related independent clauses."),
]

APOSTROPHE_ERRORS = [
    ("The dogs tail was wagging.", "The dog's tail was wagging.",
     "Use an apostrophe to show possession: dog's."),
    ("Its a beautiful day.", "It's a beautiful day.",
     "'It's' is a contraction of 'it is'; 'its' is the possessive pronoun."),
    ("The childrens toys were scattered.", "The children's toys were scattered.",
     "For irregular plurals, add apostrophe + s: children's."),
    ("The two sister's rooms are clean.", "The two sisters' rooms are clean.",
     "For plural possessives, place the apostrophe after the s: sisters'."),
    ("Dont forget your umbrella.", "Don't forget your umbrella.",
     "Contractions require an apostrophe: don't = do not."),
    ("The companys policy has changed.", "The company's policy has changed.",
     "Singular possessive requires apostrophe + s: company's."),
    ("Whos coming to the party?", "Who's coming to the party?",
     "'Who's' is a contraction of 'who is'."),
    ("The mens locker room is closed.", "The men's locker room is closed.",
     "Irregular plural possessive: men's."),
    ("Its fur is very soft.", "Its fur is very soft.",
     "Correct — 'its' as a possessive pronoun needs no apostrophe."),
    ("She's going to her sisters house.", "She's going to her sister's house.",
     "Possessive 'sister's' requires an apostrophe."),
]

DOUBLE_NEGATIVE_ERRORS = [
    ("I don't have no money.", "I don't have any money.",
     "Two negatives cancel each other out; use 'any' instead of 'no'."),
    ("She didn't say nothing.", "She didn't say anything.",
     "Replace 'nothing' with 'anything' to avoid a double negative."),
    ("He can't do nothing about it.", "He can't do anything about it.",
     "Use 'anything' to maintain a single negative."),
    ("We haven't never been there.", "We have never been there.",
     "Remove 'haven't' or 'never' — only one negative is needed."),
    ("They don't want no trouble.", "They don't want any trouble.",
     "Replace 'no' with 'any' to avoid a double negative."),
    ("I barely didn't sleep.", "I barely slept.",
     "'Barely' is already a negative; adding 'didn't' creates a double negative."),
    ("Nobody never helps me.", "Nobody ever helps me.",
     "Replace 'never' with 'ever' since 'nobody' is already negative."),
    ("She couldn't hardly breathe.", "She could hardly breathe.",
     "'Hardly' is a negative adverb; 'couldn't' creates a double negative."),
]

PRONOUN_AGREEMENT_ERRORS = [
    ("Everyone should bring their own lunch.", "Everyone should bring his or her own lunch.",
     "In formal writing, 'everyone' (singular) takes 'his or her', though 'their' is accepted informally."),
    ("The company announced their new policy.", "The company announced its new policy.",
     "A company is a singular entity; use 'its'."),
    ("Each student must submit their assignment.", "Each student must submit his or her assignment.",
     "'Each' is singular; use 'his or her' in formal contexts."),
    ("The team celebrated their victory.", "The team celebrated its victory.",
     "In American English, collective nouns take singular pronouns."),
    ("Someone left their bag on the bus.", "Someone left his or her bag on the bus.",
     "'Someone' is singular; use 'his or her' in formal writing."),
    ("If a person works hard, they will succeed.", "If a person works hard, he or she will succeed.",
     "In formal writing, use 'he or she' to agree with the singular antecedent 'a person'."),
    ("The jury gave their verdict.", "The jury gave its verdict.",
     "Collective noun 'jury' takes singular pronoun 'its' in American English."),
    ("Neither of the boys forgot their homework.", "Neither of the boys forgot his homework.",
     "'Neither' is singular; use singular pronoun 'his'."),
]

TRANSFORMATION_PAIRS = [
    ("active_to_passive", "The chef cooked the meal.", "The meal was cooked by the chef."),
    ("passive_to_active", "The book was written by the author.", "The author wrote the book."),
    ("active_to_passive", "Scientists discovered a new planet.", "A new planet was discovered by scientists."),
    ("passive_to_active", "The window was broken by the child.", "The child broke the window."),
    ("direct_to_indirect", 'She said, "I am tired."', "She said that she was tired."),
    ("indirect_to_direct", "He told me that he would come.", 'He said, "I will come."'),
    ("direct_to_indirect", 'He asked, "Are you ready?"', "He asked if I was ready."),
    ("positive_to_negative", "She always arrives on time.", "She never arrives late."),
    ("negative_to_positive", "He is not unhappy.", "He is happy."),
    ("simple_to_complex", "She was tired. She went to bed.", "Because she was tired, she went to bed."),
    ("complex_to_simple", "Although it was raining, they went for a walk.", "It was raining. They went for a walk anyway."),
    ("active_to_passive", "The manager approved the proposal.", "The proposal was approved by the manager."),
    ("direct_to_indirect", 'She asked, "Where is the library?"', "She asked where the library was."),
    ("positive_to_negative", "He is always polite.", "He is never rude."),
    ("simple_to_complex", "He studied hard. He passed the exam.", "Because he studied hard, he passed the exam."),
    ("active_to_passive", "The dog chased the cat.", "The cat was chased by the dog."),
    ("passive_to_active", "The letter was sent by the secretary.", "The secretary sent the letter."),
    ("indirect_to_direct", "She told him that she loved him.", 'She said, "I love you."'),
    ("negative_to_positive", "The task is not impossible.", "The task is possible."),
    ("complex_to_simple", "Since he was late, he missed the bus.", "He was late. He missed the bus."),
]

PUNCTUATION_ERRORS = [
    ("However she refused to give up.", "However, she refused to give up.",
     "Use a comma after introductory conjunctive adverbs like 'however'."),
    ("The meeting is on Monday January 15 2024.", "The meeting is on Monday, January 15, 2024.",
     "Commas separate elements in dates: day, month date, year."),
    ("She bought apples oranges and bananas.", "She bought apples, oranges, and bananas.",
     "Use commas to separate items in a series (Oxford comma recommended)."),
    ("Its a lovely day isnt it", "It's a lovely day, isn't it?",
     "Add apostrophe in contraction, comma before tag question, and question mark at end."),
    ("The professor said study hard for the exam", 'The professor said, "Study hard for the exam."',
     "Use a comma before a direct quotation and quotation marks around it."),
    ("After the long exhausting journey they finally arrived home.", "After the long, exhausting journey, they finally arrived home.",
     "Use commas after introductory phrases and between coordinate adjectives."),
    ("He is a doctor she is a lawyer.", "He is a doctor; she is a lawyer.",
     "Use a semicolon to join two closely related independent clauses."),
    ("The results were as follows speed accuracy and efficiency.", "The results were as follows: speed, accuracy, and efficiency.",
     "Use a colon to introduce a list."),
    ("Dont touch that its hot", "Don't touch that; it's hot.",
     "Add apostrophes in contractions and a semicolon between independent clauses."),
    ("My sister who lives in Paris is visiting next week.", "My sister, who lives in Paris, is visiting next week.",
     "Non-restrictive relative clauses are set off by commas."),
    ("Well I think you should reconsider.", "Well, I think you should reconsider.",
     "Use a comma after introductory interjections like 'well'."),
    ("The book which I borrowed from the library was fascinating.", "The book that I borrowed from the library was fascinating.",
     "Use 'that' for restrictive clauses (no commas needed)."),
    ("She asked are you coming to the party", 'She asked, "Are you coming to the party?"',
     "Direct questions within a sentence need quotation marks and a question mark."),
    ("To be honest I have no idea what happened.", "To be honest, I have no idea what happened.",
     "Use a comma after introductory phrases."),
    ("The three main causes were poverty lack of education and unemployment.", "The three main causes were poverty, lack of education, and unemployment.",
     "Separate list items with commas."),
]

WORD_CHOICE_ERRORS = [
    ("The medicine had a positive affect on the patient.", "The medicine had a positive effect on the patient.",
     "'Effect' is the noun meaning result; 'affect' is usually a verb meaning to influence."),
    ("Their going to the store later.", "They're going to the store later.",
     "'They're' is a contraction of 'they are'; 'their' is possessive."),
    ("The dog wagged it's tail.", "The dog wagged its tail.",
     "'Its' is the possessive pronoun; 'it's' is a contraction of 'it is'."),
    ("I need to lay down for a while.", "I need to lie down for a while.",
     "'Lie' means to recline; 'lay' requires a direct object."),
    ("The principle reason for the delay was weather.", "The principal reason for the delay was weather.",
     "'Principal' means main or chief; 'principle' means a rule or belief."),
    ("She complemented him on his work.", "She complimented him on his work.",
     "'Compliment' means to praise; 'complement' means to complete or enhance."),
    ("The data is less accurate then expected.", "The data is less accurate than expected.",
     "'Than' is used in comparisons; 'then' refers to time."),
    ("He was disinterested in the outcome.", "He was uninterested in the outcome.",
     "'Uninterested' means not interested; 'disinterested' means impartial."),
    ("The weather effected our plans.", "The weather affected our plans.",
     "'Affect' is the verb meaning to influence; 'effect' is usually a noun."),
    ("She excepted the award graciously.", "She accepted the award graciously.",
     "'Accept' means to receive; 'except' means to exclude."),
    ("The council decided to altar the plan.", "The council decided to alter the plan.",
     "'Alter' means to change; 'altar' is a religious table."),
    ("He past the exam with flying colours.", "He passed the exam with flying colours.",
     "'Passed' is the past tense of 'pass'; 'past' is a noun, adjective, or preposition."),
    ("The new policy will insure quality.", "The new policy will ensure quality.",
     "'Ensure' means to make certain; 'insure' relates to insurance."),
    ("She was board during the lecture.", "She was bored during the lecture.",
     "'Bored' means feeling uninterested; 'board' is a flat piece of wood or a committee."),
    ("The weather is quite whether you like it or not.", "The weather is quite unpredictable whether you like it or not.",
     "'Whether' introduces alternatives; 'weather' refers to atmospheric conditions."),
    ("He has a flare for languages.", "He has a flair for languages.",
     "'Flair' means a natural talent; 'flare' means to burn or spread outward."),
    ("The stationary was ordered last week.", "The stationery was ordered last week.",
     "'Stationery' refers to writing materials; 'stationary' means not moving."),
    ("She poured over the documents.", "She pored over the documents.",
     "'Pore over' means to study carefully; 'pour' means to flow or tip liquid."),
    ("The desert was delicious.", "The dessert was delicious.",
     "'Dessert' is the sweet course; 'desert' is an arid landscape."),
    ("He was complementary about her performance.", "He was complimentary about her performance.",
     "'Complimentary' means expressing praise or given free; 'complementary' means completing something."),
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


def generate_sentence_correction_records(n: int) -> list[dict]:
    """Generate sentence_correction records by cycling through error pools."""
    pools = {
        "subject_verb_agreement": SUBJECT_VERB_ERRORS,
        "tense_consistency": TENSE_ERRORS,
        "dangling_modifiers": DANGLING_MODIFIER_ERRORS,
        "comma_splice": COMMA_SPLICE_ERRORS,
        "run_on_sentences": RUN_ON_ERRORS,
        "apostrophe_misuse": APOSTROPHE_ERRORS,
        "double_negatives": DOUBLE_NEGATIVE_ERRORS,
        "pronoun_agreement": PRONOUN_AGREEMENT_ERRORS,
    }
    difficulties = ["beginner", "intermediate", "advanced"]
    records = []
    error_types = list(pools.keys())
    per_type = max(n // len(error_types), 1)

    for error_type, pool in pools.items():
        for i in range(per_type):
            item = pool[i % len(pool)]
            wrong, correct, explanation = item
            # Add slight variation via index suffix to ensure unique IDs
            uid_src = f"sc_{error_type}_{i}_{wrong}"
            record = {
                "record_id": _make_id(uid_src),
                "category": "sentence_correction",
                "error_type": error_type,
                "input_text": wrong,
                "output_text": correct,
                "explanation": explanation,
                "difficulty": random.choice(difficulties),
                "created_at": _now(),
            }
            records.append(record)

    # Fill remaining to reach n
    while len(records) < n:
        et = random.choice(error_types)
        pool = pools[et]
        item = random.choice(pool)
        wrong, correct, explanation = item
        uid_src = f"sc_{et}_{len(records)}_{wrong}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "sentence_correction",
            "error_type": et,
            "input_text": wrong,
            "output_text": correct,
            "explanation": explanation,
            "difficulty": random.choice(difficulties),
            "created_at": _now(),
        })

    return records[:n]


def generate_sentence_transformation_records(n: int) -> list[dict]:
    """Generate sentence_transformation records."""
    difficulties = ["beginner", "intermediate", "advanced"]
    records = []
    for i in range(n):
        pair = TRANSFORMATION_PAIRS[i % len(TRANSFORMATION_PAIRS)]
        transform_type, original, transformed = pair
        uid_src = f"st_{transform_type}_{i}_{original}"
        input_text = f"Transform the following sentence ({transform_type.replace('_', ' ')}): {original}"
        output_text = transformed
        explanation = f"Transformation type: {transform_type.replace('_', ' ')}."
        records.append({
            "record_id": _make_id(uid_src),
            "category": "sentence_transformation",
            "error_type": transform_type,
            "input_text": input_text,
            "output_text": output_text,
            "explanation": explanation,
            "difficulty": random.choice(difficulties),
            "created_at": _now(),
        })
    return records


def generate_punctuation_records(n: int) -> list[dict]:
    """Generate punctuation correction records."""
    difficulties = ["beginner", "intermediate", "advanced"]
    records = []
    for i in range(n):
        item = PUNCTUATION_ERRORS[i % len(PUNCTUATION_ERRORS)]
        wrong, correct, rule = item
        uid_src = f"punc_{i}_{wrong}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "punctuation",
            "error_type": "punctuation_error",
            "input_text": wrong,
            "output_text": correct,
            "explanation": rule,
            "difficulty": random.choice(difficulties),
            "created_at": _now(),
        })
    return records


def generate_word_choice_records(n: int) -> list[dict]:
    """Generate word_choice correction records."""
    difficulties = ["beginner", "intermediate", "advanced"]
    records = []
    for i in range(n):
        item = WORD_CHOICE_ERRORS[i % len(WORD_CHOICE_ERRORS)]
        wrong, correct, explanation = item
        uid_src = f"wc_{i}_{wrong}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "word_choice",
            "error_type": "word_choice_error",
            "input_text": wrong,
            "output_text": correct,
            "explanation": explanation,
            "difficulty": random.choice(difficulties),
            "created_at": _now(),
        })
    return records


# ---------------------------------------------------------------------------
# Database and JSONL builders
# ---------------------------------------------------------------------------

def _create_table(conn: sqlite3.Connection) -> None:
    """Create the grammar_records table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS grammar_records (
            record_id  TEXT PRIMARY KEY,
            category   TEXT,
            error_type TEXT,
            input_text TEXT,
            output_text TEXT,
            explanation TEXT,
            difficulty  TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()


def _insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records into grammar_records, ignoring duplicates. Returns inserted count."""
    sql = """
        INSERT OR IGNORE INTO grammar_records
            (record_id, category, error_type, input_text, output_text, explanation, difficulty, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (r["record_id"], r["category"], r["error_type"], r["input_text"],
         r["output_text"], r["explanation"], r["difficulty"], r["created_at"])
        for r in records
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def build_database(n_per_category: int = 300) -> None:
    """
    Build the grammar SQLite database and JSONL file.

    Parameters
    ----------
    n_per_category : int
        Number of records to generate per category (default 300).
    """
    random.seed(11)
    all_records: list[dict] = []
    all_records.extend(generate_sentence_correction_records(n_per_category))
    all_records.extend(generate_sentence_transformation_records(n_per_category))
    all_records.extend(generate_punctuation_records(n_per_category))
    all_records.extend(generate_word_choice_records(n_per_category))

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
                    "response": r["output_text"] + " — " + r["explanation"],
                    "category": r["category"],
                    "error_type": r["error_type"],
                    "difficulty": r["difficulty"],
                    "source": "janus_grammar_v1",
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        print(f"[ERROR] Failed to write JSONL: {exc}")

    total = len(all_records)
    print(f"Grammar DB  : {total} records inserted={inserted}")
    print(f"JSONL lines : {written}")
    print(f"DB path     : {DB_PATH}")
    print(f"JSONL path  : {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=400)
