"""
janus_multilingual_db.py
-------------------------
Generates multilingual and translation training data for the Janus NLP pipeline.
Produces a SQLite .db file and a .jsonl file when run directly.

Categories:
  - translation_pairs
  - false_friends
  - idiom_translation
  - cultural_context
  - language_detection
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "janus_multilingual.db")
JSONL_PATH = os.path.join(os.path.dirname(__file__), "janus_multilingual.jsonl")

random.seed(55)

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

TRANSLATION_PAIRS_DATA = [
    (
        "Hello, how are you?",
        "Spanish", "Hola, ¿cómo estás?",
        "French", "Bonjour, comment allez-vous?",
        "Back-translation note: The Spanish uses informal 'tú' form; the French uses formal 'vous' form.",
    ),
    (
        "The weather is beautiful today.",
        "Spanish", "El tiempo está hermoso hoy.",
        "German", "Das Wetter ist heute wunderschön.",
        "Back-translation note: Both translations preserve the present tense and the adjective placement.",
    ),
    (
        "I would like a cup of coffee, please.",
        "French", "Je voudrais une tasse de café, s'il vous plaît.",
        "Japanese", "コーヒーを一杯いただけますか。",
        "Back-translation note: The Japanese uses a polite request form (itadakemasu ka) rather than a direct statement.",
    ),
    (
        "Where is the nearest train station?",
        "Spanish", "¿Dónde está la estación de tren más cercana?",
        "French", "Où est la gare la plus proche?",
        "Back-translation note: Both translations use the verb 'to be' (estar/est) for location.",
    ),
    (
        "She has been studying medicine for five years.",
        "German", "Sie studiert seit fünf Jahren Medizin.",
        "French", "Elle étudie la médecine depuis cinq ans.",
        "Back-translation note: German and French use present tense with 'seit/depuis' for ongoing actions, unlike English perfect.",
    ),
    (
        "The children are playing in the park.",
        "Spanish", "Los niños están jugando en el parque.",
        "Mandarin", "孩子们在公园里玩耍。",
        "Back-translation note: Mandarin does not use a continuous tense marker; context implies the ongoing action.",
    ),
    (
        "I am sorry for the inconvenience.",
        "French", "Je suis désolé pour le dérangement.",
        "German", "Es tut mir leid für die Unannehmlichkeiten.",
        "Back-translation note: German uses a reflexive expression (es tut mir leid) literally meaning 'it does me sorrow'.",
    ),
    (
        "Can you help me find this address?",
        "Spanish", "¿Puede ayudarme a encontrar esta dirección?",
        "Japanese", "この住所を見つけるのを手伝っていただけますか。",
        "Back-translation note: Japanese uses a polite request form with 'itadakemasu ka' for formal requests.",
    ),
    (
        "The meeting has been postponed until next week.",
        "French", "La réunion a été reportée à la semaine prochaine.",
        "German", "Das Meeting wurde auf nächste Woche verschoben.",
        "Back-translation note: Both use passive constructions equivalent to the English original.",
    ),
    (
        "I love learning new languages.",
        "Spanish", "Me encanta aprender nuevos idiomas.",
        "Mandarin", "我喜欢学习新语言。",
        "Back-translation note: Spanish uses 'encantar' (to delight) rather than a direct equivalent of 'love'.",
    ),
    (
        "Please turn off the lights when you leave.",
        "French", "Veuillez éteindre les lumières en partant.",
        "German", "Bitte schalten Sie das Licht aus, wenn Sie gehen.",
        "Back-translation note: French uses 'veuillez' (please be so kind as to) for polite imperatives.",
    ),
    (
        "The report will be ready by tomorrow morning.",
        "Spanish", "El informe estará listo para mañana por la mañana.",
        "Japanese", "レポートは明日の朝までに準備できます。",
        "Back-translation note: Japanese uses a potential form (dekimasu) to express readiness.",
    ),
    (
        "He forgot his umbrella at the office.",
        "French", "Il a oublié son parapluie au bureau.",
        "German", "Er hat seinen Regenschirm im Büro vergessen.",
        "Back-translation note: Both use the perfect tense (passé composé / Perfekt) for completed past actions.",
    ),
    (
        "We need to discuss this matter urgently.",
        "Spanish", "Necesitamos discutir este asunto urgentemente.",
        "Mandarin", "我们需要紧急讨论这件事。",
        "Back-translation note: Mandarin places the adverb before the verb, unlike English.",
    ),
    (
        "The library closes at nine o'clock in the evening.",
        "French", "La bibliothèque ferme à neuf heures du soir.",
        "German", "Die Bibliothek schließt um neun Uhr abends.",
        "Back-translation note: Both translations use 24-hour or evening qualifiers consistent with local convention.",
    ),
]

FALSE_FRIENDS_DATA = [
    (
        "embarrassed (English) / embarazada (Spanish)",
        "English", "Spanish",
        "English 'embarrassed' means feeling ashamed or self-conscious. Spanish 'embarazada' means pregnant. Example (English): She was embarrassed by the mistake. Example (Spanish): Ella está embarazada de tres meses. (She is three months pregnant.)",
    ),
    (
        "sensible (English) / sensible (French)",
        "English", "French",
        "English 'sensible' means practical and reasonable. French 'sensible' means sensitive or perceptible. Example (English): She made a sensible decision. Example (French): Il est très sensible aux critiques. (He is very sensitive to criticism.)",
    ),
    (
        "gift (English) / Gift (German)",
        "English", "German",
        "English 'gift' means a present. German 'Gift' means poison. Example (English): She received a gift for her birthday. Example (German): Das Gift ist gefährlich. (The poison is dangerous.)",
    ),
    (
        "actual (English) / actuel (French)",
        "English", "French",
        "English 'actual' means real or existing in fact. French 'actuel' means current or present. Example (English): The actual cost was higher than expected. Example (French): Le problème actuel est complexe. (The current problem is complex.)",
    ),
    (
        "library (English) / librería (Spanish)",
        "English", "Spanish",
        "English 'library' is a place where books are borrowed. Spanish 'librería' is a bookshop. Example (English): I borrowed this book from the library. Example (Spanish): Compré el libro en la librería. (I bought the book at the bookshop.)",
    ),
    (
        "fabric (English) / fabrique (French)",
        "English", "French",
        "English 'fabric' means cloth or textile. French 'fabrique' means factory. Example (English): The dress is made of silk fabric. Example (French): Il travaille dans une fabrique. (He works in a factory.)",
    ),
    (
        "sympathetic (English) / sympathique (French)",
        "English", "French",
        "English 'sympathetic' means showing compassion. French 'sympathique' means likeable or pleasant. Example (English): She was sympathetic to his situation. Example (French): Il est très sympathique. (He is very likeable.)",
    ),
    (
        "eventually (English) / éventuellement (French)",
        "English", "French",
        "English 'eventually' means in the end, after a long time. French 'éventuellement' means possibly or if necessary. Example (English): He eventually found a solution. Example (French): Éventuellement, nous pourrions changer le plan. (Possibly, we could change the plan.)",
    ),
    (
        "pretend (English) / pretender (Spanish)",
        "English", "Spanish",
        "English 'pretend' means to act as if something is true when it is not. Spanish 'pretender' means to intend or to aspire to. Example (English): The children pretended to be pirates. Example (Spanish): Pretendo terminar el proyecto esta semana. (I intend to finish the project this week.)",
    ),
    (
        "chef (English) / chef (French)",
        "English", "French",
        "English 'chef' specifically means a professional cook. French 'chef' means leader or head (of any organisation). Example (English): The chef prepared an excellent meal. Example (French): Le chef de l'entreprise a démissionné. (The head of the company resigned.)",
    ),
]

IDIOM_TRANSLATION_DATA = [
    (
        "It's raining cats and dogs.",
        "Spanish", "Está lloviendo a cántaros. (It's raining in pitchers.)",
        "French", "Il pleut des cordes. (It's raining ropes.)",
        "Meaning: It is raining very heavily. Literal English: cats and dogs are falling from the sky.",
    ),
    (
        "Break a leg.",
        "German", "Hals- und Beinbruch. (Neck and leg break.)",
        "French", "Merde! (Used as a theatrical good luck expression.)",
        "Meaning: Good luck, especially before a performance. Literal English: fracture your leg.",
    ),
    (
        "The ball is in your court.",
        "Spanish", "La pelota está en tu tejado. (The ball is on your roof.)",
        "French", "C'est à toi de jouer. (It's your turn to play.)",
        "Meaning: It is your responsibility to take the next action. Literal English: a ball has landed in your side of a tennis court.",
    ),
    (
        "Bite the bullet.",
        "Spanish", "Aguantar el tipo. (Hold your type/composure.)",
        "French", "Serrer les dents. (Clench your teeth.)",
        "Meaning: To endure a painful or difficult situation. Literal English: to bite on a bullet, as soldiers did during surgery.",
    ),
    (
        "Hit the nail on the head.",
        "German", "Den Nagel auf den Kopf treffen. (To hit the nail on the head — same idiom.)",
        "Spanish", "Dar en el clavo. (To hit the nail.)",
        "Meaning: To describe exactly what is causing a situation or problem. Literal English: to strike a nail precisely.",
    ),
    (
        "Spill the beans.",
        "Spanish", "Irse de la lengua. (To let one's tongue go.)",
        "French", "Vendre la mèche. (To sell the wick/fuse.)",
        "Meaning: To reveal secret information accidentally. Literal English: to knock over a container of beans.",
    ),
    (
        "Once in a blue moon.",
        "Spanish", "De higos a brevas. (From figs to early figs — rarely.)",
        "French", "Tous les trente-six du mois. (Every 36th of the month — never.)",
        "Meaning: Very rarely. Literal English: when the moon appears blue, a rare phenomenon.",
    ),
    (
        "Bite off more than you can chew.",
        "Spanish", "Abarcar más de lo que se puede apretar. (To grasp more than you can squeeze.)",
        "French", "Avoir les yeux plus grands que le ventre. (To have eyes bigger than your stomach.)",
        "Meaning: To take on more responsibility than you can handle. Literal English: to put too much food in your mouth.",
    ),
    (
        "Kill two birds with one stone.",
        "Spanish", "Matar dos pájaros de un tiro. (Kill two birds with one shot — same image.)",
        "French", "Faire d'une pierre deux coups. (To make two hits with one stone.)",
        "Meaning: To accomplish two things with a single action. Literal English: to throw one stone and hit two birds.",
    ),
    (
        "Let the cat out of the bag.",
        "German", "Die Katze aus dem Sack lassen. (Let the cat out of the sack — same image.)",
        "French", "Vendre la mèche. (To sell the fuse — reveal a secret.)",
        "Meaning: To accidentally reveal a secret. Literal English: to release a cat hidden in a bag.",
    ),
]

CULTURAL_CONTEXT_DATA = [
    (
        "English", "Spanish",
        "Buen provecho.",
        "The phrase 'buen provecho' is said to someone who is eating, equivalent to 'enjoy your meal'. It is also said to strangers in restaurants as you pass their table — a cultural norm of acknowledging others' meals that has no direct English equivalent.",
    ),
    (
        "Japanese", "English",
        "Itadakimasu (いただきます)",
        "Said before eating, 'itadakimasu' literally means 'I humbly receive'. It expresses gratitude to everyone involved in producing the meal — farmers, cooks, and the food itself. Translating it simply as 'let's eat' loses the cultural depth of humility and gratitude.",
    ),
    (
        "French", "English",
        "Tu me manques.",
        "Literally 'you are missing from me', this French phrase expresses 'I miss you' but frames the absence as something the speaker experiences physically. The English translation 'I miss you' reverses the subject, losing the nuance of the other person's absence being felt by the speaker.",
    ),
    (
        "German", "English",
        "Schadenfreude",
        "Schadenfreude describes pleasure derived from another person's misfortune. English has borrowed this word directly because there is no single-word equivalent. A translation like 'malicious joy' captures the meaning but lacks the cultural resonance of the German original.",
    ),
    (
        "Mandarin", "English",
        "关系 (Guānxi)",
        "Guānxi refers to the network of relationships and social connections that facilitate business and social life in Chinese culture. It implies mutual obligation and reciprocity. Translating it as 'connections' or 'networking' misses the deep cultural significance of trust, obligation, and long-term relationship building.",
    ),
    (
        "Arabic", "English",
        "Inshallah (إن شاء الله)",
        "Literally 'if God wills it', inshallah is used to express hope, uncertainty, or acceptance of outcomes. In context, it can mean anything from a sincere prayer to a polite way of saying 'probably not'. Translating it simply as 'hopefully' or 'God willing' does not capture its full range of social uses.",
    ),
    (
        "Portuguese", "English",
        "Saudade",
        "Saudade describes a deep emotional state of nostalgic longing for something or someone loved and lost. It carries a bittersweet quality — the pleasure of remembering combined with the pain of absence. No single English word captures this; 'nostalgia' is the closest but lacks the melancholic depth.",
    ),
    (
        "Danish", "English",
        "Hygge",
        "Hygge refers to a quality of cosiness and comfortable conviviality that engenders a feeling of contentment or well-being. It is central to Danish culture and encompasses the atmosphere of a warm gathering, a candle-lit room, or a cosy evening. 'Cosiness' is the closest English equivalent but does not capture the social and emotional dimensions.",
    ),
    (
        "Japanese", "English",
        "木漏れ日 (Komorebi)",
        "Komorebi refers to the interplay of light and leaves when sunlight filters through trees. There is no single English word for this concept. A translation might read 'sunlight filtering through leaves', but this is a description rather than a word, reflecting how Japanese culture has named a specific natural phenomenon.",
    ),
    (
        "Spanish", "English",
        "Madrugada",
        "Madrugada refers specifically to the hours between midnight and dawn — a time that English covers only with the vague phrase 'the small hours' or 'the early hours'. In Spanish-speaking cultures, this period has its own identity, associated with late-night gatherings, music, and a particular atmosphere.",
    ),
]

LANGUAGE_DETECTION_DATA = [
    (
        "Bonjour! I am going to the marché today. Das Wetter ist schön.",
        "French, English, German",
        "High confidence. Clues: 'Bonjour' is a standard French greeting; 'marché' is French for market; 'Das Wetter ist schön' is a complete German sentence meaning 'The weather is beautiful'.",
    ),
    (
        "I love sushi and ramen. 日本語は難しいですが、面白いです。",
        "English, Japanese",
        "High confidence. Clues: The first sentence is standard English; the second sentence uses Japanese script (hiragana and kanji) and means 'Japanese is difficult but interesting'.",
    ),
    (
        "Hola amigos! The fiesta was increíble last night.",
        "Spanish, English",
        "High confidence. Clues: 'Hola amigos' is a Spanish greeting; 'fiesta' is Spanish for party; 'increíble' is Spanish for incredible. The rest is English.",
    ),
    (
        "Ich bin sehr müde. I need to sleep. Bonne nuit!",
        "German, English, French",
        "High confidence. Clues: 'Ich bin sehr müde' is German for 'I am very tired'; 'I need to sleep' is English; 'Bonne nuit' is French for 'Good night'.",
    ),
    (
        "今天天气很好。Let's go for a walk in the park.",
        "Mandarin, English",
        "High confidence. Clues: The first sentence uses Chinese characters and means 'The weather is very good today'; the second sentence is standard English.",
    ),
    (
        "Grazie mille! That was a fantastico dinner.",
        "Italian, English",
        "High confidence. Clues: 'Grazie mille' is Italian for 'thank you very much'; 'fantastico' is Italian for fantastic. The rest is English with an Italian loanword.",
    ),
    (
        "Спасибо за помощь. I really appreciate it.",
        "Russian, English",
        "High confidence. Clues: 'Спасибо за помощь' uses Cyrillic script and is Russian for 'Thank you for your help'; the second sentence is standard English.",
    ),
    (
        "Olá! Como vai? I am doing well, obrigado.",
        "Portuguese, English",
        "High confidence. Clues: 'Olá' is Portuguese for hello; 'Como vai?' is Portuguese for 'How are you?'; 'obrigado' is Portuguese for thank you. The middle phrase is English.",
    ),
    (
        "안녕하세요! Nice to meet you. 잘 부탁드립니다.",
        "Korean, English",
        "High confidence. Clues: '안녕하세요' uses Hangul script and is Korean for 'Hello'; '잘 부탁드립니다' is a Korean polite expression meaning 'I look forward to working with you'. The middle phrase is English.",
    ),
    (
        "Merhaba! The bazaar was çok güzel today.",
        "Turkish, English",
        "High confidence. Clues: 'Merhaba' is Turkish for hello; 'çok güzel' is Turkish for 'very beautiful'. The rest is English with Turkish words embedded.",
    ),
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


def generate_translation_pairs_records(n: int) -> list[dict]:
    """Generate translation_pairs records."""
    records = []
    for i in range(n):
        entry = TRANSLATION_PAIRS_DATA[i % len(TRANSLATION_PAIRS_DATA)]
        english, lang1, trans1, lang2, trans2, notes = entry
        uid_src = f"tp_{i}_{lang1}_{lang2}_{english[:20]}"
        input_text = (
            f"Translate the following English sentence into {lang1} and {lang2}:\n\n\"{english}\""
        )
        output_text = (
            f"{lang1}: {trans1}\n"
            f"{lang2}: {trans2}\n"
            f"Notes: {notes}"
        )
        records.append({
            "record_id": _make_id(uid_src),
            "category": "translation_pairs",
            "source_lang": "English",
            "target_lang": f"{lang1}, {lang2}",
            "input_text": input_text,
            "output_text": output_text,
            "notes": notes,
            "created_at": _now(),
        })
    return records


def generate_false_friends_records(n: int) -> list[dict]:
    """Generate false_friends records."""
    records = []
    for i in range(n):
        entry = FALSE_FRIENDS_DATA[i % len(FALSE_FRIENDS_DATA)]
        word_pair, lang1, lang2, explanation = entry
        uid_src = f"ff_{i}_{lang1}_{lang2}_{word_pair[:20]}"
        input_text = (
            f"Explain the false friend relationship between: {word_pair}"
        )
        output_text = explanation
        records.append({
            "record_id": _make_id(uid_src),
            "category": "false_friends",
            "source_lang": lang1,
            "target_lang": lang2,
            "input_text": input_text,
            "output_text": output_text,
            "notes": f"False friend pair: {word_pair}",
            "created_at": _now(),
        })
    return records


def generate_idiom_translation_records(n: int) -> list[dict]:
    """Generate idiom_translation records."""
    records = []
    for i in range(n):
        entry = IDIOM_TRANSLATION_DATA[i % len(IDIOM_TRANSLATION_DATA)]
        idiom, lang1, equiv1, lang2, equiv2, meaning = entry
        uid_src = f"it_{i}_{lang1}_{lang2}_{idiom[:20]}"
        input_text = (
            f"Translate the English idiom '{idiom}' into {lang1} and {lang2}. "
            f"Include the literal translation and the meaning."
        )
        output_text = (
            f"{lang1} equivalent: {equiv1}\n"
            f"{lang2} equivalent: {equiv2}\n"
            f"{meaning}"
        )
        records.append({
            "record_id": _make_id(uid_src),
            "category": "idiom_translation",
            "source_lang": "English",
            "target_lang": f"{lang1}, {lang2}",
            "input_text": input_text,
            "output_text": output_text,
            "notes": meaning,
            "created_at": _now(),
        })
    return records


def generate_cultural_context_records(n: int) -> list[dict]:
    """Generate cultural_context records."""
    records = []
    for i in range(n):
        entry = CULTURAL_CONTEXT_DATA[i % len(CULTURAL_CONTEXT_DATA)]
        source_lang, target_lang, phrase, explanation = entry
        uid_src = f"cc_{i}_{source_lang}_{target_lang}_{phrase[:20]}"
        input_text = (
            f"Translate the {source_lang} phrase '{phrase}' into {target_lang} "
            f"and explain the cultural context required for an accurate translation."
        )
        output_text = explanation
        records.append({
            "record_id": _make_id(uid_src),
            "category": "cultural_context",
            "source_lang": source_lang,
            "target_lang": target_lang,
            "input_text": input_text,
            "output_text": output_text,
            "notes": f"Phrase: {phrase}",
            "created_at": _now(),
        })
    return records


def generate_language_detection_records(n: int) -> list[dict]:
    """Generate language_detection records."""
    records = []
    for i in range(n):
        entry = LANGUAGE_DETECTION_DATA[i % len(LANGUAGE_DETECTION_DATA)]
        text, languages, explanation = entry
        uid_src = f"ld_{i}_{languages}_{text[:20]}"
        input_text = (
            f"Identify the languages present in the following text and explain the clues:\n\n\"{text}\""
        )
        output_text = (
            f"Languages detected: {languages}\n"
            f"Confidence: {explanation}"
        )
        records.append({
            "record_id": _make_id(uid_src),
            "category": "language_detection",
            "source_lang": "mixed",
            "target_lang": languages,
            "input_text": input_text,
            "output_text": output_text,
            "notes": explanation,
            "created_at": _now(),
        })
    return records


# ---------------------------------------------------------------------------
# Database and JSONL builders
# ---------------------------------------------------------------------------

def _create_table(conn: sqlite3.Connection) -> None:
    """Create the multilingual_records table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS multilingual_records (
            record_id   TEXT PRIMARY KEY,
            category    TEXT,
            source_lang TEXT,
            target_lang TEXT,
            input_text  TEXT,
            output_text TEXT,
            notes       TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()


def _insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records into multilingual_records, ignoring duplicates."""
    sql = """
        INSERT OR IGNORE INTO multilingual_records
            (record_id, category, source_lang, target_lang, input_text, output_text, notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (r["record_id"], r["category"], r["source_lang"], r["target_lang"],
         r["input_text"], r["output_text"], r["notes"], r["created_at"])
        for r in records
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def build_database(n_per_category: int = 300) -> None:
    """
    Build the multilingual SQLite database and JSONL file.

    Parameters
    ----------
    n_per_category : int
        Number of records to generate per category (default 300).
    """
    random.seed(55)
    all_records: list[dict] = []
    all_records.extend(generate_translation_pairs_records(n_per_category))
    all_records.extend(generate_false_friends_records(n_per_category))
    all_records.extend(generate_idiom_translation_records(n_per_category))
    all_records.extend(generate_cultural_context_records(n_per_category))
    all_records.extend(generate_language_detection_records(n_per_category))

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
                    "source_lang": r["source_lang"],
                    "target_lang": r["target_lang"],
                    "source": "janus_multilingual_v1",
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        print(f"[ERROR] Failed to write JSONL: {exc}")

    total = len(all_records)
    print(f"Multilingual DB : {total} records inserted={inserted}")
    print(f"JSONL lines     : {written}")
    print(f"DB path         : {DB_PATH}")
    print(f"JSONL path      : {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=400)
