"""
janus_comprehension_db.py
--------------------------
Generates reading comprehension training data for the Janus NLP pipeline.
Produces a SQLite .db file and a .jsonl file when run directly.

Categories:
  - passage_qa
  - inference
  - main_idea
  - summarisation
  - fact_vs_opinion
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "janus_comprehension.db")
JSONL_PATH = os.path.join(os.path.dirname(__file__), "janus_comprehension.jsonl")

random.seed(44)

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

PASSAGE_QA_DATA = [
    (
        "science",
        "The human body contains approximately 37 trillion cells. Each cell performs specific functions that keep the body alive. Red blood cells carry oxygen, while white blood cells fight infection. Nerve cells transmit signals throughout the body. Without this cellular cooperation, life would not be possible.",
        "What is the function of red blood cells?",
        "Red blood cells carry oxygen throughout the body, as stated in the passage.",
        "beginner",
    ),
    (
        "history",
        "The printing press was invented by Johannes Gutenberg around 1440. It revolutionised the spread of information by making books affordable and widely available. Before its invention, books were copied by hand, a slow and expensive process. The printing press played a key role in the Renaissance and the Reformation.",
        "How did the printing press change the availability of books?",
        "The printing press made books affordable and widely available, whereas previously they had to be copied by hand, which was slow and expensive.",
        "intermediate",
    ),
    (
        "technology",
        "Artificial intelligence refers to the simulation of human intelligence in machines. AI systems can learn from data, recognise patterns, and make decisions. Applications include voice assistants, medical diagnosis, and autonomous vehicles. The field has grown rapidly since the development of deep learning algorithms.",
        "What are three applications of artificial intelligence mentioned in the passage?",
        "The passage mentions voice assistants, medical diagnosis, and autonomous vehicles as applications of artificial intelligence.",
        "intermediate",
    ),
    (
        "nature",
        "Coral reefs are among the most biodiverse ecosystems on Earth. They cover less than one percent of the ocean floor but support approximately 25 percent of all marine species. Reefs are threatened by rising ocean temperatures, pollution, and overfishing. Scientists warn that without intervention, many reefs could disappear within decades.",
        "Why are coral reefs considered important despite their small size?",
        "Although coral reefs cover less than one percent of the ocean floor, they support approximately 25 percent of all marine species, making them critically important for marine biodiversity.",
        "intermediate",
    ),
    (
        "business",
        "Supply chain disruptions during the pandemic exposed vulnerabilities in global manufacturing. Many companies relied on single-source suppliers, leaving them unable to adapt when those suppliers were affected. As a result, businesses began diversifying their supply chains and increasing local production. This shift has had lasting effects on global trade patterns.",
        "What lesson did businesses learn from pandemic-era supply chain disruptions?",
        "Businesses learned that relying on single-source suppliers creates vulnerability. They responded by diversifying supply chains and increasing local production.",
        "advanced",
    ),
    (
        "culture",
        "The concept of time varies significantly across cultures. In many Western societies, punctuality is considered a sign of respect and professionalism. In contrast, some cultures in Latin America and the Middle East have a more flexible attitude towards time, where relationships take priority over schedules. These differences can lead to misunderstandings in international business settings.",
        "How do attitudes towards time differ between Western and some non-Western cultures?",
        "Western cultures often treat punctuality as a sign of respect, while some Latin American and Middle Eastern cultures prioritise relationships over strict schedules.",
        "advanced",
    ),
    (
        "science",
        "Photosynthesis is the process by which plants convert sunlight into food. Using chlorophyll in their leaves, plants absorb carbon dioxide from the air and water from the soil. Sunlight provides the energy needed to convert these into glucose and oxygen. The oxygen is released into the atmosphere, making photosynthesis essential for life on Earth.",
        "What raw materials do plants use in photosynthesis?",
        "Plants use carbon dioxide from the air and water from the soil as raw materials in photosynthesis.",
        "beginner",
    ),
    (
        "history",
        "The Industrial Revolution began in Britain in the late 18th century and transformed manufacturing from hand production to machine-based processes. Steam power enabled factories to produce goods on a massive scale. Urbanisation accelerated as workers moved from rural areas to cities. The revolution fundamentally changed economic and social structures worldwide.",
        "What role did steam power play in the Industrial Revolution?",
        "Steam power enabled factories to produce goods on a massive scale, which was central to the transformation of manufacturing during the Industrial Revolution.",
        "intermediate",
    ),
    (
        "technology",
        "Blockchain is a distributed ledger technology that records transactions across multiple computers. Each transaction is grouped into a block and linked to the previous one, forming a chain. This structure makes it extremely difficult to alter past records. Blockchain is the foundation of cryptocurrencies like Bitcoin and has applications in supply chain management and healthcare.",
        "Why is blockchain considered secure?",
        "Blockchain is considered secure because each transaction is linked to the previous one in a chain, making it extremely difficult to alter past records.",
        "advanced",
    ),
    (
        "nature",
        "Migratory birds travel thousands of kilometres each year between their breeding and wintering grounds. They navigate using the Earth's magnetic field, the position of the sun, and star patterns. Some species, like the Arctic tern, travel from the Arctic to the Antarctic and back — a round trip of approximately 70,000 kilometres.",
        "How do migratory birds navigate during their journeys?",
        "Migratory birds navigate using the Earth's magnetic field, the position of the sun, and star patterns.",
        "intermediate",
    ),
]

INFERENCE_DATA = [
    (
        "science",
        "Maria arrived at the laboratory at 7 a.m., before any of her colleagues. She immediately put on her lab coat and safety goggles, checked her experiment notes, and began preparing her samples. By the time her supervisor arrived at 9 a.m., she had already completed the first stage of the experiment.",
        "What can we infer about Maria's attitude towards her work?",
        "We can infer that Maria is highly dedicated and diligent. Her early arrival, immediate preparation, and rapid progress suggest she is motivated and takes her work seriously.",
        "intermediate",
    ),
    (
        "business",
        "The restaurant had been fully booked every weekend for the past three months. The owner had recently hired two additional chefs and extended the opening hours. A local food critic had given the restaurant a glowing five-star review the previous month.",
        "What can we infer about the restaurant's recent performance?",
        "We can infer that the restaurant has become very popular and successful, likely as a result of the positive review. The full bookings and expansion of staff and hours suggest strong demand.",
        "intermediate",
    ),
    (
        "history",
        "After the treaty was signed, trade between the two nations increased dramatically. Merchants who had previously avoided the border region began establishing businesses there. Within five years, a new town had grown up around the old checkpoint.",
        "What can we infer about the relationship between the two nations before the treaty?",
        "We can infer that relations between the two nations were tense or hostile before the treaty, as merchants avoided the border region and trade was limited.",
        "advanced",
    ),
    (
        "culture",
        "When the new employee greeted her colleagues with a bow, some of them looked surprised and exchanged glances. She noticed their reaction and quickly extended her hand for a handshake instead.",
        "What can we infer about the cultural background of the new employee and her colleagues?",
        "We can infer that the new employee comes from a culture where bowing is a common greeting, while her colleagues are from a culture where handshakes are the norm. Her quick adaptation suggests she is socially aware.",
        "advanced",
    ),
    (
        "nature",
        "The farmer noticed that the leaves on his crops had turned yellow and the soil felt unusually dry despite recent rainfall. He called an agricultural expert, who arrived the following morning with testing equipment.",
        "What can we infer about the likely cause of the crops' condition?",
        "We can infer that the crops may be suffering from a nutrient deficiency or a drainage problem, since the soil is dry despite rainfall. The farmer's urgency in calling an expert suggests the situation is serious.",
        "advanced",
    ),
    (
        "technology",
        "The software update was released on a Tuesday. By Wednesday morning, the company's support inbox had received over 2,000 messages. The development team worked through the night.",
        "What can we infer about the software update?",
        "We can infer that the software update contained a significant bug or caused problems for users, prompting a large volume of complaints and requiring urgent attention from the development team.",
        "intermediate",
    ),
    (
        "science",
        "The patient had not visited a doctor in over ten years. When she finally went for a check-up, the doctor ordered a series of tests and asked her to return the following week.",
        "What can we infer about the doctor's findings during the check-up?",
        "We can infer that the doctor found something that warranted further investigation, as ordering multiple tests and scheduling a follow-up suggests concern about the patient's health.",
        "intermediate",
    ),
    (
        "business",
        "The company's share price fell by 18% on the day the CEO announced his resignation. Trading volume was three times higher than usual.",
        "What can we infer about investor confidence in the company?",
        "We can infer that investors had significant confidence in the CEO and were concerned about the company's future without him, as evidenced by the sharp drop in share price and high trading volume.",
        "advanced",
    ),
]

MAIN_IDEA_DATA = [
    (
        "science",
        "Sleep is essential for physical and mental health. During sleep, the body repairs tissues, consolidates memories, and regulates hormones. Chronic sleep deprivation has been linked to obesity, heart disease, and depression. Despite this, many people in modern societies regularly sleep fewer than the recommended seven to nine hours per night.",
        "The main idea is that sleep is vital for health, yet many people do not get enough of it. Supporting details include the body's repair processes during sleep and the health risks associated with sleep deprivation.",
        "intermediate",
    ),
    (
        "technology",
        "Social media platforms have transformed how people communicate and consume information. They allow instant sharing of news, opinions, and personal updates across the globe. However, they have also been linked to the spread of misinformation, increased anxiety among young users, and the erosion of privacy. The benefits and drawbacks of social media continue to be debated.",
        "The main idea is that social media has both transformed communication and introduced significant social challenges. Supporting details include global connectivity, misinformation, mental health concerns, and privacy issues.",
        "intermediate",
    ),
    (
        "history",
        "The Roman Empire's decline was the result of multiple interconnected factors. Military overextension left borders vulnerable to invasion. Economic troubles, including inflation and heavy taxation, weakened the state. Political instability, with frequent changes of emperor, undermined governance. The combination of these pressures eventually led to the fall of the Western Roman Empire in 476 AD.",
        "The main idea is that the Roman Empire fell due to a combination of military, economic, and political factors rather than any single cause. Supporting details include overextension, inflation, taxation, and political instability.",
        "advanced",
    ),
    (
        "nature",
        "Deforestation is one of the most pressing environmental issues of our time. Forests are cleared for agriculture, logging, and urban development, destroying habitats and reducing biodiversity. Trees also absorb carbon dioxide, so their removal accelerates climate change. Reforestation efforts are underway in many countries, but they cannot keep pace with the rate of destruction.",
        "The main idea is that deforestation poses serious environmental threats, including habitat loss and climate change, and that current restoration efforts are insufficient. Supporting details include causes of deforestation and the role of trees in carbon absorption.",
        "intermediate",
    ),
    (
        "business",
        "Remote work has become a permanent feature of many industries following the pandemic. Employees report higher job satisfaction and better work-life balance when working from home. However, employers have raised concerns about productivity, collaboration, and company culture. Many organisations have adopted hybrid models that combine remote and in-office work.",
        "The main idea is that remote work offers benefits for employees but presents challenges for employers, leading many organisations to adopt hybrid arrangements. Supporting details include job satisfaction, productivity concerns, and the rise of hybrid models.",
        "intermediate",
    ),
    (
        "culture",
        "Languages are disappearing at an alarming rate. Of the approximately 7,000 languages spoken today, linguists estimate that half will be extinct by the end of the century. When a language dies, unique cultural knowledge, oral traditions, and ways of understanding the world are lost forever. Efforts to document and revitalise endangered languages are growing but remain underfunded.",
        "The main idea is that language extinction is a serious cultural loss, and that preservation efforts are insufficient. Supporting details include the rate of language loss, the cultural knowledge embedded in languages, and the state of revitalisation efforts.",
        "advanced",
    ),
]

SUMMARISATION_DATA = [
    (
        "science",
        "The Amazon rainforest, often called the 'lungs of the Earth', produces approximately 20 percent of the world's oxygen. It is home to an estimated 10 percent of all species on the planet, many of which have not yet been discovered. The forest also plays a critical role in regulating the global water cycle by releasing vast amounts of water vapour into the atmosphere. However, deforestation driven by agriculture, logging, and mining has destroyed significant portions of the forest. Scientists warn that if deforestation continues at its current rate, the Amazon could reach a tipping point beyond which it can no longer sustain itself. International efforts to protect the forest have had limited success due to economic pressures and political challenges.",
        "The Amazon rainforest is vital to global oxygen production, biodiversity, and the water cycle, but ongoing deforestation threatens to push it past a point of no return, with international protection efforts proving insufficient.",
        "advanced",
    ),
    (
        "history",
        "The Cold War was a period of geopolitical tension between the United States and the Soviet Union that lasted from the end of World War II until the dissolution of the Soviet Union in 1991. Although the two superpowers never engaged in direct military conflict, they competed through proxy wars, an arms race, and a space race. The ideological divide between capitalism and communism shaped global politics for decades. Key events included the Korean War, the Cuban Missile Crisis, and the Vietnam War. The fall of the Berlin Wall in 1989 symbolised the beginning of the end of the Cold War.",
        "The Cold War was a decades-long ideological and geopolitical rivalry between the US and USSR that shaped global politics through proxy conflicts and an arms race, ending with the Soviet Union's collapse in 1991.",
        "advanced",
    ),
    (
        "technology",
        "Electric vehicles (EVs) are becoming increasingly popular as governments and consumers seek to reduce carbon emissions. Advances in battery technology have extended the range of EVs, addressing one of the main concerns of potential buyers. Charging infrastructure is expanding rapidly in many countries, though gaps remain in rural areas. The cost of EVs has fallen significantly over the past decade, making them more accessible. However, concerns remain about the environmental impact of battery production and the source of electricity used to charge them. Major car manufacturers have committed to phasing out petrol and diesel vehicles within the next two decades.",
        "Electric vehicles are gaining popularity due to improved range, falling costs, and expanding infrastructure, though concerns about battery production and electricity sources remain as manufacturers commit to phasing out combustion engines.",
        "advanced",
    ),
    (
        "nature",
        "Bees play an indispensable role in global food production through pollination. Approximately one third of the food humans consume depends on bee pollination, including fruits, vegetables, and nuts. Bee populations have been declining sharply due to habitat loss, pesticide use, disease, and climate change. The economic value of pollination services provided by bees is estimated at hundreds of billions of dollars annually. Efforts to protect bees include reducing pesticide use, planting wildflower habitats, and supporting organic farming. Without intervention, the decline of bee populations could have severe consequences for food security worldwide.",
        "Bees are essential pollinators responsible for a third of human food production, but their populations are declining due to multiple threats, with serious implications for global food security if protective measures are not taken.",
        "intermediate",
    ),
    (
        "business",
        "The gig economy has grown rapidly over the past decade, with platforms like Uber, Deliveroo, and Fiverr connecting workers with short-term tasks. Proponents argue that gig work offers flexibility and autonomy that traditional employment does not. Critics, however, point out that gig workers often lack job security, benefits such as sick pay and pensions, and legal protections. Several countries have introduced legislation to improve conditions for gig workers, with varying degrees of success. The debate reflects broader questions about the future of work and the balance between flexibility and security.",
        "The gig economy offers workers flexibility but lacks the security and benefits of traditional employment, prompting legislative responses that reflect ongoing debates about the future of work.",
        "intermediate",
    ),
]

FACT_VS_OPINION_DATA = [
    (
        "science",
        "The Earth is approximately 4.5 billion years old. Scientists believe this is the most fascinating period in the planet's history. Carbon dating is used to determine the age of organic materials. Some researchers think that life on Earth began much earlier than previously thought. The oldest known fossils are approximately 3.5 billion years old.",
        "Facts: The Earth is approximately 4.5 billion years old (scientific consensus); carbon dating is used to determine the age of organic materials (established method); the oldest known fossils are approximately 3.5 billion years old (scientific finding). Opinions: Scientists believe this is the most fascinating period in the planet's history (subjective judgment); some researchers think life began earlier than previously thought (speculative claim, not yet established).",
        "intermediate",
    ),
    (
        "technology",
        "Smartphones were first introduced in the late 1990s. They are undoubtedly the greatest invention of the 20th century. The iPhone was released by Apple in 2007. Social media apps have made smartphones indispensable. Over 6 billion people worldwide now own a smartphone.",
        "Facts: Smartphones were first introduced in the late 1990s; the iPhone was released by Apple in 2007; over 6 billion people worldwide own a smartphone. Opinions: They are undoubtedly the greatest invention of the 20th century (subjective superlative); social media apps have made smartphones indispensable (debatable claim).",
        "beginner",
    ),
    (
        "history",
        "World War II ended in 1945. It was the most devastating conflict in human history. Approximately 70 to 85 million people died as a result of the war. The Holocaust resulted in the systematic murder of six million Jewish people. The war should never be forgotten.",
        "Facts: World War II ended in 1945; approximately 70 to 85 million people died; the Holocaust resulted in the murder of six million Jewish people. Opinions: It was the most devastating conflict in human history (while widely supported, 'most devastating' involves a value judgment); the war should never be forgotten (a moral imperative, not a factual claim).",
        "intermediate",
    ),
    (
        "business",
        "Amazon was founded by Jeff Bezos in 1994. It is the best company to work for in the technology sector. Amazon employs over 1.5 million people worldwide. The company's treatment of warehouse workers has been widely criticised. Every consumer should support ethical businesses.",
        "Facts: Amazon was founded by Jeff Bezos in 1994; Amazon employs over 1.5 million people worldwide; the company's treatment of warehouse workers has been widely criticised (documented fact). Opinions: It is the best company to work for in the technology sector (subjective ranking); every consumer should support ethical businesses (normative claim).",
        "intermediate",
    ),
    (
        "culture",
        "The Eiffel Tower was built in 1889 for the World's Fair in Paris. It is the most beautiful structure ever built. The tower attracts approximately 7 million visitors per year. French culture is the most sophisticated in the world. The tower was initially criticised by many Parisians.",
        "Facts: The Eiffel Tower was built in 1889 for the World's Fair; it attracts approximately 7 million visitors per year; it was initially criticised by many Parisians. Opinions: It is the most beautiful structure ever built (subjective aesthetic judgment); French culture is the most sophisticated in the world (cultural bias and subjective claim).",
        "beginner",
    ),
    (
        "nature",
        "Climate change is caused primarily by human activities, according to the scientific consensus. Rising global temperatures are the most urgent problem facing humanity. Sea levels have risen by approximately 20 centimetres since 1900. Governments are not doing enough to address climate change. The Arctic is warming at twice the global average rate.",
        "Facts: Climate change is caused primarily by human activities (scientific consensus); sea levels have risen by approximately 20 centimetres since 1900; the Arctic is warming at twice the global average rate. Opinions: Rising global temperatures are the most urgent problem facing humanity (prioritisation judgment); governments are not doing enough (evaluative claim that depends on standards).",
        "advanced",
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


def generate_passage_qa_records(n: int) -> list[dict]:
    """Generate passage_qa records."""
    records = []
    for i in range(n):
        entry = PASSAGE_QA_DATA[i % len(PASSAGE_QA_DATA)]
        topic, passage, question, answer, difficulty = entry
        uid_src = f"pqa_{i}_{topic}_{passage[:20]}"
        input_text = f"Read the following passage and answer the question.\n\nPassage: {passage}\n\nQuestion: {question}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "passage_qa",
            "topic": topic,
            "passage": passage,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "created_at": _now(),
        })
    return records


def generate_inference_records(n: int) -> list[dict]:
    """Generate inference records."""
    records = []
    for i in range(n):
        entry = INFERENCE_DATA[i % len(INFERENCE_DATA)]
        topic, passage, question, answer, difficulty = entry
        uid_src = f"inf_{i}_{topic}_{passage[:20]}"
        input_text = f"Read the following passage and make an inference to answer the question.\n\nPassage: {passage}\n\nQuestion: {question}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "inference",
            "topic": topic,
            "passage": passage,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "created_at": _now(),
        })
    return records


def generate_main_idea_records(n: int) -> list[dict]:
    """Generate main_idea records."""
    records = []
    for i in range(n):
        entry = MAIN_IDEA_DATA[i % len(MAIN_IDEA_DATA)]
        topic, passage, answer, difficulty = entry
        uid_src = f"mi_{i}_{topic}_{passage[:20]}"
        question = "What is the main idea of this passage? Include supporting details."
        input_text = f"Read the following passage and identify the main idea.\n\nPassage: {passage}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "main_idea",
            "topic": topic,
            "passage": passage,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "created_at": _now(),
        })
    return records


def generate_summarisation_records(n: int) -> list[dict]:
    """Generate summarisation records."""
    records = []
    for i in range(n):
        entry = SUMMARISATION_DATA[i % len(SUMMARISATION_DATA)]
        topic, passage, summary, difficulty = entry
        uid_src = f"sum_{i}_{topic}_{passage[:20]}"
        question = "Summarise this passage in 1-2 sentences."
        input_text = f"Read the following passage and write a concise summary.\n\nPassage: {passage}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "summarisation",
            "topic": topic,
            "passage": passage,
            "question": question,
            "answer": summary,
            "difficulty": difficulty,
            "created_at": _now(),
        })
    return records


def generate_fact_vs_opinion_records(n: int) -> list[dict]:
    """Generate fact_vs_opinion records."""
    records = []
    for i in range(n):
        entry = FACT_VS_OPINION_DATA[i % len(FACT_VS_OPINION_DATA)]
        topic, passage, answer, difficulty = entry
        uid_src = f"fvo_{i}_{topic}_{passage[:20]}"
        question = "Identify which statements in this passage are facts and which are opinions. Justify your answers."
        input_text = f"Read the following passage and distinguish facts from opinions.\n\nPassage: {passage}"
        records.append({
            "record_id": _make_id(uid_src),
            "category": "fact_vs_opinion",
            "topic": topic,
            "passage": passage,
            "question": question,
            "answer": answer,
            "difficulty": difficulty,
            "created_at": _now(),
        })
    return records


# ---------------------------------------------------------------------------
# Database and JSONL builders
# ---------------------------------------------------------------------------

def _create_table(conn: sqlite3.Connection) -> None:
    """Create the comprehension_records table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS comprehension_records (
            record_id  TEXT PRIMARY KEY,
            category   TEXT,
            topic      TEXT,
            passage    TEXT,
            question   TEXT,
            answer     TEXT,
            difficulty TEXT,
            created_at TEXT
        )
    """)
    conn.commit()


def _insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records into comprehension_records, ignoring duplicates."""
    sql = """
        INSERT OR IGNORE INTO comprehension_records
            (record_id, category, topic, passage, question, answer, difficulty, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (r["record_id"], r["category"], r["topic"], r["passage"],
         r["question"], r["answer"], r["difficulty"], r["created_at"])
        for r in records
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def build_database(n_per_category: int = 300) -> None:
    """
    Build the comprehension SQLite database and JSONL file.

    Parameters
    ----------
    n_per_category : int
        Number of records to generate per category (default 300).
    """
    random.seed(44)
    all_records: list[dict] = []
    all_records.extend(generate_passage_qa_records(n_per_category))
    all_records.extend(generate_inference_records(n_per_category))
    all_records.extend(generate_main_idea_records(n_per_category))
    all_records.extend(generate_summarisation_records(n_per_category))
    all_records.extend(generate_fact_vs_opinion_records(n_per_category))

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
                    "instruction": r["question"],
                    "response": r["answer"],
                    "category": r["category"],
                    "topic": r["topic"],
                    "difficulty": r["difficulty"],
                    "source": "janus_comprehension_v1",
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        print(f"[ERROR] Failed to write JSONL: {exc}")

    total = len(all_records)
    print(f"Comprehension DB : {total} records inserted={inserted}")
    print(f"JSONL lines      : {written}")
    print(f"DB path          : {DB_PATH}")
    print(f"JSONL path       : {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=400)
