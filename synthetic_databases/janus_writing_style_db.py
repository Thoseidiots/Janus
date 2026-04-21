"""
janus_writing_style_db.py
--------------------------
Generates writing style and tone training data for the Janus NLP pipeline.
Produces a SQLite .db file and a .jsonl file when run directly.

Categories:
  - tone_rewrite
  - style_analysis
  - clarity_improvement
  - audience_adaptation
  - register_shift
"""

import hashlib
import json
import os
import random
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "janus_writing_style.db")
JSONL_PATH = os.path.join(os.path.dirname(__file__), "janus_writing_style.jsonl")

random.seed(33)

# ---------------------------------------------------------------------------
# Data pools
# ---------------------------------------------------------------------------

TONE_REWRITE_DATA = [
    (
        "We need to fix the bug in the system before the deadline.",
        "formal",
        "It is imperative that the identified system defect be resolved prior to the stipulated deadline.",
        "Formal tone uses passive constructions, precise vocabulary, and avoids contractions.",
    ),
    (
        "We need to fix the bug in the system before the deadline.",
        "casual",
        "Hey, we've got to sort out that bug before we run out of time!",
        "Casual tone uses contractions, informal vocabulary, and a conversational feel.",
    ),
    (
        "We need to fix the bug in the system before the deadline.",
        "persuasive",
        "Fixing this critical bug before the deadline is not just important — it is essential to maintaining our reputation and delivering on our promise to clients.",
        "Persuasive tone uses strong language, appeals to consequences, and motivates action.",
    ),
    (
        "The employee made an error in the financial report.",
        "empathetic",
        "We understand that mistakes can happen, and we appreciate the effort that went into the financial report. Let's work together to correct the error and move forward.",
        "Empathetic tone acknowledges feelings, avoids blame, and focuses on collaboration.",
    ),
    (
        "The employee made an error in the financial report.",
        "technical",
        "A data entry discrepancy was identified in the Q3 financial report, resulting in a variance of approximately 3.2% in the net revenue figures.",
        "Technical tone uses precise terminology, quantifies issues, and avoids emotional language.",
    ),
    (
        "The employee made an error in the financial report.",
        "humorous",
        "Looks like our financial report decided to go rogue and invent its own numbers — time to have a little chat with the spreadsheet!",
        "Humorous tone uses wit, light-hearted language, and avoids taking the situation too seriously.",
    ),
    (
        "The project is behind schedule.",
        "formal",
        "Regrettably, the project has not progressed in accordance with the established timeline.",
        "Formal tone uses hedging language and avoids direct blame.",
    ),
    (
        "The project is behind schedule.",
        "casual",
        "So, we're a bit behind on the project — no big deal, we'll catch up!",
        "Casual tone downplays the issue and uses informal phrasing.",
    ),
    (
        "The project is behind schedule.",
        "persuasive",
        "While the project has encountered delays, this is precisely the moment to demonstrate our resilience and commitment by redoubling our efforts.",
        "Persuasive tone reframes the negative as an opportunity.",
    ),
    (
        "The new policy will affect all employees.",
        "formal",
        "The newly implemented policy will have implications for all members of staff across the organisation.",
        "Formal tone uses 'members of staff' and 'organisation' for a professional register.",
    ),
    (
        "The new policy will affect all employees.",
        "empathetic",
        "We know that changes like this can feel unsettling, and we want to assure every team member that we are here to support you through this transition.",
        "Empathetic tone acknowledges emotional impact and offers reassurance.",
    ),
    (
        "The new policy will affect all employees.",
        "technical",
        "The revised policy framework will be applied uniformly across all personnel classifications, effective from the date of implementation.",
        "Technical tone uses precise, policy-oriented language.",
    ),
    (
        "Sales have dropped this quarter.",
        "persuasive",
        "Although this quarter's figures present a challenge, they also reveal a clear opportunity to refine our strategy and emerge stronger in the next period.",
        "Persuasive tone reframes challenges as opportunities.",
    ),
    (
        "Sales have dropped this quarter.",
        "formal",
        "The organisation has observed a decline in sales revenue during the current fiscal quarter.",
        "Formal tone uses passive voice and precise financial terminology.",
    ),
    (
        "Sales have dropped this quarter.",
        "humorous",
        "Our sales figures seem to have taken a little holiday this quarter — time to bring them back from the beach!",
        "Humorous tone uses metaphor and light-hearted language to soften bad news.",
    ),
    (
        "Please submit your report by Friday.",
        "formal",
        "You are kindly requested to submit the aforementioned report no later than the close of business on Friday.",
        "Formal tone uses polite request forms and precise time references.",
    ),
    (
        "Please submit your report by Friday.",
        "casual",
        "Just make sure you get that report in by Friday, yeah?",
        "Casual tone uses informal phrasing and a relaxed tone.",
    ),
    (
        "Please submit your report by Friday.",
        "persuasive",
        "Submitting your report by Friday will ensure the team can move forward without delays — your contribution is vital to our collective success.",
        "Persuasive tone appeals to team benefit and individual importance.",
    ),
    (
        "The meeting has been cancelled.",
        "formal",
        "We wish to inform you that the scheduled meeting has been cancelled. Further communication regarding rescheduling will follow.",
        "Formal tone uses passive constructions and promises follow-up.",
    ),
    (
        "The meeting has been cancelled.",
        "casual",
        "Hey, just a heads-up — the meeting's off. We'll let you know when it's back on.",
        "Casual tone uses informal language and a friendly tone.",
    ),
]

STYLE_ANALYSIS_DATA = [
    (
        "The sun dipped below the horizon, painting the sky in shades of amber and rose. A gentle breeze stirred the leaves, carrying with it the scent of rain. In the distance, a lone bird called out, its song fading into the gathering dusk.",
        "Tone: reflective and serene. Voice: third-person omniscient, lyrical. Sentence variety: good — mix of long and medium sentences with varied structures. Vocabulary level: intermediate to advanced (amber, stirred, gathering dusk). Suggestions: The passage is evocative; consider adding a human element to ground the imagery.",
    ),
    (
        "You need to update the software immediately. Failure to do so will result in security vulnerabilities. Follow the steps below. Do not skip any steps.",
        "Tone: urgent and directive. Voice: second-person, imperative. Sentence variety: low — all sentences are short and declarative. Vocabulary level: basic to intermediate. Suggestions: Add transitional phrases to improve flow; consider explaining why each step matters to increase compliance.",
    ),
    (
        "I think maybe we could possibly consider looking into some potential options for perhaps improving the current situation, if that might be feasible.",
        "Tone: hesitant and uncertain. Voice: first-person, passive. Sentence variety: low — single long, hedged sentence. Vocabulary level: basic. Suggestions: Remove excessive hedging ('maybe', 'possibly', 'perhaps'); be direct about the recommendation.",
    ),
    (
        "The results clearly demonstrate that the intervention was effective. Participants showed a 34% improvement in outcomes compared to the control group. These findings are consistent with previous research.",
        "Tone: objective and authoritative. Voice: third-person, academic. Sentence variety: moderate — mix of complex and simple sentences. Vocabulary level: academic. Suggestions: The writing is clear and evidence-based; consider adding a brief discussion of limitations.",
    ),
    (
        "Our product is the best on the market. Customers love it. You will too. Don't miss out.",
        "Tone: promotional and assertive. Voice: second-person, direct. Sentence variety: low — all short, punchy sentences. Vocabulary level: basic. Suggestions: Support claims with evidence; vary sentence length to avoid a choppy feel.",
    ),
]

CLARITY_IMPROVEMENT_DATA = [
    (
        "Due to the fact that the weather conditions were not conducive to the carrying out of outdoor activities, the decision was made by the management team to postpone the event.",
        "Management postponed the event because the weather was unsuitable for outdoor activities.",
        "Removed 'due to the fact that' (wordy), converted passive voice to active, and eliminated redundant phrases.",
    ),
    (
        "It is important to note that in the event that you are unable to attend the meeting, it would be appreciated if you could make arrangements to send a representative on your behalf.",
        "If you cannot attend the meeting, please send a representative.",
        "Removed 'it is important to note that', 'in the event that', and 'it would be appreciated if'; used direct imperative.",
    ),
    (
        "The utilisation of excessive and unnecessary verbiage in written communications has the effect of rendering the intended message unclear and difficult for the reader to comprehend.",
        "Using too many words makes writing unclear and hard to understand.",
        "Replaced 'utilisation' with 'using', removed redundant 'excessive and unnecessary', simplified 'rendering the intended message unclear'.",
    ),
    (
        "At this point in time, we are currently in the process of conducting an investigation into the matter.",
        "We are currently investigating the matter.",
        "Removed 'at this point in time' (redundant with 'currently'), and 'in the process of conducting an investigation into' (wordy for 'investigating').",
    ),
    (
        "The reason why the project failed was because of the fact that there was a lack of sufficient resources available.",
        "The project failed because of insufficient resources.",
        "Removed 'the reason why...was because of the fact that' (circular), and 'available' (redundant).",
    ),
    (
        "In my personal opinion, I believe that the proposed solution is not without its merits.",
        "I believe the proposed solution has merit.",
        "Removed 'in my personal opinion' (redundant with 'I believe'), and converted double negative to positive.",
    ),
    (
        "There are a number of different factors that need to be taken into consideration when making a decision of this nature.",
        "Several factors must be considered when making this decision.",
        "Replaced 'there are a number of different factors' with 'several factors', and simplified the verb phrase.",
    ),
    (
        "It goes without saying that communication is of the utmost importance in any organisation.",
        "Communication is vital in any organisation.",
        "Removed 'it goes without saying that' (if it goes without saying, don't say it), and replaced 'of the utmost importance' with 'vital'.",
    ),
    (
        "The new system that has been implemented is one that is designed to facilitate the more efficient processing of customer requests.",
        "The new system is designed to process customer requests more efficiently.",
        "Removed relative clause redundancy and simplified the verb phrase.",
    ),
    (
        "We are writing to inform you that your application has been received by us and is currently under review by our team.",
        "We have received your application and are reviewing it.",
        "Converted passive to active voice and removed redundant phrases.",
    ),
]

AUDIENCE_ADAPTATION_DATA = [
    (
        "The mitochondria are organelles found in eukaryotic cells that generate most of the cell's supply of ATP through oxidative phosphorylation.",
        "expert", "beginner",
        "Mitochondria are tiny structures inside your cells that act like batteries, producing the energy your body needs to function.",
        "Replaced technical terms (organelles, eukaryotic, ATP, oxidative phosphorylation) with simple analogies and everyday language.",
    ),
    (
        "The stock market experienced significant volatility due to macroeconomic uncertainty and shifting investor sentiment.",
        "expert", "general public",
        "The stock market had a turbulent period as investors grew uncertain about the broader economy.",
        "Replaced 'volatility' with 'turbulent period', 'macroeconomic uncertainty' with 'uncertain about the broader economy', and 'investor sentiment' with 'investors grew uncertain'.",
    ),
    (
        "To install the software, execute the installer binary with elevated privileges and follow the on-screen prompts.",
        "technical", "non-technical",
        "To install the software, right-click the installer file, select 'Run as administrator', and follow the instructions on screen.",
        "Replaced 'execute the installer binary' with 'right-click the installer file', and 'elevated privileges' with 'Run as administrator'.",
    ),
    (
        "The defendant entered a plea of not guilty and requested a trial by jury.",
        "legal", "general public",
        "The accused said they did not commit the crime and asked for their case to be decided by a jury.",
        "Replaced legal terms with plain language equivalents.",
    ),
    (
        "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
        "adult", "child",
        "Plants are like little factories that use sunlight to make their own food, which they store as sugar.",
        "Used a simple analogy (factory) and replaced technical terms with child-friendly language.",
    ),
    (
        "The algorithm uses a greedy approach to find a locally optimal solution at each step.",
        "technical", "business",
        "The system makes the best available choice at each step to reach a good overall result quickly.",
        "Replaced 'algorithm', 'greedy approach', and 'locally optimal solution' with business-friendly language.",
    ),
    (
        "Inflation erodes purchasing power, reducing the real value of savings over time.",
        "economic", "teenager",
        "When prices go up (that's inflation), the money you've saved can buy less stuff than it used to.",
        "Used relatable language and a direct example relevant to a teenager's experience.",
    ),
    (
        "The patient presents with dyspnoea, tachycardia, and peripheral oedema.",
        "medical", "patient",
        "You are experiencing shortness of breath, a fast heartbeat, and swelling in your legs and feet.",
        "Replaced medical terminology with plain descriptions of symptoms.",
    ),
]

REGISTER_SHIFT_DATA = [
    (
        "Hey, just wanted to check in about that thing we talked about. Any updates?",
        "informal_email", "formal_email",
        "Dear [Name], I am writing to follow up on our previous discussion. Could you please provide an update at your earliest convenience? Kind regards, [Your Name]",
        "Formal email uses a salutation, complete sentences, polite request forms, and a sign-off.",
    ),
    (
        "I am writing to formally request your assistance in resolving the aforementioned matter at your earliest convenience.",
        "formal_email", "informal_email",
        "Hi, could you help me sort out that issue we mentioned? Thanks!",
        "Informal email removes formal phrases, uses contractions, and adopts a friendly tone.",
    ),
    (
        "The findings suggest a correlation between sleep deprivation and reduced cognitive performance.",
        "academic", "social_media",
        "Not getting enough sleep seriously messes with your brain — science confirms it! 😴🧠 #SleepMatters",
        "Social media uses casual language, emojis, and hashtags to engage a broad audience.",
    ),
    (
        "lol cant believe they actually did that, so random 😂",
        "social_media", "academic",
        "The decision in question was unexpected and appears to lack a clear rationale, warranting further analysis.",
        "Academic register uses formal vocabulary, avoids contractions and emojis, and maintains an objective tone.",
    ),
    (
        "The quarterly report indicates a 12% increase in revenue, driven primarily by growth in the Asia-Pacific region.",
        "business_report", "casual_conversation",
        "So we had a pretty good quarter — revenue went up by 12%, mostly because things are going really well in Asia-Pacific.",
        "Casual conversation uses simpler vocabulary, contractions, and a conversational structure.",
    ),
    (
        "Gonna grab some food, you in?",
        "casual_conversation", "formal_invitation",
        "I would like to extend an invitation to join me for a meal. Please let me know if you are available.",
        "Formal invitation uses complete sentences, polite phrasing, and avoids contractions.",
    ),
    (
        "The committee resolved to defer the matter pending further consultation with relevant stakeholders.",
        "formal_report", "plain_english",
        "The committee decided to wait before making a decision, so they can talk to the people involved first.",
        "Plain English avoids jargon ('defer', 'pending', 'stakeholders') and uses simple, direct language.",
    ),
    (
        "This product is absolutely amazing and you need it in your life right now!!!",
        "social_media_ad", "formal_advertisement",
        "This product offers exceptional quality and is designed to meet your needs effectively.",
        "Formal advertisement removes exclamation marks, hyperbole, and second-person urgency.",
    ),
    (
        "We regret to inform you that your application has been unsuccessful on this occasion.",
        "formal_rejection", "empathetic_informal",
        "We're really sorry to let you know that we won't be moving forward with your application this time. We hope you'll keep trying — you clearly have a lot to offer.",
        "Empathetic informal tone softens the rejection with encouragement and a warm, personal voice.",
    ),
    (
        "I dunno, maybe we should just like, try a different approach or something.",
        "casual_speech", "professional_suggestion",
        "I would recommend exploring an alternative approach to address this challenge more effectively.",
        "Professional suggestion replaces hedging and filler words with confident, direct language.",
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


def generate_tone_rewrite_records(n: int) -> list[dict]:
    """Generate tone_rewrite records."""
    records = []
    for i in range(n):
        entry = TONE_REWRITE_DATA[i % len(TONE_REWRITE_DATA)]
        original, tone, rewritten, explanation = entry
        uid_src = f"tr_{i}_{tone}_{original[:30]}"
        input_text = f"Rewrite the following paragraph in a {tone} tone:\n\n{original}"
        output_text = rewritten
        records.append({
            "record_id": _make_id(uid_src),
            "category": "tone_rewrite",
            "tone": tone,
            "input_text": input_text,
            "output_text": output_text,
            "explanation": explanation,
            "created_at": _now(),
        })
    return records


def generate_style_analysis_records(n: int) -> list[dict]:
    """Generate style_analysis records."""
    records = []
    for i in range(n):
        entry = STYLE_ANALYSIS_DATA[i % len(STYLE_ANALYSIS_DATA)]
        paragraph, analysis = entry
        uid_src = f"sa_{i}_{paragraph[:30]}"
        input_text = f"Analyse the writing style of the following paragraph:\n\n{paragraph}"
        output_text = analysis
        records.append({
            "record_id": _make_id(uid_src),
            "category": "style_analysis",
            "tone": "mixed",
            "input_text": input_text,
            "output_text": output_text,
            "explanation": "Style analysis covering tone, voice, sentence variety, vocabulary, and suggestions.",
            "created_at": _now(),
        })
    return records


def generate_clarity_improvement_records(n: int) -> list[dict]:
    """Generate clarity_improvement records."""
    records = []
    for i in range(n):
        entry = CLARITY_IMPROVEMENT_DATA[i % len(CLARITY_IMPROVEMENT_DATA)]
        verbose, clear, explanation = entry
        uid_src = f"ci_{i}_{verbose[:30]}"
        input_text = f"Improve the clarity of the following paragraph:\n\n{verbose}"
        output_text = clear
        records.append({
            "record_id": _make_id(uid_src),
            "category": "clarity_improvement",
            "tone": "clear",
            "input_text": input_text,
            "output_text": output_text,
            "explanation": explanation,
            "created_at": _now(),
        })
    return records


def generate_audience_adaptation_records(n: int) -> list[dict]:
    """Generate audience_adaptation records."""
    records = []
    for i in range(n):
        entry = AUDIENCE_ADAPTATION_DATA[i % len(AUDIENCE_ADAPTATION_DATA)]
        original, orig_audience, target_audience, adapted, explanation = entry
        uid_src = f"aa_{i}_{orig_audience}_{target_audience}_{original[:20]}"
        input_text = (
            f"Adapt the following text from {orig_audience} to {target_audience}:\n\n{original}"
        )
        output_text = adapted
        records.append({
            "record_id": _make_id(uid_src),
            "category": "audience_adaptation",
            "tone": f"{orig_audience}_to_{target_audience}",
            "input_text": input_text,
            "output_text": output_text,
            "explanation": explanation,
            "created_at": _now(),
        })
    return records


def generate_register_shift_records(n: int) -> list[dict]:
    """Generate register_shift records."""
    records = []
    for i in range(n):
        entry = REGISTER_SHIFT_DATA[i % len(REGISTER_SHIFT_DATA)]
        original, from_register, to_register, shifted, explanation = entry
        uid_src = f"rs_{i}_{from_register}_{to_register}_{original[:20]}"
        input_text = (
            f"Shift the register of the following text from {from_register} to {to_register}:\n\n{original}"
        )
        output_text = shifted
        records.append({
            "record_id": _make_id(uid_src),
            "category": "register_shift",
            "tone": f"{from_register}_to_{to_register}",
            "input_text": input_text,
            "output_text": output_text,
            "explanation": explanation,
            "created_at": _now(),
        })
    return records


# ---------------------------------------------------------------------------
# Database and JSONL builders
# ---------------------------------------------------------------------------

def _create_table(conn: sqlite3.Connection) -> None:
    """Create the style_records table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS style_records (
            record_id   TEXT PRIMARY KEY,
            category    TEXT,
            tone        TEXT,
            input_text  TEXT,
            output_text TEXT,
            explanation TEXT,
            created_at  TEXT
        )
    """)
    conn.commit()


def _insert_records(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records into style_records, ignoring duplicates."""
    sql = """
        INSERT OR IGNORE INTO style_records
            (record_id, category, tone, input_text, output_text, explanation, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    rows = [
        (r["record_id"], r["category"], r["tone"], r["input_text"],
         r["output_text"], r["explanation"], r["created_at"])
        for r in records
    ]
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)


def build_database(n_per_category: int = 300) -> None:
    """
    Build the writing style SQLite database and JSONL file.

    Parameters
    ----------
    n_per_category : int
        Number of records to generate per category (default 300).
    """
    random.seed(33)
    all_records: list[dict] = []
    all_records.extend(generate_tone_rewrite_records(n_per_category))
    all_records.extend(generate_style_analysis_records(n_per_category))
    all_records.extend(generate_clarity_improvement_records(n_per_category))
    all_records.extend(generate_audience_adaptation_records(n_per_category))
    all_records.extend(generate_register_shift_records(n_per_category))

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
                    "tone": r["tone"],
                    "source": "janus_writing_style_v1",
                }
                fh.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1
    except Exception as exc:
        print(f"[ERROR] Failed to write JSONL: {exc}")

    total = len(all_records)
    print(f"Writing Style DB : {total} records inserted={inserted}")
    print(f"JSONL lines      : {written}")
    print(f"DB path          : {DB_PATH}")
    print(f"JSONL path       : {JSONL_PATH}")


if __name__ == "__main__":
    build_database(n_per_category=400)
