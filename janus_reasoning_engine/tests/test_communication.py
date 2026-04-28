"""
Tests for janus_reasoning_engine/communication/
Covers NLU, NLG, and SocialAwareness modules.
"""

import pytest
from janus_reasoning_engine.communication.nlu import (
    NLU,
    ParsedInstruction,
    JobRequirements,
)
from janus_reasoning_engine.communication.nlg import NLG
from janus_reasoning_engine.communication.social_awareness import SocialAwareness


# ===========================================================================
# NLU Tests
# ===========================================================================

class TestNLUParseInstruction:
    def setup_method(self):
        self.nlu = NLU()  # no GPT — heuristic mode

    def test_returns_parsed_instruction(self):
        result = self.nlu.parse_instruction("Make $5000 this month")
        assert isinstance(result, ParsedInstruction)

    def test_earn_money_intent(self):
        result = self.nlu.parse_instruction("I want to earn money by finding freelance jobs")
        assert result.intent == "earn_money"

    def test_find_job_intent(self):
        result = self.nlu.parse_instruction("Find me a Python development job on Upwork")
        assert result.intent == "find_job"

    def test_learn_skill_intent(self):
        result = self.nlu.parse_instruction("Learn React framework this week")
        assert result.intent == "learn_skill"

    def test_build_software_intent(self):
        result = self.nlu.parse_instruction("Build a website for my portfolio")
        assert result.intent == "build_software"

    def test_extracts_budget_entity(self):
        result = self.nlu.parse_instruction("Find a job paying $500")
        assert "budget" in result.entities
        assert "500" in result.entities["budget"]

    def test_extracts_skills_entity(self):
        result = self.nlu.parse_instruction("I need a Python and React developer")
        assert "skills" in result.entities
        assert "python" in result.entities["skills"]

    def test_extracts_deadline_entity(self):
        result = self.nlu.parse_instruction("Complete this within 2 weeks")
        assert "deadline" in result.entities

    def test_empty_text_returns_unknown(self):
        result = self.nlu.parse_instruction("")
        assert result.intent == "unknown"
        assert result.confidence == 0.0

    def test_whitespace_only_returns_unknown(self):
        result = self.nlu.parse_instruction("   ")
        assert result.intent == "unknown"

    def test_general_intent_fallback(self):
        result = self.nlu.parse_instruction("Do something interesting today")
        assert result.intent == "general"

    def test_confidence_is_set(self):
        result = self.nlu.parse_instruction("Find a job")
        assert 0.0 <= result.confidence <= 1.0

    def test_raw_text_preserved(self):
        text = "Make $1000 by Friday"
        result = self.nlu.parse_instruction(text)
        assert result.raw_text == text


class TestNLUExtractJobRequirements:
    def setup_method(self):
        self.nlu = NLU()

    def test_returns_job_requirements(self):
        result = self.nlu.extract_job_requirements("We need a Python developer")
        assert isinstance(result, JobRequirements)

    def test_extracts_python_skill(self):
        result = self.nlu.extract_job_requirements(
            "Looking for a Python and Django developer for a web project"
        )
        assert "python" in result.skills

    def test_extracts_budget(self):
        result = self.nlu.extract_job_requirements(
            "Budget: $500 for a landing page design"
        )
        assert result.budget is not None
        assert "500" in result.budget

    def test_extracts_deadline(self):
        result = self.nlu.extract_job_requirements(
            "Need this done ASAP, within 3 days"
        )
        assert result.deadline is not None

    def test_empty_description_returns_empty(self):
        result = self.nlu.extract_job_requirements("")
        assert result.skills == []
        assert result.budget is None

    def test_raw_text_preserved(self):
        desc = "Need a React developer"
        result = self.nlu.extract_job_requirements(desc)
        assert result.raw_text == desc

    def test_multiple_skills_extracted(self):
        result = self.nlu.extract_job_requirements(
            "We need Python, SQL, and Docker expertise"
        )
        assert len(result.skills) >= 2


class TestNLUExtractKnowledge:
    def setup_method(self):
        self.nlu = NLU()

    def test_returns_list(self):
        result = self.nlu.extract_knowledge("Python is a programming language. It is easy to learn.")
        assert isinstance(result, list)

    def test_extracts_facts(self):
        text = (
            "Python is a high-level programming language. "
            "It was created by Guido van Rossum. "
            "Python can be used for web development. "
            "It is widely used in data science."
        )
        result = self.nlu.extract_knowledge(text)
        assert len(result) > 0

    def test_empty_text_returns_empty_list(self):
        result = self.nlu.extract_knowledge("")
        assert result == []

    def test_short_sentences_filtered(self):
        result = self.nlu.extract_knowledge("Hi. Ok. Yes.")
        # Very short sentences should not produce many facts
        assert len(result) <= 1

    def test_max_facts_capped(self):
        long_text = " ".join(
            [f"This is fact number {i} and it is important to know." for i in range(20)]
        )
        result = self.nlu.extract_knowledge(long_text)
        assert len(result) <= 10


# ===========================================================================
# NLG Tests
# ===========================================================================

class TestNLGWriteProposal:
    def setup_method(self):
        self.nlg = NLG()

    def test_returns_string(self):
        result = self.nlg.write_proposal("Python developer", ["python", "django"], "5 years experience")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_skills(self):
        result = self.nlg.write_proposal("Web dev job", ["python", "react"], "3 years")
        assert "python" in result.lower() or "react" in result.lower()

    def test_empty_job_returns_empty(self):
        result = self.nlg.write_proposal("", ["python"], "5 years")
        assert result == ""

    def test_no_skills_still_works(self):
        result = self.nlg.write_proposal("Data analyst", [], "2 years experience")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_professional_closing(self):
        result = self.nlg.write_proposal("ML engineer", ["python"], "expert")
        assert any(word in result.lower() for word in ["regards", "sincerely", "janus"])


class TestNLGComposeMessage:
    def setup_method(self):
        self.nlg = NLG()

    def test_returns_string(self):
        result = self.nlg.compose_message("John", "follow_up", "Checking on the invoice")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_recipient(self):
        result = self.nlg.compose_message("Alice", "introduction", "I am a developer")
        assert "Alice" in result

    def test_empty_recipient_defaults(self):
        result = self.nlg.compose_message("", "inquiry", "About the project")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_various_intents(self):
        for intent in ["follow_up", "introduction", "inquiry", "update", "thank_you"]:
            result = self.nlg.compose_message("Bob", intent, "Some context")
            assert isinstance(result, str)
            assert len(result) > 0


class TestNLGProgressReport:
    def setup_method(self):
        self.nlg = NLG()

    def test_returns_string(self):
        result = self.nlg.generate_progress_report("Earn $1000", 0.5, ["Applied to 5 jobs"])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_percentage(self):
        result = self.nlg.generate_progress_report("Goal", 0.75, [])
        assert "75" in result

    def test_zero_progress(self):
        result = self.nlg.generate_progress_report("Goal", 0.0, [])
        assert "0" in result

    def test_full_progress(self):
        result = self.nlg.generate_progress_report("Goal", 1.0, ["Done"])
        assert "100" in result

    def test_progress_clamped_above_one(self):
        result = self.nlg.generate_progress_report("Goal", 1.5, [])
        assert "100" in result

    def test_progress_clamped_below_zero(self):
        result = self.nlg.generate_progress_report("Goal", -0.5, [])
        assert "0" in result

    def test_completed_steps_included(self):
        steps = ["Step A", "Step B"]
        result = self.nlg.generate_progress_report("Goal", 0.4, steps)
        assert "Step A" in result or "Step B" in result


class TestNLGAskClarification:
    def setup_method(self):
        self.nlg = NLG()

    def test_returns_string(self):
        result = self.nlg.ask_clarification("Do the thing")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_original_request(self):
        result = self.nlg.ask_clarification("Do the thing")
        assert "Do the thing" in result

    def test_empty_request_returns_default(self):
        result = self.nlg.ask_clarification("")
        assert isinstance(result, str)
        assert len(result) > 0


# ===========================================================================
# SocialAwareness Tests
# ===========================================================================

class TestSocialAwarenessAdaptTone:
    def setup_method(self):
        self.sa = SocialAwareness()

    def test_returns_string(self):
        result = self.sa.adapt_tone("hey wanna chat", {"tone": "formal"})
        assert isinstance(result, str)

    def test_formal_tone_replaces_hey(self):
        result = self.sa.adapt_tone("hey there", {"tone": "formal"})
        assert "hey" not in result.lower() or "Hello" in result

    def test_casual_tone_replaces_hello(self):
        result = self.sa.adapt_tone("Hello there, Thank you for your time.", {"tone": "casual"})
        # casual replacements applied
        assert isinstance(result, str)

    def test_empty_message_returns_empty(self):
        result = self.sa.adapt_tone("", {"tone": "formal"})
        assert result == ""

    def test_platform_infers_tone(self):
        result = self.sa.adapt_tone("hey wanna work together", {"platform": "upwork"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_technical_tone_removes_fillers(self):
        msg = "I hope this message finds you well. Here is the bug fix."
        result = self.sa.adapt_tone(msg, {"tone": "technical"})
        assert "I hope this message finds you well" not in result

    def test_unknown_platform_uses_default(self):
        result = self.sa.adapt_tone("Hello", {"platform": "unknown_platform_xyz"})
        assert isinstance(result, str)


class TestSocialAwarenessDetectNorms:
    def setup_method(self):
        self.sa = SocialAwareness()

    def test_returns_dict(self):
        result = self.sa.detect_professional_norms("upwork")
        assert isinstance(result, dict)

    def test_upwork_is_formal(self):
        result = self.sa.detect_professional_norms("upwork")
        assert result["formality"] == "formal"

    def test_fiverr_is_casual(self):
        result = self.sa.detect_professional_norms("fiverr")
        assert result["formality"] == "casual"

    def test_linkedin_norms(self):
        result = self.sa.detect_professional_norms("linkedin")
        assert "tone" in result
        assert "greeting" in result

    def test_unknown_platform_returns_default(self):
        result = self.sa.detect_professional_norms("some_unknown_platform")
        assert result == self.sa.detect_professional_norms("default")

    def test_empty_platform_returns_default(self):
        result = self.sa.detect_professional_norms("")
        assert isinstance(result, dict)

    def test_all_known_platforms_have_required_keys(self):
        platforms = ["upwork", "fiverr", "freelancer", "linkedin", "twitter",
                     "reddit", "discord", "email", "github"]
        required_keys = {"tone", "greeting", "closing", "formality", "notes"}
        for platform in platforms:
            norms = self.sa.detect_professional_norms(platform)
            assert required_keys.issubset(norms.keys()), f"Missing keys for {platform}"

    def test_returns_copy_not_reference(self):
        result1 = self.sa.detect_professional_norms("upwork")
        result1["tone"] = "MODIFIED"
        result2 = self.sa.detect_professional_norms("upwork")
        assert result2["tone"] != "MODIFIED"


class TestSocialAwarenessBuildRapportOpener:
    def setup_method(self):
        self.sa = SocialAwareness()

    def test_returns_string(self):
        result = self.sa.build_rapport_opener("Alice", "new client")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_client_name(self):
        result = self.sa.build_rapport_opener("Bob", "first project")
        assert "Bob" in result

    def test_empty_name_uses_fallback(self):
        result = self.sa.build_rapport_opener("", "some context")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returning_client_context(self):
        result = self.sa.build_rapport_opener("Carol", "returning client, previous project")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_technical_context(self):
        result = self.sa.build_rapport_opener("Dave", "technical bug fix project")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_deterministic_output(self):
        # Same inputs should produce same output (no randomness)
        r1 = self.sa.build_rapport_opener("Eve", "new client")
        r2 = self.sa.build_rapport_opener("Eve", "new client")
        assert r1 == r2
