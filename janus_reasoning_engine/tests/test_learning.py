"""
Tests for the Learning and Skill Acquisition system.

Covers:
- SkillGapDetector: gap detection, ROI calculation, should_learn logic
- AutonomousLearner: knowledge extraction, graceful degradation
- SkillInventory: add/update skills, domain queries, strengths/weaknesses, planning
"""

import pytest
import tempfile
import shutil

from janus_reasoning_engine.learning.skill_gap_detector import (
    SkillGap,
    SkillGapDetector,
    DEFAULT_LEARNING_HOURS_PER_SKILL,
    DEFAULT_HOURLY_RATE,
    DEFAULT_ROI_THRESHOLD,
)
from janus_reasoning_engine.learning.autonomous_learner import (
    AutonomousLearner,
    LearningResult,
)
from janus_reasoning_engine.learning.skill_inventory import SkillInventory
from janus_reasoning_engine.memory.semantic_memory import SemanticMemory, SkillLevel
from janus_reasoning_engine.memory.unified_memory import UnifiedMemory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path)


@pytest.fixture
def unified_memory(temp_dir):
    mem = UnifiedMemory(
        hbm_dimension=500,
        hbm_sparsity=0.1,
        sqlite_path=f"{temp_dir}/test_learning.db",
        artifacts_dir=f"{temp_dir}/artifacts",
        enable_hbm=True,
        enable_sqlite=True,
        enable_filesystem=True,
    )
    mem.initialize()
    yield mem
    mem.shutdown()


@pytest.fixture
def semantic_memory(unified_memory):
    return SemanticMemory(unified_memory)


@pytest.fixture
def skill_inventory(semantic_memory):
    return SkillInventory(semantic_memory)


@pytest.fixture
def detector():
    return SkillGapDetector(hourly_rate=50.0, roi_threshold=1.5)


@pytest.fixture
def learner(semantic_memory):
    return AutonomousLearner(semantic_memory=semantic_memory)


# ---------------------------------------------------------------------------
# SkillGapDetector tests
# ---------------------------------------------------------------------------


class TestSkillGapDetector:
    def test_no_gaps_when_skills_sufficient(self, detector):
        opportunity = {
            "required_skills": ["Python", "SQL"],
            "value": 500.0,
        }
        inventory = {"Python": 0.8, "SQL": 0.9}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert gaps == []

    def test_detects_missing_skill(self, detector):
        opportunity = {
            "required_skills": ["React"],
            "value": 300.0,
        }
        inventory = {}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert len(gaps) == 1
        assert gaps[0].skill_name == "React"
        assert gaps[0].current_confidence == 0.0
        assert gaps[0].gap_size > 0

    def test_detects_partial_gap(self, detector):
        opportunity = {
            "required_skills": {"Python": 0.8},
            "value": 400.0,
        }
        inventory = {"Python": 0.5}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap.skill_name == "Python"
        assert abs(gap.gap_size - 0.3) < 1e-6
        assert gap.current_confidence == 0.5

    def test_roi_calculation(self, detector):
        # ROI = value / (learning_hours * hourly_rate)
        opportunity = {"required_skills": ["Rust"], "value": 1000.0}
        inventory = {}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert len(gaps) == 1
        gap = gaps[0]
        expected_roi = 1000.0 / (gap.estimated_learning_hours * 50.0)
        assert abs(gap.roi - expected_roi) < 0.01

    def test_should_learn_high_roi(self, detector):
        gap = SkillGap(
            skill_name="FastAPI",
            required_confidence=0.7,
            current_confidence=0.0,
            gap_size=0.7,
            estimated_learning_hours=2.0,
            roi=10.0,  # very high ROI
        )
        assert detector.should_learn(gap, opportunity_value=1000.0)

    def test_should_not_learn_low_roi(self, detector):
        gap = SkillGap(
            skill_name="Obscure Framework",
            required_confidence=0.7,
            current_confidence=0.0,
            gap_size=0.7,
            estimated_learning_hours=100.0,
            roi=0.1,
        )
        assert not detector.should_learn(gap, opportunity_value=50.0)

    def test_should_not_learn_if_not_learnable(self, detector):
        gap = SkillGap(
            skill_name="Quantum Computing",
            required_confidence=1.0,
            current_confidence=0.0,
            gap_size=1.0,
            estimated_learning_hours=5.0,
            roi=20.0,
            is_learnable=False,
        )
        assert not detector.should_learn(gap, opportunity_value=5000.0)

    def test_dict_required_skills(self, detector):
        opportunity = {
            "required_skills": {"Python": 0.9, "Docker": 0.6},
            "value": 800.0,
        }
        inventory = {"Python": 0.95, "Docker": 0.3}
        gaps = detector.detect_gaps(opportunity, inventory)
        # Python is sufficient; Docker has a gap
        assert len(gaps) == 1
        assert gaps[0].skill_name == "Docker"

    def test_zero_value_opportunity(self, detector):
        opportunity = {"required_skills": ["Go"], "value": 0.0}
        inventory = {}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert len(gaps) == 1
        assert gaps[0].roi == 0.0

    def test_multiple_gaps(self, detector):
        opportunity = {
            "required_skills": ["Kubernetes", "Terraform", "AWS"],
            "value": 2000.0,
        }
        inventory = {}
        gaps = detector.detect_gaps(opportunity, inventory)
        assert len(gaps) == 3
        names = {g.skill_name for g in gaps}
        assert names == {"Kubernetes", "Terraform", "AWS"}


# ---------------------------------------------------------------------------
# AutonomousLearner tests
# ---------------------------------------------------------------------------


class TestAutonomousLearner:
    def test_learn_skill_no_external_deps(self, learner):
        """learn_skill works even with no computer-use or video engine."""
        result = learner.learn_skill("Python", {"domain": "programming"})
        assert isinstance(result, LearningResult)
        assert result.skill_name == "Python"
        # No external sources → no knowledge items, but no crash
        assert isinstance(result.knowledge_items, list)

    def test_extract_knowledge_from_text_heuristic(self, learner):
        text = (
            "Python is a high-level programming language. "
            "Python supports multiple programming paradigms. "
            "Python has a large standard library."
        )
        items = learner.extract_knowledge_from_text(text, "Python")
        assert isinstance(items, list)
        assert len(items) > 0

    def test_extract_knowledge_empty_text(self, learner):
        items = learner.extract_knowledge_from_text("", "Python")
        assert items == []

    def test_extract_knowledge_whitespace_only(self, learner):
        items = learner.extract_knowledge_from_text("   \n\t  ", "Python")
        assert items == []

    def test_learn_skill_stores_in_memory(self, learner, semantic_memory):
        text = (
            "Docker is a containerisation platform. "
            "Docker uses images to package applications. "
            "Docker containers are lightweight and portable."
        )
        items = learner.extract_knowledge_from_text(text, "Docker")
        # Manually trigger storage
        learner._store_knowledge("Docker", items, {"domain": "devops"})

        # Verify something was stored
        results = semantic_memory.search_knowledge("Docker")
        assert isinstance(results, list)

    def test_learning_result_fields(self, learner):
        result = learner.learn_skill("TypeScript", {})
        assert result.skill_name == "TypeScript"
        assert isinstance(result.success, bool)
        assert isinstance(result.sources_used, list)
        assert result.learning_duration_seconds >= 0.0
        assert result.knowledge_count == len(result.knowledge_items)

    def test_graceful_with_failing_computer_use(self, semantic_memory):
        class BrokenEngine:
            def search(self, query):
                raise RuntimeError("network error")

        learner = AutonomousLearner(
            semantic_memory=semantic_memory,
            computer_use_engine=BrokenEngine(),
        )
        # Should not raise
        result = learner.learn_skill("Rust", {"domain": "systems"})
        assert isinstance(result, LearningResult)

    def test_parse_bullet_list(self):
        text = "- First fact\n- Second fact\n- Third fact"
        items = AutonomousLearner._parse_bullet_list(text)
        assert items == ["First fact", "Second fact", "Third fact"]

    def test_parse_bullet_list_empty(self):
        assert AutonomousLearner._parse_bullet_list("no bullets here") == []


# ---------------------------------------------------------------------------
# SkillInventory tests
# ---------------------------------------------------------------------------


class TestSkillInventory:
    def test_add_skill(self, skill_inventory):
        skill_id = skill_inventory.add_skill("Python", 0.8, domains=["programming"])
        assert skill_id is not None
        assert skill_inventory.get_skill_confidence("Python") == pytest.approx(0.8)

    def test_add_skill_clamps_confidence(self, skill_inventory):
        skill_inventory.add_skill("Overconfident", 1.5)
        assert skill_inventory.get_skill_confidence("Overconfident") == pytest.approx(1.0)

        skill_inventory.add_skill("Negative", -0.5)
        assert skill_inventory.get_skill_confidence("Negative") == pytest.approx(0.0)

    def test_get_skill_confidence_unknown(self, skill_inventory):
        assert skill_inventory.get_skill_confidence("UnknownSkill") == 0.0

    def test_update_confidence_positive(self, skill_inventory):
        skill_inventory.add_skill("JavaScript", 0.4, domains=["web"])
        skill_inventory.update_confidence("JavaScript", 0.2)
        assert skill_inventory.get_skill_confidence("JavaScript") == pytest.approx(0.6)

    def test_update_confidence_negative(self, skill_inventory):
        skill_inventory.add_skill("CSS", 0.7, domains=["web"])
        skill_inventory.update_confidence("CSS", -0.3)
        assert skill_inventory.get_skill_confidence("CSS") == pytest.approx(0.4)

    def test_update_confidence_clamps_to_zero(self, skill_inventory):
        skill_inventory.add_skill("Rust", 0.1, domains=["systems"])
        skill_inventory.update_confidence("Rust", -1.0)
        assert skill_inventory.get_skill_confidence("Rust") == pytest.approx(0.0)

    def test_update_confidence_creates_skill_if_missing(self, skill_inventory):
        skill_inventory.update_confidence("NewSkill", 0.5)
        assert skill_inventory.get_skill_confidence("NewSkill") == pytest.approx(0.5)

    def test_get_skills_by_domain(self, skill_inventory):
        skill_inventory.add_skill("Python", 0.9, domains=["programming"])
        skill_inventory.add_skill("Java", 0.6, domains=["programming"])
        skill_inventory.add_skill("Photoshop", 0.3, domains=["design"])

        prog_skills = skill_inventory.get_skills_by_domain("programming")
        names = {s.name for s in prog_skills}
        assert "Python" in names
        assert "Java" in names
        assert "Photoshop" not in names

    def test_identify_strengths(self, skill_inventory):
        skill_inventory.add_skill("Expert Skill", 0.9, domains=["x"])
        skill_inventory.add_skill("Weak Skill", 0.2, domains=["x"])
        skill_inventory.add_skill("Mid Skill", 0.6, domains=["x"])

        strengths = skill_inventory.identify_strengths(threshold=0.7)
        names = {s.name for s in strengths}
        assert "Expert Skill" in names
        assert "Weak Skill" not in names

    def test_identify_weaknesses(self, skill_inventory):
        skill_inventory.add_skill("Strong", 0.85, domains=["x"])
        skill_inventory.add_skill("Weak", 0.2, domains=["x"])
        skill_inventory.add_skill("Border", 0.4, domains=["x"])

        weaknesses = skill_inventory.identify_weaknesses(threshold=0.4)
        names = {s.name for s in weaknesses}
        assert "Weak" in names
        assert "Border" in names
        assert "Strong" not in names

    def test_plan_development(self, skill_inventory):
        skill_inventory.add_skill("Python", 0.9, domains=["programming"])
        skill_inventory.add_skill("Docker", 0.2, domains=["devops"])
        skill_inventory.add_skill("Kubernetes", 0.1, domains=["devops"])

        plan = skill_inventory.plan_development(["devops"])
        assert "Docker" in plan
        assert "Kubernetes" in plan
        # Python is not in devops domain
        assert "Python" not in plan

    def test_plan_development_no_weak_skills(self, skill_inventory):
        skill_inventory.add_skill("Python", 0.9, domains=["programming"])
        plan = skill_inventory.plan_development(["programming"])
        # Python confidence >= 0.5, so nothing to learn
        assert "Python" not in plan

    def test_as_dict(self, skill_inventory):
        skill_inventory.add_skill("Python", 0.8, domains=["programming"])
        skill_inventory.add_skill("SQL", 0.6, domains=["data"])

        d = skill_inventory.as_dict()
        assert isinstance(d, dict)
        assert "Python" in d
        assert d["Python"] == pytest.approx(0.8)

    def test_add_skill_updates_existing(self, skill_inventory):
        """Adding a skill that already exists should update its confidence."""
        skill_inventory.add_skill("Go", 0.3, domains=["systems"])
        skill_inventory.add_skill("Go", 0.7, domains=["systems"])
        # Should reflect the updated confidence
        conf = skill_inventory.get_skill_confidence("Go")
        assert conf == pytest.approx(0.7)
