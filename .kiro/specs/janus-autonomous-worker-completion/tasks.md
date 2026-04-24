# Implementation Plan: Janus Autonomous Worker Completion

## Overview

Implement `janus_worker_core.py` (new file) containing all new components, plus targeted extensions to `janus_autonomous_worker.py`. All platform interaction goes through the already-built `janus_computer_use.py` / `janus_platform_browser.py` (computer-use-first). `janus_wallet.py` is the single source of truth for all financial state — no parallel tracking. All 16 correctness properties from the design are covered by Hypothesis property-based tests.

Component order: WorkerDatabase → DecisionEngine → WorkGenerator → QualityAssurance → LearningEngine → MarketAnalyzer → InvestmentEngine → MonitoringSystem → WorkCycle → Integration → Tests

---

## Tasks

- [x] 1. Implement WorkerDatabase in `janus_worker_core.py`
  - Create `janus_worker_core.py` with the `WorkerDatabase` class using the exact schema from the design (`jobs`, `skills`, `learning_resources`, `cycle_summaries` tables)
  - Implement `insert_job()`, `update_job_status()`, `get_job()`, `list_jobs()` using SQLite transactions with rollback on error
  - Implement `upsert_skill()`, `get_skill()`, `list_skills()` for skill persistence
  - Implement `insert_learning_resource()` and `insert_cycle_summary()`
  - All writes wrapped in `with conn:` transactions; retry once on `sqlite3.Error` before logging and continuing
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7_

  - [x] 1.1 Write property test for job persistence round-trip (Property 13)
    - **Property 13: Job persistence round-trip preserves all fields**
    - Use `@given(st.text(), st.text(), st.text(), st.floats(min_value=0))` to generate random `JobRecord` values; write then read back and assert `id`, `title`, `description`, `platform`, `budget`, `status` are identical
    - **Validates: Requirements 9.3**

  - [x] 1.2 Write property test for skill persistence round-trip (Property 14)
    - **Property 14: Skill persistence round-trip preserves all fields**
    - Use `@given(st.text(), st.sampled_from(SkillLevel), st.integers(min_value=0), st.floats(0.0, 1.0))` to generate random `Skill` values; write then read back and assert `name`, `level`, `experience_pts`, `success_rate` are identical
    - **Validates: Requirements 9.4**

- [x] 2. Implement DecisionEngine in `janus_worker_core.py`
  - Implement `DecisionEngine` with `WEIGHTS = {"skill_match": 0.40, "budget": 0.30, "deadline": 0.20, "learning": 0.10}`
  - Implement `_skill_match_score()`, `_budget_score()`, `_deadline_score()`, `_learning_score()` — each returns a float clamped to [0.0, 1.0]
  - Implement `score_job()` as the weighted sum of the four sub-scores
  - Implement `select_jobs()`: sort by score descending, filter out scores < 0.5, return top-N
  - Pure functions — no I/O, no database access
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x] 2.1 Write property test for job score bounds (Property 7)
    - **Property 7: Job score is always in [0.0, 1.0]**
    - Use `@given(browser_job_strategy(), skill_dict_strategy())` to generate arbitrary inputs; assert `0.0 <= score_job(job, skills) <= 1.0`
    - **Validates: Requirements 6.1**

  - [x] 2.2 Write property test for weighted score formula (Property 8)
    - **Property 8: Weighted job score equals the formula**
    - Use `@given(st.floats(0,1), st.floats(0,1), st.floats(0,1), st.floats(0,1))` for the four sub-scores; mock the four `_*_score` methods to return those values; assert total equals `0.40*s_match + 0.30*s_budget + 0.20*s_deadline + 0.10*s_learning` within 1e-9
    - **Validates: Requirements 6.2**

  - [x] 2.3 Write property test for top-N job selection (Property 9)
    - **Property 9: Selected jobs are always the top-N by score**
    - Use `@given(st.lists(browser_job_strategy(), min_size=0, max_size=20), st.integers(1, 10))` to generate job lists and capacity N; assert returned jobs are exactly the top-N by score and no selected job has score < 0.5
    - **Validates: Requirements 6.3, 6.4**

- [x] 3. Implement WorkGenerator in `janus_worker_core.py`
  - Implement `WorkGenerator.__init__(brain)` accepting `AvusBrain` or `JanusGPT`
  - Implement `_build_prompt(job: BrowserJob) -> str` — must embed `job.title`, `job.description`, and each element of `job.required_skills` as substrings
  - Implement `_format_output(raw: str, job_type: str) -> str` for code/document/design job types
  - Implement `_validate_quality(work: str, job: BrowserJob) -> float` returning a float in [0.0, 1.0]
  - Implement `generate(job) -> WorkResult` with up to `MAX_RETRIES=3` attempts; return `WorkResult` with `quality_score=0.0` if all retries fail
  - `WorkResult` dataclass must have non-None `quality_score`, `generation_time_seconds`, `attempts`
  - _Requirements: 1.1, 1.2, 1.3, 1.5, 1.6, 18.1, 18.5_

  - [x] 3.1 Write property test for prompt completeness (Property 1)
    - **Property 1: Work prompt contains all job context fields**
    - Use `@given(non_empty_text(), non_empty_text(), st.lists(non_empty_text(), min_size=1))` for title, description, skills; assert each appears as a substring in `_build_prompt(job)`
    - **Validates: Requirements 1.2**

  - [x] 3.2 Write property test for WorkResult required fields (Property 4)
    - **Property 4: Work metrics always contain required fields**
    - Use `@given(browser_job_strategy())` with a mocked brain; assert returned `WorkResult` has non-None `quality_score`, `generation_time_seconds`, `attempts`, with `quality_score` in [0.0, 1.0] and `generation_time_seconds >= 0`
    - **Validates: Requirements 1.5, 8.3**

- [x] 4. Implement QualityAssurance in `janus_worker_core.py`
  - Implement `QualityAssurance` with `MIN_QUALITY_THRESHOLD = 0.7` and `MAX_RETRIES = 3`
  - Implement `_check_completeness()`, `_check_relevance()`, `_check_format()` — each returns a float clamped to [0.0, 1.0]
  - Implement `validate(work: WorkResult, job: BrowserJob) -> QAResult` where `score` is the average of the three sub-scores, all clamped to [0.0, 1.0]
  - `QAResult.passed` is `True` iff `score >= MIN_QUALITY_THRESHOLD`
  - _Requirements: 14.1, 14.2, 14.3, 14.4_

  - [x] 4.1 Write property test for QA score bounds (Property 2)
    - **Property 2: Quality validator score is always in [0.0, 1.0]**
    - Use `@given(st.text(), browser_job_strategy())` to generate arbitrary work strings and jobs; assert `QAResult.score`, `completeness`, `relevance`, `format_score` are all in [0.0, 1.0]
    - **Validates: Requirements 1.3, 14.1**

- [x] 5. Implement LearningEngine in `janus_worker_core.py`
  - Implement `LearningEngine.__init__(engine: ComputerUseEngine, brain, db: WorkerDatabase)`
  - Implement `_search_youtube(query: str) -> List[Dict]` using `BrowserComputerUse` to open YouTube and read listings via OCR; fall back to `_search_web()` if YouTube navigation fails
  - Implement `_search_web(query: str) -> List[Dict]` using `BrowserComputerUse` to open Google/DuckDuckGo
  - Implement `_extract_concepts(content: str, skill: str) -> List[str]` — calls `brain.ask()` with a structured prompt; returns non-empty list for non-empty input
  - Implement `_map_concepts_to_skills(concepts: List[str]) -> Dict[str, float]` — maps each concept to a known skill name from the DB skills registry
  - Implement `learn_skill(skill_name: str) -> LearningResult`; persist resource to DB via `WorkerDatabase`
  - Exponential backoff (`[1, 2, 4, 8, 16]` seconds) on browser navigation retries
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 3.1, 3.2, 3.3, 13.1, 13.2, 13.3_

  - [x] 5.1 Write property test for concept extraction (Property 5)
    - **Property 5: Concept extraction returns valid skill mappings**
    - Use `@given(st.text(min_size=1), st.text(min_size=1))` for content and skill; assert `_extract_concepts()` returns non-empty list, and every key returned by `_map_concepts_to_skills()` exists in the known skills registry
    - **Validates: Requirements 2.4, 2.5**

- [x] 6. Implement MarketAnalyzer in `janus_worker_core.py`
  - Implement `MarketAnalyzer.__init__(db: WorkerDatabase, brain)`
  - Implement `trending_skills(history: List[JobRecord]) -> List[str]` — counts skill frequency across job records
  - Implement `high_paying_types(history: List[JobRecord]) -> List[str]` — groups by job type, sorts by average budget
  - Implement `skill_roi(history: List[JobRecord]) -> Dict[str, float]` — maps skill name to average payment for jobs requiring it
  - Implement `analyze(job_history: List[JobRecord]) -> MarketAnalysis` — works on empty list (returns empty lists/dicts, not None)
  - All methods are pure functions over `List[JobRecord]`; no I/O
  - _Requirements: 19.1, 19.2, 19.3, 19.7, 19.8_

  - [x] 6.1 Write property test for market analysis required keys (Property 16)
    - **Property 16: Market analysis always returns required keys**
    - Use `@given(st.lists(job_record_strategy()))` including the empty list; assert `MarketAnalysis` has non-None `trending_skills`, `high_paying_job_types`, `emerging_opportunities`, `skill_roi`, `recommendations`
    - **Validates: Requirements 19.1**

- [x] 7. Implement InvestmentEngine in `janus_worker_core.py`
  - Implement `InvestmentEngine.__init__(wallet: JanusWallet, brain)` — use the existing `JanusWallet` instance for all balance queries and expense recording
  - Implement `_should_invest(balance: Decimal) -> bool` — returns `True` when `balance > COMPUTE_THRESHOLD` (100.0 USD)
  - Implement `_prioritize_investments(balance: Decimal, weak_skills: List[str]) -> List[InvestmentAction]` — compute threshold $100, course threshold $50
  - Implement `evaluate_and_invest() -> List[InvestmentAction]` — calls `wallet.get_balance()`, calls `wallet.record_expense()` for each executed investment
  - Never create parallel financial state; all money tracking goes through `JanusWallet`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [x] 8. Implement MonitoringSystem in `janus_worker_core.py`
  - Implement `MonitoringSystem.__init__(log_path: str = "janus_worker.log")` with a `FileHandler` writing structured JSON log lines
  - Implement `log_event()`, `log_job_claimed()`, `log_work_generated()`, `log_job_completed()`, `log_payment()`, `log_skill_improved()`, `log_error()` — each writes a JSON object with `timestamp`, `event_type`, and relevant fields; credentials are never included in log output
  - Implement `get_metrics() -> PerformanceMetrics` — aggregates from log or in-memory counters; all fields non-None
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

  - [x] 8.1 Write property test for performance metrics fields (Property 15)
    - **Property 15: Performance metrics contain all required fields**
    - Use `@given(st.lists(job_record_strategy()), st.lists(transaction_strategy()))` for job and payment history; assert `PerformanceMetrics` has non-None `jobs_completed`, `total_earned`, `average_job_value`, `skill_levels`, `error_rate`
    - **Validates: Requirements 8.8**

- [x] 9. Implement WorkCycle in `janus_worker_core.py`
  - Implement `WorkCycle.__init__(db, wallet, decision_engine, work_generator, learning_engine, quality_assurance, investment_engine, market_analyzer, monitor, max_concurrent_jobs=5)`
  - Implement `_discover_jobs() -> List[BrowserJob]` — calls `PlatformBrowser` (Upwork first, Fiverr fallback); on both failing, returns empty list and schedules learning session
  - Implement `_execute_job(job: BrowserJob) -> JobResult` — calls `WorkGenerator.generate()`, then `QualityAssurance.validate()`; only calls `PlatformBrowser.deliver()` / `UpworkBrowser.submit_work()` when `qa_result.passed == True`; records income via `wallet.record_income()` on payment
  - Implement `run_one_cycle() -> CycleSummary` — sequences: discover → evaluate (DecisionEngine) → execute jobs (up to `max_concurrent_jobs` concurrently via `asyncio.gather`) → learn → invest → persist `CycleSummary`
  - Implement `run_forever()` — calls `run_one_cycle()` in a loop; catches all exceptions per cycle, logs via `MonitoringSystem.log_error()`, and continues
  - Job priority queue ordered by `deadline` ascending (earliest deadline first)
  - _Requirements: 17.1, 17.2, 17.3, 17.4, 17.5, 17.6, 20.1, 20.2, 20.3_

  - [x] 9.1 Write property test for concurrent job limit (Property 10)
    - **Property 10: Active job count never exceeds configured maximum**
    - Use `@given(st.integers(1, 10), st.lists(browser_job_strategy(), min_size=0, max_size=20))` for `max_concurrent_jobs` and job lists; simulate claim/complete events and assert active count never exceeds the configured maximum
    - **Validates: Requirements 17.1**

  - [x] 9.2 Write property test for deadline-ordered priority queue (Property 11)
    - **Property 11: Job priority queue is ordered by deadline ascending**
    - Use `@given(st.lists(browser_job_strategy(with_distinct_deadlines=True), min_size=2))` to generate jobs with distinct deadlines; assert the priority order places the earliest deadline first
    - **Validates: Requirements 17.4**

  - [x] 9.3 Write property test for low-quality work not submitted (Property 3)
    - **Property 3: Low-quality work is never submitted**
    - Use `@given(st.floats(0.0, 0.699))` for quality scores below threshold; mock `WorkGenerator.generate()` to return a `WorkResult` with that score; assert `PlatformBrowser.deliver()` and `UpworkBrowser.submit_work()` are never called
    - **Validates: Requirements 1.4, 14.3**

- [x] 10. Checkpoint — Ensure all tests pass
  - Run `pytest tests/ -x --tb=short` and confirm all property-based and unit tests pass; ask the user if any questions arise before proceeding to integration

- [x] 11. Wire components into `janus_autonomous_worker.py`
  - [x] 11.1 Add imports and lazy-init block for all `janus_worker_core` components at the top of `janus_autonomous_worker.py`
    - Import `WorkerDatabase`, `DecisionEngine`, `WorkGenerator`, `QualityAssurance`, `LearningEngine`, `MarketAnalyzer`, `InvestmentEngine`, `MonitoringSystem`, `WorkCycle` from `janus_worker_core`
    - Wrap imports in `try/except ImportError` following the existing pattern in the file
    - _Requirements: 9.1, 9.2_

  - [x] 11.2 Replace the existing `_run_work_cycle()` stub in `janus_autonomous_worker.py` with a delegation to `WorkCycle.run_one_cycle()`
    - Instantiate `WorkerDatabase`, `JanusWallet` (reuse existing `HAS_WALLET` instance), and all engines inside the existing `JanusAutonomousWorker.__init__()` when `HAS_WORKER_CORE` is True
    - Delegate `run_forever()` to `WorkCycle.run_forever()`
    - Keep existing fallback path when `HAS_WORKER_CORE` is False
    - _Requirements: 1.1, 6.1, 17.1, 20.1_

- [x] 12. Write integration tests in `tests/test_integration.py`
  - [x] 12.1 Full cycle smoke test
    - Mock `PlatformBrowser` to return one `BrowserJob`; mock `AvusBrain.ask()` to return valid work text; run `WorkCycle.run_one_cycle()`; assert DB has a job record with status `completed` and `JanusWallet` ledger has an income transaction
    - _Requirements: 1.1, 4.2, 9.3, 15.1_

  - [x] 12.2 Browser fallback test
    - Mock `UpworkBrowser` to raise an exception; assert `FiverrBrowser` is called next; assert cycle completes without crashing
    - _Requirements: 5.3, 20.1_

  - [x] 12.3 Quality gate integration test
    - Mock `AvusBrain.ask()` to return low-quality work (score < 0.7) for all 3 retries; assert `PlatformBrowser.deliver()` is never called and job is marked failed in DB
    - _Requirements: 1.4, 14.3_

- [x] 13. Write financial property test in `tests/test_financial.py`
  - [x] 13.1 Write property test for financial aggregation correctness (Property 12)
    - **Property 12: Financial aggregation is arithmetically correct**
    - Use `@given(st.lists(transaction_strategy()))` to generate arbitrary transaction lists; assert `total_earned = sum(income amounts)`, `total_spent = sum(expense amounts)`, `current_balance = total_earned - total_spent` using `JanusWallet` / `WalletAnalytics.build_report()`
    - **Validates: Requirements 15.3**

- [x] 14. Write error recovery property test in `tests/test_error_recovery.py`
  - [x] 14.1 Write property test for exponential backoff sequence (Property 6)
    - **Property 6: Exponential backoff delays follow the correct sequence**
    - Use `@given(st.integers(min_value=1, max_value=5))` for attempt number N; assert the computed delay equals `2**(N-1)` seconds (1, 2, 4, 8, 16)
    - **Validates: Requirements 5.1**

- [x] 15. Final checkpoint — Ensure all tests pass
  - Run `pytest tests/ --tb=short` and confirm all 16 property tests and all unit/integration tests pass; ask the user if any questions arise

## Notes

- Sub-tasks marked with `*` are optional and can be skipped for a faster MVP
- Each property test must use `@settings(max_examples=100)` and be tagged with `# Feature: janus-autonomous-worker-completion, Property N`
- All browser interactions (`PlatformBrowser`, `BrowserComputerUse`, `ComputerUseEngine`) must be mocked with `unittest.mock.AsyncMock` in tests — no display or network required
- `janus_wallet.py` is already built and working — never duplicate its financial state
- `janus_computer_use.py` and `janus_platform_browser.py` are already built — use them, don't reimplement
- All new code goes into `janus_worker_core.py` (new file) plus minimal extensions to `janus_autonomous_worker.py`
