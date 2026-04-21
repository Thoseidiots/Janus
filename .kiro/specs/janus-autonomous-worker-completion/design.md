# Janus Autonomous Worker System - Design Document

## Overview

Janus is a fully autonomous AI worker system that operates independently across multiple job platforms (Upwork, Fiverr), generates real work using the Avus AI brain, learns continuously from web resources, processes real payments, and reinvests earnings in self-improvement. This design specifies the complete architecture, component interactions, data flows, and integration patterns required for a production-ready autonomous worker.

The system operates in continuous cycles: discovering jobs → evaluating opportunities → generating work → submitting deliverables → processing payments → learning and improving → reinvesting earnings.

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│             JANUS AUTONOMOUS WORKER                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          ORCHESTRATION & DECISION ENGINE                │  │
│  │  - Work Cycle Manager                                   │  │
│  │  - Job Evaluation & Scoring                             │  │
│  │  - Resource Allocation                                  │  │
│  │  - Autonomous Decision Making                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│             ▲                                                   │
│             │                                                   │
│  ┌────────────┬──────────────┼──────────────┬────────────────┐ │
│  │            │              │              │                │ │
│  ▼            ▼              ▼              ▼                ▼ │
│ ┌──────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│ │ Job  │  │ Work     │  │Learning  │  │Payment   │  │Error │ │
│ │Finder│  │Generator │  │Engine    │  │Processor │  │Recov.│ │
│ └──────┘  └──────────┘  └──────────┘  └──────────┘  └──────┘ │
│    │          │             │             │            │      │
│    └────────────────────────┼─────────────┴────────────┘      │
│             │                                                   │
│             ┌────────▼────────┐                                │
│             │  Avus AI Brain  │                                │
│             │  (Work Gen)     │                                │
│             └─────────────────┘                                │
│             │                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          EXTERNAL INTEGRATIONS                          │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │ Upwork   │  │ Fiverr   │  │ YouTube  │  │ Web      │ │  │
│  │  │ API      │  │ API      │  │ API      │  │ Search   │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
│             │                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          PERSISTENCE & STATE                            │  │
│  │  ┌──────────────────────────────────────────────────┐   │  │
│  │  │  SQLite Database                                 │   │  │
│  │  │  - Jobs & Status                                 │   │  │
│  │  │  - Skills & Experience                           │   │  │
│  │  │  - Financial Transactions                        │   │  │
│  │  │  - Learning History                              │   │  │
│  │  │  - Performance Metrics                           │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────────┘  │
│             │                                                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          MONITORING & LOGGING                           │  │
│  │  - Event Logging                                         │  │
│  │  - Performance Metrics                                   │  │
│  │  - Error Tracking                                        │  │
│  │  - Financial Audit Trail                                │  │
│  └──────────────────────────────────────────────────────────┘  │
│             │                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interactions

#### 1. Work Cycle Flow

```
START
  │
  ├─→ [Job Finder] Search for available jobs
  │     │
  │     ├─→ Query Upwork API
  │     ├─→ Query Fiverr API
  │     └─→ Store jobs in database
  │
  ├─→ [Decision Engine] Evaluate & score jobs
  │     │
  │     ├─→ Calculate skill match (40%)
  │     ├─→ Calculate budget value (30%)
  │     ├─→ Calculate deadline urgency (20%)
  │     ├─→ Calculate learning opportunity (10%)
  │     └─→ Claim top-scoring jobs (score > 0.5)
  │
  ├─→ [Work Generator] Generate work for claimed jobs
  │     │
  │     ├─→ Connect to Avus AI brain
  │     ├─→ Generate work based on job context
  │     ├─→ Validate quality (length, coherence, relevance)
  │     ├─→ Regenerate if quality < 0.7
  │     └─→ Submit to job platform
  │
  ├─→ [Payment Processor] Check for payments
  │     │
  │     ├─→ Query job platform for payment status
  │     ├─→ Record transaction in database
  │     ├─→ Update financial state
  │     └─→ Retry on failure (exponential backoff)
  │
  ├─→ [Learning Engine] Improve skills
  │     │
  │     ├─→ Identify weak skills
  │     ├─→ Search YouTube for tutorials
  │     ├─→ Search web for resources
  │     ├─→ Extract concepts from resources
  │     └─→ Update skill proficiency
  │
  ├─→ [Investment Engine] Reinvest earnings
  │     │
  │     ├─→ Analyze balance
  │     ├─→ Calculate ROI for investments
  │     ├─→ Purchase compute resources (if balance > $100)
  │     ├─→ Purchase training data (if needed)
  │     └─→ Purchase courses (if balance > $50)
  │
  └─→ SLEEP (1 hour) → REPEAT
```

#### 2. Work Generation Flow

```
Job Claimed
  │
  ├─→ Extract job context
  │     ├─ Title, description, required_skills
  │     ├─ Budget, deadline, platform
  │     └─ Client requirements
  │
  ├─→ Build AI prompt
  │     ├─ Job type detection
  │     ├─ Skill requirements
  │     ├─ Quality expectations
  │     └─ Format requirements
  │
  ├─→ Call Avus AI brain
  │     ├─ Send prompt + context
  │     ├─ Receive generated work
  │     └─ Track generation time
  │
  ├─→ Validate quality
  │     ├─ Check length (minimum words)
  │     ├─ Check coherence (grammar, structure)
  │     ├─ Check relevance (matches requirements)
  │     └─ Calculate quality score
  │
  ├─→ Quality check
  │     ├─ If score >= 0.7: Submit work
  │     ├─ If score < 0.7: Regenerate with adjusted params
  │     └─ If regeneration fails: Mark job as failed
  │
  └─→ Submit to platform
```

#### 3. Learning Flow

```
Skill Improvement Needed
  │
  ├─→ Identify weak skills
  │     └─ Skills with level < EXPERT
  │
  ├─→ Search for resources
  │     ├─ YouTube search (skill name)
  │     ├─ Web search (skill + tutorial)
  │     └─ Rank by relevance & quality
  │
  ├─→ Select top resource
  │     ├─ Fetch content
  │     ├─ Extract metadata
  │     └─ Get transcript (if available)
  │
  ├─→ Extract concepts
  │     ├─ Use AI to analyze content
  │     ├─ Identify key learning points
  │     └─ Map to skill knowledge
  │
  ├─→ Update skill
  │     ├─ Gain experience points
  │     ├─ Check for level up
  │     ├─ Update success rate
  │     └─ Store in database
  │
  └─→ Track learning
      └─ Log resource, concepts, timestamp
```

#### 4. Payment Processing Flow

```
Job Completed
  │
  ├─→ Query platform for payment status
  │     ├─ Upwork: GET /jobs/{id}/payment
  │     └─ Fiverr: GET /gigs/{id}/payment
  │
  ├─→ Payment received?
  │     ├─ YES: Record transaction
  │     └─ NO: Retry later (exponential backoff)
  │
  ├─→ Record transaction
  │     ├─ Amount, timestamp, job_id
  │     ├─ Platform, status
  │     └─ Store in database
  │
  ├─→ Update finances
  │     ├─ Add to total_earned
  │     ├─ Add to current_balance
  │     ├─ Update average_job_value
  │     └─ Increment jobs_completed
  │
  └─→ Generate audit record
      └─ For tax/compliance purposes
```

## Core Components

### 1. Job Finder

**Responsibility**: Discover available jobs from multiple platforms

**Key Methods**:
- `find_jobs(skills: List[str]) -> List[Job]`
- `query_upwork(skills) -> List[Job]`
- `query_fiverr(skills) -> List[Job]`
- `store_job(job: Job) -> None`

**Data Flow**:
- Input: List of skills Janus has
- Process: Query each platform's API
- Output: List of available jobs stored in database

**Error Handling**:
- API timeout: Retry with exponential backoff
- API error: Log and try alternative platform
- Rate limit: Queue requests and retry after cooldown

### 2. Decision Engine

**Responsibility**: Evaluate jobs and make autonomous decisions

**Key Methods**:
- `score_job(job: Job) -> float`
- `evaluate_and_claim_jobs() -> None`
- `calculate_skill_match(job) -> float`
- `calculate_budget_value(job) -> float`
- `calculate_deadline_urgency(job) -> float`
- `calculate_learning_opportunity(job) -> float`

**Scoring Formula**:
```
score = (skill_match * 0.4) + (budget * 0.3) + (deadline * 0.2) + (learning * 0.1)
```

**Decision Rules**:
- Claim job if score > 0.5
- Claim up to 5 concurrent jobs
- Prioritize high-budget jobs
- Prioritize jobs with learning opportunities
- Reject jobs with insufficient skill match

### 3. Work Generator

**Responsibility**: Generate high-quality work using Avus AI brain

**Key Methods**:
- `generate_work(job: Job) -> str`
- `validate_quality(work: str, job: Job) -> float`
- `build_prompt(job: Job) -> str`
- `call_avus_brain(prompt: str) -> str`
- `regenerate_if_needed(work: str, job: Job) -> str`

**Quality Validation**:
- Minimum length: 500 words (configurable by job type)
- Coherence: Grammar, structure, readability
- Relevance: Matches job requirements
- Format: Correct file type/structure

**Quality Score Calculation**:
```
quality_score = (length_score * 0.3) + (coherence_score * 0.4) + (relevance_score * 0.3)
```

**Regeneration Strategy**:
- If quality < 0.7: Adjust prompt parameters
- Increase detail level, add examples
- Try alternative generation approach
- Max 3 regeneration attempts per job

### 4. Learning Engine

**Responsibility**: Improve skills through continuous learning

**Key Methods**:
- `find_learning_resources(topic: str, skill: str) -> List[LearningResource]`
- `search_youtube(topic: str) -> List[LearningResource]`
- `search_web(topic: str) -> List[LearningResource]`
- `learn_from_resource(resource: LearningResource) -> List[str]`
- `extract_concepts(content: str) -> List[str]`
- `update_skill(skill: Skill, concepts: List[str]) -> None`

**Learning Process**:
1. Identify weak skills (level < EXPERT)
2. Search for educational resources
3. Rank resources by relevance and quality
4. Fetch and parse content
5. Extract key concepts using AI
6. Update skill knowledge and experience
7. Track learning in database

**Resource Prioritization**:
- YouTube tutorials (high quality, structured)
- Online courses (comprehensive, verified)
- Blog posts and articles (quick reference)
- Documentation (authoritative)

### 5. Payment Processor

**Responsibility**: Handle real money transactions

**Key Methods**:
- `check_payments() -> None`
- `process_payment(job: Job) -> Optional[float]`
- `record_transaction(amount: float, job_id: str) -> None`
- `update_finances(amount: float) -> None`
- `retry_failed_payment(job: Job) -> bool`

**Payment Flow**:
1. Query platform for payment status
2. Validate amount matches job budget
3. Record transaction in database
4. Update financial state
5. Generate audit record
6. Retry on failure (max 5 attempts)

**Retry Strategy**:
- Exponential backoff: 1s, 2s, 4s, 8s, 16s
- Max 5 retries per payment
- Log all retry attempts
- Alert on persistent failures

### 6. Error Recovery System

**Responsibility**: Handle failures gracefully and maintain reliability

**Key Methods**:
- `retry_with_backoff(func, max_retries=5) -> Any`
- `handle_api_error(error: Exception) -> None`
- `fallback_to_alternative_platform() -> None`
- `queue_operation(operation: Operation) -> None`
- `resume_queued_operations() -> None`

**Error Handling Strategies**:
- API timeout: Retry with increased timeout
- API error: Log and try alternative platform
- Network error: Queue operation and retry when online
- Database error: Rollback transaction and retry
- Critical error: Alert and pause operations

**Retry Logic**:
```
backoff_time = base_delay * (2 ^ attempt_number)
max_backoff = 300 seconds (5 minutes)
max_retries = 5
```

### 7. Investment Engine

**Responsibility**: Intelligently invest earnings in self-improvement

**Key Methods**:
- `analyze_investment_opportunities() -> List[Investment]`
- `calculate_roi(investment: Investment) -> float`
- `purchase_compute_resources() -> bool`
- `purchase_training_data() -> bool`
- `purchase_courses() -> bool`
- `track_investment_roi(investment: Investment) -> None`

**Investment Thresholds**:
- GPU compute: Balance > $100
- Training data: Balance > $75
- Online courses: Balance > $50

**ROI Calculation**:
```
roi = (expected_earnings_increase / investment_cost) * 100
```

**Investment Priorities**:
1. Skills appearing in high-paying jobs
2. Skills with high market demand
3. Skills that unlock new job categories
4. Compute resources for faster work generation

## Data Models

### Job

```python
@dataclass
class Job:
    id: str
    title: str
    description: str
    required_skills: List[str]
    budget: float
    deadline: datetime
    platform: str  # upwork, fiverr
    status: JobStatus
    claimed_by: Optional[str]
    completion_time: Optional[float]
    quality_score: Optional[float]
    payment_received: bool
```

### Skill

```python
@dataclass
class Skill:
    name: str
    level: SkillLevel  # BEGINNER, INTERMEDIATE, ADVANCED, EXPERT
    experience_points: int
    last_used: Optional[datetime]
    success_rate: float
```

### FinancialState

```python
@dataclass
class FinancialState:
    total_earned: float
    total_spent: float
    current_balance: float
    jobs_completed: int
    average_job_value: float
```

### LearningResource

```python
@dataclass
class LearningResource:
    url: str
    title: str
    type: str  # youtube, article, tutorial
    topic: str
    duration_minutes: int
    completed: bool
    learned_concepts: List[str]
```

## Database Schema

### jobs table
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    budget REAL,
    status TEXT,
    platform TEXT,
    completed_at TIMESTAMP,
    payment_received REAL
);
```

### skills table
```sql
CREATE TABLE skills (
    name TEXT PRIMARY KEY,
    level INTEGER,
    experience_points INTEGER,
    success_rate REAL
);
```

### learning table
```sql
CREATE TABLE learning (
    url TEXT PRIMARY KEY,
    title TEXT,
    topic TEXT,
    completed BOOLEAN,
    learned_at TIMESTAMP
);
```

### financials table
```sql
CREATE TABLE financials (
    date TIMESTAMP,
    type TEXT,
    amount REAL,
    description TEXT
);
```

## Integration Points

### Upwork API

**Authentication**: OAuth2 or API key
**Endpoints**:
- `GET /profiles/v1/search/jobs` - Search for jobs
- `POST /profiles/v1/jobs/{id}/apply` - Apply for job
- `POST /profiles/v1/jobs/{id}/submit` - Submit work
- `GET /profiles/v1/jobs/{id}/payment` - Get payment status

### Fiverr API

**Authentication**: API key
**Endpoints**:
- `GET /v1/gigs/search` - Search for gigs
- `POST /v1/gigs/{id}/offer` - Submit offer
- `POST /v1/gigs/{id}/submit` - Submit work
- `GET /v1/gigs/{id}/payment` - Get payment status

### YouTube API

**Authentication**: API key
**Endpoints**:
- `GET /youtube/v3/search` - Search videos
- `GET /youtube/v3/videos` - Get video metadata
- `GET /youtube/v3/captions` - Get video transcript

### Web Search API

**Options**: Google Custom Search, Bing Search, DuckDuckGo
**Endpoints**:
- Search for learning resources
- Rank by relevance and authority
- Fetch and parse content

### Avus AI Brain

**Connection**: Local or remote inference
**Input**: Job context + prompt
**Output**: Generated work
**Fallback**: Template-based generation if AI unavailable

## Security Considerations

### Credential Management

- Store API keys in encrypted format (AES-256)
- Load credentials only when needed
- Never log or display credentials in plain text
- Use environment variables for sensitive data
- Rotate credentials on expiration

### API Security

- Use HTTPS for all API calls
- Validate SSL certificates
- Implement rate limiting
- Handle authentication errors gracefully
- Log security events

### Data Protection

- Encrypt sensitive data in database
- Implement database backups
- Audit financial transactions
- Maintain transaction logs for compliance
- Implement access controls

## Monitoring & Observability

### Key Metrics

- Jobs discovered per cycle
- Jobs claimed per cycle
- Jobs completed per cycle
- Average job value
- Total earnings
- Skill levels and experience
- Work quality scores
- Payment success rate
- Error rate and types
- API response times

### Logging

- Event logging: All significant events
- Error logging: All errors with stack traces
- Performance logging: Generation time, API latency
- Financial logging: All transactions
- Learning logging: Resources used, concepts learned

### Alerts

- Critical errors: Pause operations and alert
- Payment failures: Alert after 3 retries
- API unavailability: Switch to alternative platform
- Low balance: Prioritize high-paying jobs
- Skill degradation: Recommend learning

## Deployment Architecture

### Local Development

- SQLite database
- Mock API responses
- Logging to console and file
- Single-threaded execution

### Production Deployment

- PostgreSQL database (for scalability)
- Real API integrations
- Structured logging (JSON format)
- Async/concurrent job processing
- Docker containerization
- Kubernetes orchestration (optional)

### Scaling Considerations

- Horizontal scaling: Multiple worker instances
- Load balancing: Distribute jobs across workers
- Database replication: Master-slave setup
- Caching: Redis for frequently accessed data
- Message queue: For async operations

## Correctness Properties

### Property 1: Financial Accuracy

**Specification**: All earnings must be accurately tracked and reconciled with platform records.

**Implementation**:
- Record every payment transaction
- Validate amounts against job budget
- Maintain audit trail
- Reconcile daily with platform

### Property 2: Skill Consistency

**Specification**: Skill levels must increase monotonically and reflect actual learning.

**Implementation**:
- Track experience points
- Validate level-up thresholds
- Record learning resources
- Prevent skill degradation without reason

### Property 3: Job Completion

**Specification**: Every claimed job must be completed or explicitly failed.

**Implementation**:
- Track job status transitions
- Require completion or failure reason
- Maintain job history
- Alert on stuck jobs

### Property 4: Work Quality

**Specification**: All submitted work must meet minimum quality standards.

**Implementation**:
- Validate quality before submission
- Track quality scores
- Regenerate if below threshold
- Maintain quality metrics

### Property 5: Autonomous Operation

**Specification**: System must operate without human intervention for extended periods.

**Implementation**:
- Implement error recovery
- Handle edge cases gracefully
- Maintain state across restarts
- Alert on critical issues only

