# Janus Quick Start - Local Only, No APIs

## 5-Minute Setup

### Docker (Easiest)
```bash
cd C:\Users\legac\Downloads\Janus-main
docker-compose up -d
```

Done. Agent is running.

### View logs:
```bash
docker-compose logs -f janus-agent
```

### Stop:
```bash
docker-compose down
```

---

## How It Works (No APIs)

**1. You record a tutorial video** (e.g., "how to check bank balance")
   - Just record your screen doing the task normally
   - 1-2 minutes is enough

**2. Place video in `videos/` folder**
   ```
   videos/
   ├── how_to_check_balance.mp4
   ├── how_to_transfer_money.mp4
   └── how_to_pay_bills.mp4
   ```

**3. Agent watches & learns**
   - Extracts frames from video
   - Detects UI elements (buttons, fields)
   - Infers actions (click, type, scroll)
   - Builds reusable skill

**4. Agent executes skill**
   - Replays the learned actions
   - Adapts if UI slightly changes
   - Logs everything locally

**5. Agent improves**
   - Learns from success/failure
   - Refines confidence scores
   - Adds to skill library

---

## Example: Financial Automation (No APIs)

### What You Can Do:
✅ Bank account login & balance checks
✅ Money transfers between accounts
✅ Stock price monitoring (free websites)
✅ Bill payments
✅ Expense tracking

### Why No APIs Needed?
- You're automating the **UI**, not calling APIs
- Like teaching a human to use a browser
- No rate limits, no subscriptions

### Example Workflow:

**Step 1: Record a video**
```
1. Open browser
2. Go to mybank.com
3. Click login
4. Enter username
5. Enter password
6. Click submit
7. Screenshot shows balance
```
(30 seconds to 2 minutes)

**Step 2: Save as `videos/check_balance.mp4`**

**Step 3: Run Agent**
```bash
docker-compose up -d
docker-compose logs -f
```

**Step 4: Agent learns**
```
[INFO] Learning from: check_balance.mp4
[INFO] Extracted 45 frames
[INFO] Detected actions: click, type, click, type, click, wait
[INFO] Skill created: check_balance (6 actions, 0.85 confidence)
```

**Step 5: Agent executes**
```python
from skill_executor import SkillExecutor

executor.execute_skill("check_balance", dry_run=False)
# Agent automatically:
# - Opens browser
# - Navigates to bank
# - Fills in credentials
# - Clicks submit
# - Extracts balance
# - Logs result
```

---

## Directory Structure

```
Janus-main/
├── janus_main.py              # Main agent (entry point)
├── local_vision.py            # Vision processing (no APIs)
├── video_learner.py           # Learn from videos
├── browser_automation.py      # Browser control (Selenium)
├── docker-compose.yml         # Docker setup
├── Dockerfile                 # Container build
├── requirements_enhanced.txt   # Python deps
│
├── videos/                    # Your tutorial videos
│   ├── check_balance.mp4
│   ├── transfer_money.mp4
│   └── pay_bills.mp4
│
├── data/                      # Agent state
│   ├── skill_library.json
│   ├── identity_object.json
│   └── janus_task_log.json
│
└── logs/                      # Execution logs
    └── agent.log
```

---

## Common Tasks

### Learn a New Skill

```bash
# 1. Record video of task (1-2 min)
# 2. Save to videos/ folder
# 3. Agent auto-learns

# Or manually:
python -c "
from video_learner import VideoLearner
from local_vision import LocalVisionAnalyzer

learner = VideoLearner()
analyzer = LocalVisionAnalyzer()
learner.create_skill_from_video('videos/my_task.mp4', 'my_skill', analyzer)
"
```

### Execute a Learned Skill

```bash
python -c "
from skill_executor import SkillExecutor
from video_learner import VideoLearner

learner = VideoLearner()
executor = SkillExecutor(learner)
result = executor.execute_skill('my_skill', dry_run=False)
print(result)
"
```

### Monitor Agent Progress

```bash
# Watch logs in real-time
docker-compose logs -f janus-agent

# Or check task log file
cat data/janus_task_log.json

# Or check skill library
cat data/skill_library.json
```

---

## Troubleshooting

### Agent doesn't start
```bash
# Check if Docker is running
docker ps

# Check logs
docker-compose logs janus-agent

# Check ports aren't in use
netstat -an | findstr 8000
```

### Skills don't execute
```bash
# Test with dry_run first
executor.execute_skill('my_skill', dry_run=True)

# Check if skill exists
learner.list_skills()

# Check for errors in task log
cat data/janus_task_log.json
```

### OCR not working
```bash
# Install Tesseract
# Windows: https://github.com/UB-Mannheim/tesseract/wiki
# macOS: brew install tesseract
# Linux: sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

### Browser won't open
```bash
# Check Chrome is installed
chrome --version

# Download ChromeDriver
# https://chromedriver.chromium.org/
# Match your Chrome version

# Extract to drivers/ folder
```

---

## Zero to Hero: 10-Minute Example

### Minute 1-2: Record Video
```
1. Open OBS or ScreenFlow
2. Record yourself logging into your bank
3. Save as videos/check_balance.mp4
```

### Minute 3: Start Agent
```bash
cd Janus-main
docker-compose up -d
```

### Minute 4: Check Logs
```bash
docker-compose logs -f janus-agent
# See: "Skill created: check_balance (5 actions, 0.88 confidence)"
```

### Minute 5: Execute Skill
```python
from skill_executor import SkillExecutor
from video_learner import VideoLearner

learner = VideoLearner()
executor = SkillExecutor(learner)

# Dry run first
executor.execute_skill('check_balance', dry_run=True)

# Real execution
result = executor.execute_skill('check_balance', dry_run=False)
print(f"Balance: {result.get('balance')}")
```

### Minutes 6-10: Iterate
- Record more tasks
- Build skill library
- Improve accuracy
- Add error handling

---

## What You Get

✅ **Fully autonomous** - no manual intervention needed
✅ **Completely local** - nothing sent to cloud
✅ **No API keys** - no costs, no limits
✅ **Video-powered** - learn from any tutorial
✅ **Self-improving** - learns from mistakes
✅ **Offline-first** - works without internet

---

## Next Level: Advanced Configuration

Once comfortable, you can:
- Add email/SMS notifications
- Connect to task queue (Redis)
- Integrate with messaging apps
- Build multi-agent coordination
- Add error recovery logic
- Optimize performance

---

## Questions?

Check out:
- `LOCAL_ONLY_README.md` - Full documentation
- `JANUS_SETUP.md` - Architecture details
- `local_vision.py` - Vision code examples
- `video_learner.py` - Learning examples
- `browser_automation.py` - Automation examples

---

**Everything runs locally. Your data is yours. No subscriptions. No APIs. Complete control.**
