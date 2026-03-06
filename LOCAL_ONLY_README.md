# Janus Autonomous AI Agent - Local-Only Edition

**No APIs. No API Keys. Completely Offline Capable.**

Janus is a persistent, autonomous AI agent that learns from videos and automates tasks through screen and browser automation. Everything runs locally—no external API calls to OpenAI, Claude, or any cloud service.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           JANUS AUTONOMOUS AGENT                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │ LOCAL VISION PROCESSING                      │  │
│  │  • Frame analysis (OpenCV)                   │  │
│  │  • UI element detection                      │  │
│  │  • Text extraction (Tesseract OCR)           │  │
│  │  • Action inference                          │  │
│  └──────────────────────────────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │ LOCAL LEARNING MODULE                        │  │
│  │  • Watch tutorial videos                     │  │
│  │  • Extract action sequences                  │  │
│  │  • Build skill library                       │  │
│  │  • Improve from experience                   │  │
│  └──────────────────────────────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │ BROWSER AUTOMATION                           │  │
│  │  • Selenium-powered                          │  │
│  │  • Financial task automation                 │  │
│  │  • Form filling                              │  │
│  │  • Data scraping                             │  │
│  └──────────────────────────────────────────────┘  │
│                         │                          │
│  ┌──────────────────────────────────────────────┐  │
│  │ PERSISTENCE LAYER                            │  │
│  │  • Redis (state/tasks)                       │  │
│  │  • SQLite (skill library)                    │  │
│  │  • JSON (identity/logs)                      │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

## Setup

### Prerequisites
- Docker & Docker Compose (recommended)
- OR Python 3.11+, Chrome/Chromium, Tesseract

### Option 1: Docker (Recommended)

```bash
cd C:\Users\legac\Downloads\Janus-main
docker-compose up -d
docker-compose logs -f janus-agent
```

### Option 2: Local Python Setup

```bash
# Install Python dependencies
pip install -r requirements_enhanced.txt

# Install system dependencies
# Linux (Ubuntu/Debian):
sudo apt-get install tesseract-ocr chromium-browser

# macOS:
brew install tesseract chromium

# Windows:
# Download and install from: https://github.com/UB-Mannheim/tesseract/wiki

# Download ChromeDriver:
# https://chromedriver.chromium.org/
# Extract to: C:\Users\legac\Downloads\Janus-main\drivers\

# Run agent
python janus_main.py
```

## Key Modules

### 1. **local_vision.py** - Computer Vision (No APIs)
- UI element detection
- Text recognition (Tesseract OCR)
- Frame change detection
- Action inference from video

```python
from local_vision import LocalVisionAnalyzer

analyzer = LocalVisionAnalyzer()
summary = analyzer.get_screenshot_summary(frame)
action = analyzer.infer_action(prev_frame, curr_frame)
```

### 2. **video_learner.py** - Learn from Videos
- Extract frames from tutorial videos
- Analyze action sequences
- Build reusable skills
- Execute learned workflows

```python
from video_learner import VideoLearner, SkillExecutor

learner = VideoLearner()
skill = learner.create_skill_from_video(
    "tutorial.mp4",
    "open_browser_skill",
    analyzer
)

executor = SkillExecutor(learner)
executor.execute_skill("open_browser_skill", dry_run=False)
```

### 3. **browser_automation.py** - Browser Control
- Selenium-powered automation
- Financial task workflows
- Form filling
- Data extraction

```python
from browser_automation import BrowserAutomationAgent

agent = BrowserAutomationAgent()
agent.start_browser(headless=False)
agent.navigate_to("https://example-bank.com")
agent.type_text("input[name='username']", "myusername")
agent.click("button[type='submit']")
```

## Usage Examples

### Example 1: Learn from Video & Execute

```bash
# Place tutorial videos in videos/ directory
# Videos should demonstrate a workflow (e.g., "how to check bank balance")

mkdir videos
# Copy your tutorial videos here

# Start agent - it will automatically:
# 1. Watch videos
# 2. Learn action sequences
# 3. Build skill library
# 4. Test execution

docker-compose up -d
docker-compose logs -f janus-agent
```

### Example 2: Automate Bank Login (No API)

```python
from browser_automation import BrowserAutomationAgent, FinancialAutomationWorkflow

# Create agent
agent = BrowserAutomationAgent()
agent.start_browser(headless=False)

# Create workflow
workflow = FinancialAutomationWorkflow(agent)

# Check balance by simulating human interaction
result = workflow.check_bank_balance(
    bank_url="https://mybank.com/login",
    username_selector="input[name='username']",
    password_selector="input[name='password']",
    username="your_username",
    password="your_password"
)

print(result)
agent.close()
```

### Example 3: Monitor Stock Prices (Free Websites)

```python
from browser_automation import BrowserAutomationAgent, FinancialAutomationWorkflow

agent = BrowserAutomationAgent()
workflow = FinancialAutomationWorkflow(agent)

# Monitor stock from free website (no API key needed)
result = workflow.monitor_stock_price(
    stock_url="https://finance.yahoo.com/quote/AAPL",
    stock_symbol="AAPL"
)

# Screenshot saved automatically
print(f"Screenshot: {result['screenshot']}")
print(f"Price: {result.get('current_price', 'Not found')}")
```

## Data Flow

```
Tutorial Videos
       ↓
  [VideoLearner] ← Extracts frames
       ↓
 [Local Vision] ← Analyzes changes
       ↓
 Skill Library ← Infers actions
       ↓
  [Executor]   ← Replays & improves
       ↓
 Browser/Screen Automation
```

## Skills Library

Skills are JSON files stored in `skill_library.json`:

```json
{
  "name": "check_bank_balance",
  "description": "Login and retrieve account balance",
  "actions": [
    {"type": "navigate", "url": "https://bank.com"},
    {"type": "click", "x": 100, "y": 50},
    {"type": "type", "text": "username"},
    {"type": "click", "x": 100, "y": 100},
    {"type": "type", "text": "password"},
    {"type": "click", "x": 150, "y": 150},
    {"type": "wait", "seconds": 3},
    {"type": "screenshot"},
    {"type": "extract_text", "region": [200, 200, 400, 250]}
  ],
  "confidence": 0.85
}
```

## Financial Automation Examples

### No APIs Needed For:
✅ Bank account login & balance check (UI scraping)
✅ Stock price monitoring (public websites)
✅ Expense tracking (form filling + screenshot)
✅ Money transfers (simulated UI interaction)
✅ Bill payment (browser automation)

### Why No APIs?
- Free websites like Yahoo Finance, Investing.com provide price data
- Bank websites have UI we can automate with Selenium
- No rate limits, no subscription costs
- Data is yours - nothing sent to third parties

## Customization

### Add New Skills

1. **Record a tutorial video** showing the task
2. **Run video learner:**
   ```python
   learner.create_skill_from_video(
       "my_tutorial.mp4",
       "my_skill_name",
       analyzer
   )
   ```
3. **Execute the learned skill:**
   ```python
   executor.execute_skill("my_skill_name", dry_run=True)  # Test first
   executor.execute_skill("my_skill_name", dry_run=False) # Execute
   ```

### Integrate With External Systems

```python
# Add Redis queue for background tasks
from arq import create_pool

# Add database for persistence
import sqlite3

# Combine with messaging
# Add email, SMS, Telegram notifications
```

## Troubleshooting

**Issue: "Cannot find Tesseract"**
```
Solution: Install Tesseract binary and set PATH
Windows: https://github.com/UB-Mannheim/tesseract/wiki
Linux: sudo apt-get install tesseract-ocr
macOS: brew install tesseract
```

**Issue: "ChromeDriver not found"**
```
Solution: Download from https://chromedriver.chromium.org/
Match your Chrome version. Extract to drivers/ folder.
```

**Issue: "Selenium can't connect to browser"**
```
Solution: 
1. Check if Chrome is installed
2. Try headless=False to see what's happening
3. Verify ChromeDriver version matches Chrome
```

**Issue: "Vision analysis inaccurate"**
```
Solution:
1. Improve video quality (better lighting, clearer UI)
2. Add more training videos for the same task
3. Adjust confidence thresholds in local_vision.py
```

## Performance Tips

- **Videos**: 720p or higher, 30fps
- **Headless**: Use `headless=True` for faster automation
- **OCR**: Tesseract is slower than APIs but completely free
- **Caching**: UI elements are cached to avoid re-detection
- **Parallelization**: Run multiple agents with Redis queue

## Security Considerations

- ✅ **Credentials stored locally** in environment files or secure storage
- ✅ **No data sent to cloud** - everything on your machine
- ✅ **Audit trail**: All actions logged locally in JSON
- ✅ **Screenshots stored locally**: No remote uploads
- ❌ **Don't store passwords in code** - use environment variables:
  ```bash
  export BANK_USERNAME=myuser
  export BANK_PASSWORD=mypass
  
  python -c "import os; print(os.getenv('BANK_USERNAME'))"
  ```

## Next Steps

1. **Record tutorial videos** for tasks you want to automate
2. **Place videos in `videos/` folder**
3. **Run agent** - it will learn automatically
4. **Monitor logs** - see what skills it learns
5. **Test execution** with `dry_run=True`
6. **Execute for real** with `dry_run=False`
7. **Iterate** - improve with more examples

## Contributing

Want to add support for:
- PDF extraction?
- Email automation?
- Spreadsheet manipulation?
- More browser scenarios?

Just add new modules and integrate with the main agent loop!

## License

Local-only, no restrictions. Use as you wish.

---

**Remember**: Everything runs locally. Your data is yours. No subscriptions, no API keys, no cloud dependencies.
