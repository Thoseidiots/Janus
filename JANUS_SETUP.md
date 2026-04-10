# Janus Autonomous Agent - Docker Setup

## ✅ Status: Running

Your Janus autonomous AI agent is now containerized and running with Docker.

### What was created:

1. **Docker Image (janus-agent:latest)**
   - Multi-stage build: Rust compiler stage + Python runtime stage
   - Based on Python 3.11-slim
   - 2.58GB total size, pre-cached
   - Includes all dependencies: ffmpeg, audio libraries, computer vision tools, Python packages

2. **Docker Compose Configuration (docker-compose.yml)**
   - **janus-agent service**: Main autonomous agent
     - Port 8000: API endpoint
     - Port 8081: Web interface
     - Port 9000: Debug port
   - **redis service**: State management and task queue
     - Port 6379: Redis server
   - Volumes: data/, logs/, tasks/ directories
   - Health checks enabled
   - Automatic restart policy

3. **Vision/Automation Module (vision_automation.py)**
   - Screen capture capabilities
   - Mouse/keyboard control
   - Task recording and replay
   - Ready for integration with vision APIs (Claude, GPT-4V)

4. **Main Entry Point (janus_main.py)**
   - Headless-compatible autonomous loop
   - Task logging and history
   - Identity management
   - Graceful shutdown handling

5. **Supporting Files**
   - `.dockerignore`: Optimize build context
   - `setup.sh`: Quick start script
   - `Dockerfile`: Production-ready multi-stage build

### Running Janus:

#### Start the agent:
```bash
cd C:\Users\legac\Downloads\Janus-main
docker-compose up -d
```

#### View logs:
```bash
docker-compose logs -f janus-agent
```

#### Stop the agent:
```bash
docker-compose down
```

#### Execute a command in the container:
```bash
docker-compose exec janus-agent python -c "print('Hello from Janus')"
```

### Current Behavior:

The agent runs in **headless mode** and:
1. Initializes with its persistent identity
2. Starts the autonomous cognitive loop
3. Logs tasks to `janus_task_log.json`
4. Cycles through observe → plan → propose → verify (simulated)
5. Runs for 10 iterations then stops (by design)

### Next Steps to Add Real Autonomy:

#### 1. **Vision + Screen Automation** (for desktop tasks)
   - Uncomment `from vision_automation import ...` in janus_main.py
   - Run on local machine (not Docker) or use X11 forwarding
   - Integrate Claude/GPT-4V for screen analysis

#### 2. **Task Queue Integration**
   - Connect Redis to a task management system
   - Add webhook listeners for external commands
   - Implement RPC layer for task dispatch

#### 3. **Financial APIs** (without direct API calls)
   - Use Selenium/Playwright for browser automation
   - Automate login workflows
   - Screenshot and analyze financial dashboards
   - Execute actions via simulated clicks/typing

#### 4. **Learning from Videos**
   - Integrate video ingestion pipeline
   - Use computer vision to extract workflows
   - Convert video frames to action sequences
   - Build a skill library

#### 5. **External Integrations**
   - Email: Connect SMTP/IMAP
   - Chat: Telegram, Discord, Slack
   - Market Data: Use free APIs (Alpha Vantage, Finnhub)
   - Cloud Storage: S3, Google Drive

### Key Architecture:

```
┌─────────────────────────────────────────┐
│       Janus Autonomous Agent            │
├─────────────────────────────────────────┤
│  • Identity (persistent)                │
│  • Cognitive Loop (O→P→P→V)             │
│  • Task Logger                          │
│  • Vision Module (optional)             │
│  • Redis State Store                    │
└─────────────────────────────────────────┘
         │                      │
         v                      v
    Docker Network         Redis 6379
   Port 8000-9000
```

### Files Modified/Created:

- ✅ `Dockerfile` - Production build
- ✅ `docker-compose.yml` - Orchestration
- ✅ `janus_main.py` - Entry point
- ✅ `vision_automation.py` - Vision module (optional)
- ✅ `.dockerignore` - Build optimization
- ✅ `setup.sh` - Quick start

### Troubleshooting:

**Container keeps restarting?**
```bash
docker-compose logs janus-agent
```

**Port already in use?**
Edit `docker-compose.yml`, change port mappings (e.g., 8081:8080)

**Memory issues?**
Adjust in docker-compose.yml:
```yaml
deploy:
  resources:
    limits:
      memory: 4G
```

---

**Next**: Once the foundation is solid, we can add:
- Real task automation
- Video-based learning
- Financial integrations
- Multi-agent coordination

Let me know what capability you want to build next!
