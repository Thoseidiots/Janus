# Quick Install Guide - Missing Development Tools

Based on the Janus analysis, here are the recommended tools to install with ready-to-use commands.

---

## 🔴 High Priority (Install These First)

### 1. Node.js + npm (JavaScript/TypeScript Development)

**Why**: Essential for modern web development, required for React, Vue, Angular, and most frontend tooling.

```powershell
# Option 1: Using winget (recommended)
winget install OpenJS.NodeJS.LTS

# Option 2: Using official installer
# Download from: https://nodejs.org/

# Verify installation
node --version
npm --version
```

**After installation, install common tools**:
```powershell
npm install -g yarn pnpm typescript ts-node
npm install -g @angular/cli create-react-app vite
```

---

### 2. PostgreSQL (Production Database)

**Why**: Industry-standard relational database, much more powerful than SQLite.

```powershell
# Using winget
winget install PostgreSQL.PostgreSQL.16

# Or download from: https://www.postgresql.org/download/windows/

# Verify installation
psql --version
```

**Post-install setup**:
```powershell
# Set password for postgres user
# Start pgAdmin or use psql to configure
```

---

### 3. CMake (Cross-Platform Build System)

**Why**: Essential for C/C++ projects, especially cross-platform ones.

```powershell
# Using winget
winget install Kitware.CMake

# Verify installation
cmake --version
```

---

## 🟡 Medium Priority (Useful Additions)

### 4. Redis (Caching & Session Storage)

**Why**: Fast in-memory data store, essential for web applications.

```powershell
# Using winget
winget install Redis.Redis

# Or using Chocolatey
choco install redis-64

# Verify installation
redis-cli --version
```

---

### 5. kubectl (Kubernetes CLI)

**Why**: Required for Kubernetes development and deployment.

```powershell
# Using winget
winget install Kubernetes.kubectl

# Verify installation
kubectl version --client
```

---

### 6. jq (JSON Processor)

**Why**: Essential for processing JSON in scripts and command line.

```powershell
# Using winget
winget install jqlang.jq

# Verify installation
jq --version
```

---

### 7. wget (File Downloader)

**Why**: Useful for downloading files in scripts.

```powershell
# Using winget
winget install GNU.Wget2

# Verify installation
wget --version
```

---

### 8. Ninja (Fast Build System)

**Why**: Much faster than Make for C/C++ builds.

```powershell
# Using winget
winget install Ninja-build.Ninja

# Verify installation
ninja --version
```

---

## 🟢 Low Priority (Install If Needed)

### 9. Go (Go Language)

**Why**: Only if you need Go development.

```powershell
# Using winget
winget install GoLang.Go

# Verify installation
go version
```

---

### 10. Java JDK (Java Development)

**Why**: Only if you need Java development.

```powershell
# Using winget (JDK 21 LTS)
winget install Oracle.JDK.21

# Or OpenJDK
winget install Microsoft.OpenJDK.21

# Verify installation
java --version
javac --version
```

---

### 11. MongoDB (NoSQL Database)

**Why**: Only if you need document-based database.

```powershell
# Using winget
winget install MongoDB.Server

# Verify installation
mongod --version
```

---

### 12. TensorFlow (ML Framework)

**Why**: Only if you need TensorFlow (you already have PyTorch).

```powershell
# Using pip
pip install tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 📦 Batch Install Script

Save this as `install_dev_tools.ps1` and run in PowerShell (as Administrator):

```powershell
# High Priority Tools
Write-Host "Installing High Priority Tools..." -ForegroundColor Green
winget install OpenJS.NodeJS.LTS -e --accept-source-agreements --accept-package-agreements
winget install PostgreSQL.PostgreSQL.16 -e --accept-source-agreements --accept-package-agreements
winget install Kitware.CMake -e --accept-source-agreements --accept-package-agreements

# Medium Priority Tools
Write-Host "Installing Medium Priority Tools..." -ForegroundColor Yellow
winget install Redis.Redis -e --accept-source-agreements --accept-package-agreements
winget install Kubernetes.kubectl -e --accept-source-agreements --accept-package-agreements
winget install jqlang.jq -e --accept-source-agreements --accept-package-agreements
winget install GNU.Wget2 -e --accept-source-agreements --accept-package-agreements
winget install Ninja-build.Ninja -e --accept-source-agreements --accept-package-agreements

Write-Host "Installation complete! Please restart your terminal." -ForegroundColor Green
Write-Host "Run 'python run_ultra_fast_analysis.py' to verify installations." -ForegroundColor Cyan
```

**To run**:
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_dev_tools.ps1
```

---

## 🔄 After Installation

### 1. Verify Installations

```powershell
# Check Node.js
node --version
npm --version

# Check PostgreSQL
psql --version

# Check CMake
cmake --version

# Check Redis
redis-cli --version

# Check kubectl
kubectl version --client

# Check jq
jq --version

# Check wget
wget --version

# Check Ninja
ninja --version
```

---

### 2. Re-run Janus Analysis

```powershell
# Run the ultra-fast analysis again
python run_ultra_fast_analysis.py

# This will create a new report showing the newly installed tools
```

---

### 3. Compare Results

```powershell
# Compare before and after
# Old report: capabilities_report_ultra_fast.json
# New report: capabilities_report_ultra_fast.json (will be overwritten)

# Tip: Rename the old report first
Rename-Item capabilities_report_ultra_fast.json capabilities_report_before.json
python run_ultra_fast_analysis.py
# Now you have both: capabilities_report_before.json and capabilities_report_ultra_fast.json
```

---

## 📊 Expected Impact

After installing the high-priority tools, you should see:

| Category | Before | After (Expected) |
|----------|--------|------------------|
| **Languages** | 29 | 35+ (Node.js tools) |
| **Package Managers** | 4 | 7+ (npm, yarn, pnpm) |
| **Databases** | 3 | 6+ (PostgreSQL tools) |
| **Build Tools** | 6 | 9+ (CMake, Ninja) |
| **Total Apps with Capabilities** | 265 | 300+ |

---

## 🆘 Troubleshooting

### winget not found
```powershell
# Install App Installer from Microsoft Store
# Or update Windows to latest version
```

### Permission denied
```powershell
# Run PowerShell as Administrator
# Right-click PowerShell → "Run as Administrator"
```

### Installation fails
```powershell
# Try installing one at a time
# Check winget search results:
winget search nodejs
winget search postgresql
```

### Path not updated
```powershell
# Restart your terminal/PowerShell
# Or manually add to PATH:
$env:Path += ";C:\Program Files\nodejs"
```

---

## 📚 Additional Resources

- **Node.js**: https://nodejs.org/docs/
- **PostgreSQL**: https://www.postgresql.org/docs/
- **CMake**: https://cmake.org/documentation/
- **Redis**: https://redis.io/docs/
- **kubectl**: https://kubernetes.io/docs/reference/kubectl/

---

**Generated**: 2026-04-26  
**Based on**: Janus Dependency Analyzer Ultra-Fast Analysis  
**Platform**: Windows (amd64)
