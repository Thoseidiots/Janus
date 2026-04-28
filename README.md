# Janus Dependency Analyzer

A cross-platform system that discovers installed applications, analyzes their capabilities, maps current Janus dependencies, and generates prioritized implementation roadmaps for internalizing external dependencies.

## Overview

The Janus Dependency Analyzer helps Janus become more self-reliant by identifying external applications that could be replaced with internal implementations. The system provides:

- **Cross-platform application discovery** (Windows, macOS, Linux)
- **Capability analysis** using multiple strategies
- **Dependency mapping** to identify current usage patterns
- **Priority scoring** for implementation planning
- **Implementation roadmaps** with effort estimates

## Features

### Application Discovery
- Registry scanning (Windows)
- Package manager integration (apt, yum, pacman, Homebrew, etc.)
- Standard installation directory scanning
- Portable application detection (AppImages, etc.)

### Capability Analysis
- Documentation parsing (README, man pages)
- Help text analysis (--help, -h flags)
- Command-line interface analysis
- API endpoint detection (REST, GraphQL)

### Security & Privacy
- Respects system access controls
- Encrypts stored metadata
- Comprehensive audit logging
- Minimal privilege operation

## Installation

### From Source

```bash
git clone https://github.com/janus-ai/dependency-analyzer.git
cd dependency-analyzer
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/janus-ai/dependency-analyzer.git
cd dependency-analyzer
pip install -e ".[dev,test]"
```

## Quick Start

### Basic System Scan

```bash
# Scan system for installed applications
janus-analyzer scan

# Scan with verbose output
janus-analyzer -v scan

# Save results to JSON file
janus-analyzer scan --format json --output results.json
```

### Analyze Specific Application

```bash
# Analyze capabilities of a specific application
janus-analyzer analyze "firefox"
janus-analyzer analyze "git"
```

### System Information

```bash
# Display system and configuration information
janus-analyzer info
```

## Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Presentation Layer                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ CLI Interface│  │ REST API    │  │ Report Generator    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                   Business Logic Layer                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │System Scanner│  │Capability   │  │ Priority Engine     │  │
│  │             │  │Analyzer     │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Dependency   │  │Roadmap      │  │ Configuration       │  │
│  │Mapper       │  │Generator    │  │ Manager             │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                      Data Layer                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Application  │  │Configuration│  │ Analysis Cache      │  │
│  │Catalog      │  │Store        │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────────┐
│                    Platform Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │Windows      │  │macOS        │  │ Linux Scanner       │  │
│  │Scanner      │  │Scanner      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### System Scanner
Orchestrates platform-specific discovery mechanisms to build comprehensive application inventories.

### Capability Analyzer
Uses multiple analysis strategies to identify application capabilities:
- **Documentation Analysis**: Parses README files, man pages, embedded docs
- **Help Text Analysis**: Executes apps with --help flags
- **CLI Analysis**: Analyzes command-line argument patterns
- **API Analysis**: Scans for REST/GraphQL endpoints

### Platform Scanners
Platform-specific implementations for:
- **Windows**: Registry scanning, Program Files, Windows Store
- **macOS**: Applications folder, Homebrew, LaunchServices
- **Linux**: Package managers, /usr/bin, AppImages, Flatpak

## Configuration

Create a configuration file to customize behavior:

```yaml
# Scanning configuration
scan_exclusion_patterns:
  - "*/temp/*"
  - "*/cache/*"
scan_timeout_seconds: 300
max_applications_per_scan: 10000

# Analysis configuration
min_confidence_threshold: 0.5
analysis_timeout_seconds: 60

# Priority weights
priority_weights:
  usage: 0.3
  complexity: 0.2
  security: 0.25
  performance: 0.25

# Security settings
respect_access_controls: true
encrypt_stored_data: true
audit_logging_enabled: true
```

## Development

### Project Structure

```
janus_dependency_analyzer/
├── core/                   # Core data models and interfaces
│   ├── models.py          # Data models (Application, Capability, etc.)
│   └── interfaces.py      # Abstract base classes
├── scanners/              # Platform-specific scanners
│   ├── system_scanner.py  # Main scanner orchestrator
│   ├── windows_scanner.py # Windows-specific implementation
│   ├── macos_scanner.py   # macOS-specific implementation
│   └── linux_scanner.py   # Linux-specific implementation
├── analyzers/             # Capability analysis components
│   ├── capability_analyzer.py  # Main analyzer
│   └── strategies/        # Analysis strategies
│       ├── documentation_strategy.py
│       ├── help_text_strategy.py
│       ├── cli_strategy.py
│       └── api_strategy.py
└── cli.py                 # Command-line interface
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=janus_dependency_analyzer

# Run specific test file
pytest tests/test_models.py

# Run property-based tests
pytest -m property
```

### Code Quality

```bash
# Format code
black janus_dependency_analyzer/

# Sort imports
isort janus_dependency_analyzer/

# Type checking
mypy janus_dependency_analyzer/

# Linting
flake8 janus_dependency_analyzer/
```

## API Reference

### Core Models

#### Application
Represents an installed application with metadata:
```python
@dataclass
class Application:
    id: str
    name: str
    version: str
    installation_path: Path
    executable_path: Path
    platform: Platform
    metadata: ApplicationMetadata
    is_accessible: bool
```

#### Capability
Represents a capability provided by an application:
```python
@dataclass
class Capability:
    id: str
    application_id: str
    name: str
    category: CapabilityCategory
    interface_type: InterfaceType
    confidence_score: float
    parameters: List[Parameter]
```

### Scanner Interface

```python
class SystemScanner(ABC):
    @abstractmethod
    def scan_full(self) -> ScanResult:
        """Perform complete system scan"""
        
    @abstractmethod
    def scan_incremental(self, last_scan_time: datetime) -> ScanResult:
        """Scan for changes since last scan"""
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests
- Update documentation for new features
- Use property-based testing for complex logic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Roadmap

- [ ] **Phase 1**: Core scanning and analysis (Current)
- [ ] **Phase 2**: Dependency mapping and priority scoring
- [ ] **Phase 3**: Implementation roadmap generation
- [ ] **Phase 4**: Web interface and advanced reporting
- [ ] **Phase 5**: Integration with CI/CD pipelines

## Support

- **Documentation**: [https://janus-dependency-analyzer.readthedocs.io/](https://janus-dependency-analyzer.readthedocs.io/)
- **Issues**: [GitHub Issues](https://github.com/janus-ai/dependency-analyzer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/janus-ai/dependency-analyzer/discussions)

## Acknowledgments

- Built for the Janus AI system
- Inspired by dependency analysis tools in the software engineering community
- Thanks to all contributors and the open-source community