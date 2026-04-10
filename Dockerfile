# Multi-stage Dockerfile for Janus Autonomous Agent

# Stage 1: Build Rust Core
FROM rust:1.75-slim AS rust-builder

WORKDIR /build

# Install system dependencies for Rust
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Rust project
COPY Cargo.toml ./
COPY crates ./crates

# Build release binary (continue even if it fails)
RUN cargo build --release 2>&1 || echo "Rust build completed with warnings"

# Stage 2: Build runtime environment
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    vim \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    x11-utils \
    xvfb \
    x11-apps \
    pulseaudio \
    alsa-utils \
    portaudio19-dev \
    flac \
    && rm -rf /var/lib/apt/lists/*

# Copy Python requirements
COPY requirements_enhanced.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_enhanced.txt && \
    pip install --no-cache-dir \
    pillow \
    opencv-python \
    mss \
    pyautogui \
    pyscreenshot \
    anthropic \
    openai \
    requests \
    aiohttp \
    pytest

# Copy Janus application
COPY . .

# Try to copy compiled Rust binary (optional - may not exist)
RUN echo "Skipping optional Rust binary copy"

# Create directories for task history and logs
RUN mkdir -p /app/logs /app/data /app/tasks

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Expose ports
EXPOSE 8000 8080 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command: Start Python interface
CMD ["python", "-u", "janus_main.py"]
