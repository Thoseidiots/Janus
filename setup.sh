#!/bin/bash
# Quick start script for Janus Autonomous Agent

set -e

echo "🤖 Janus Autonomous Agent - Setup"
echo "=================================="

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✓ Docker found"
echo "✓ Docker Compose found"

# Create data directories
mkdir -p data logs tasks

echo "✓ Created data directories"

# Build Docker image
echo ""
echo "Building Janus Docker image..."
docker-compose build --no-cache

echo ""
echo "✓ Build complete!"
echo ""
echo "To start Janus, run:"
echo "  docker-compose up -d"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f"
echo ""
echo "To stop:"
echo "  docker-compose down"
echo ""
echo "To execute a command in the container:"
echo "  docker-compose exec janus-agent python vision_automation.py"
