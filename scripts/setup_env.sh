#!/bin/bash
# MLA HFT System - Environment Setup Script
# Optimized for Apple M4 (Mac mini 10 cores, 24GB RAM)

set -e

echo "============================================"
echo "MLA HFT System - Environment Setup"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.9"

echo -e "${YELLOW}Checking Python version...${NC}"
echo -e "${GREEN}Python $PYTHON_VERSION found.${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo -e "${YELLOW}Project directory: $PROJECT_DIR${NC}"

# Create virtual environment
VENV_DIR="$PROJECT_DIR/.venv"

if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing...${NC}"
    rm -rf "$VENV_DIR"
fi

echo -e "${YELLOW}Creating virtual environment...${NC}"
python3 -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip wheel setuptools

# Install dependencies with ARM optimizations
echo -e "${YELLOW}Installing dependencies (optimized for Apple Silicon)...${NC}"

# Set environment variables for ARM optimization
export ARCHFLAGS="-arch arm64"
export CMAKE_OSX_ARCHITECTURES="arm64"

# Install PyTorch with MPS (Metal Performance Shaders) support for M4
pip install torch torchvision torchaudio

# Install remaining dependencies
pip install -r "$PROJECT_DIR/requirements.txt"

# Create necessary directories
echo -e "${YELLOW}Creating project directories...${NC}"
mkdir -p "$PROJECT_DIR/data/historical"
mkdir -p "$PROJECT_DIR/data/backtests"
mkdir -p "$PROJECT_DIR/data/trading"
mkdir -p "$PROJECT_DIR/data/backups"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/models"

# Create .env template if not exists
ENV_FILE="$PROJECT_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating .env template...${NC}"
    cat > "$ENV_FILE" << EOF
# Hyperliquid API Configuration
# IMPORTANT: Fill in your actual credentials
HYPERLIQUID_WALLET_ADDRESS=your_wallet_address_here
HYPERLIQUID_PRIVATE_KEY=your_private_key_here

# Optional: Redis for caching (if using)
REDIS_URL=redis://localhost:6379

# Logging level
LOG_LEVEL=INFO

# Environment
ENVIRONMENT=production
EOF
    echo -e "${RED}IMPORTANT: Edit .env file with your Hyperliquid credentials!${NC}"
fi

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "
import torch
import pandas
import numpy
import websockets
print(f'PyTorch: {torch.__version__}')
print(f'MPS Available: {torch.backends.mps.is_available()}')
print(f'Pandas: {pandas.__version__}')
print(f'NumPy: {numpy.__version__}')
print('All core dependencies installed successfully!')
"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your Hyperliquid credentials"
echo "2. Activate the virtual environment:"
echo "   source $VENV_DIR/bin/activate"
echo "3. Run the system:"
echo "   python scripts/launch.py"
echo ""
