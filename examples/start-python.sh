#!/bin/bash

# Check if a Python file path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <path_to_python_file>"
  exit 1
fi

PYTHON_FILE=$1

# Check if the file exists
if [ ! -f "$PYTHON_FILE" ]; then
  echo "Error: File '$PYTHON_FILE' not found."
  exit 1
fi

# Determine the operating system
OS=$(uname -s)
echo "Detected OS: $OS"

# Create a virtual environment
VENV_DIR="venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
fi

# Activate the virtual environment
if [ "$OS" == "Darwin" ] || [ "$OS" == "Linux" ]; then
  source "$VENV_DIR/bin/activate"
elif [ "$OS" == "CYGWIN" ] || [ "$OS" == "MINGW" ] || [ "$OS" == "MSYS" ]; then
  source "$VENV_DIR/Scripts/activate"
else
  echo "Unsupported operating system: $OS"
  exit 1
fi

# Install dependencies (if any)
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null || echo "No requirements.txt found, skipping dependency installation."

# Install the MCP Python SDK from GitHub
echo "Installing MCP Python SDK from GitHub..."
pip install git+https://github.com/modelcontextprotocol/python-sdk.git

# Run the Python script and pipe output to standard I/O
echo "Running Python script: $PYTHON_FILE"
python "$PYTHON_FILE"

# Deactivate the virtual environment
echo "Deactivating virtual environment..."
deactivate

# Clean up (optional)
echo "Done. Returning to normal command line."