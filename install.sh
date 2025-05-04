#!/bin/bash

# Install Python
apt-get -y update
apt-get install -y python3-pip python3-venv

echo "📦 Creating virtual environment..."
python3.10 -m venv env
source env/bin/activate

echo "⬇️ Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete."
