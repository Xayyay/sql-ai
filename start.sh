#!/bin/bash
echo "============================================"
echo "  SQL AI Summarizer - Starting..."
echo "============================================"
echo

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "ERROR: Python 3 is not installed."
    exit 1
fi

# Install deps if needed
if ! python3 -c "import fastapi" &>/dev/null; then
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
fi

echo "Open your browser at: http://localhost:8000"
echo "Press Ctrl+C to stop."
echo

# Allow TLS 1.0 for SQL Server 2008 R2 compatibility
export OPENSSL_CONF=/etc/ssl/openssl_mssql.cnf

python3 app.py
