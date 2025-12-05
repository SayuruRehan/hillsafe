#!/bin/bash

# Start HillSafe backend server
echo "Starting HillSafe backend..."

# Change to backend directory
cd backend

# Run the FastAPI server with uvicorn
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload