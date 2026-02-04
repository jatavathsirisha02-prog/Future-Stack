@echo off
REM Run FastAPI backend (start first)
echo Starting FastAPI backend on http://127.0.0.1:8000
start "FastAPI" cmd /k uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
timeout /t 3
REM Run Streamlit UI
echo Starting Streamlit UI on http://localhost:8501
streamlit run app.py
