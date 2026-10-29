@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: setting up ELK. This takes a few minutes...
  python -m venv .venv || (echo Python was not found. Install Python 3.12+ from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt || (echo Installing packages failed. & pause & exit /b 1)
)
echo Starting ELK at http://localhost:8501 ...
.venv\Scripts\python -m streamlit run app.py --server.address localhost
pause
