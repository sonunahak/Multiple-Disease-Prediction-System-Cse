# Multiple Disease Prediction System

A Streamlit-based machine learning application for predicting the likelihood of diabetes, heart disease, Parkinson's disease, and breast cancer from user-provided measurements.

## Features

- Diabetes prediction
- Heart disease prediction
- Parkinson's disease prediction
- Breast cancer prediction
- SQLite-backed patient prediction history
- Sidebar navigation with `streamlit-option-menu`

## Requirements

- Python 3.10 or newer
- Windows, macOS, or Linux

## Installation

Create and activate a virtual environment from the project root:

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the application

From the project root, with the virtual environment activated:

```bash
python -m streamlit run scripts/app.py
```

Then open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

For a headless launch on a fixed port:

```bash
python -m streamlit run scripts/app.py --server.headless true --server.port 8501
```

## Project structure

```text
.
├── datasets/       # CSV datasets used by the notebooks
├── models/         # Saved machine learning models
├── notebooks/      # Model training and exploration notebooks
├── scripts/
│   └── app.py      # Streamlit application entry point
├── requirements.txt
└── patient_records.db  # SQLite history database, created by the app
```

## Notes

- The application loads model files from the `models/` directory relative to `scripts/app.py`.
- Prediction history is stored in `patient_records.db` in the current working directory.
- This project is for educational and demonstration purposes. Predictions are not medical diagnoses.
