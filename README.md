
# AI Statistical Assistant — MVP (Local, Laptop-Only)

This is a minimal prototype you can run on your laptop. It lets a statistician upload data,
choose a *tool* (Python for now), pick an analysis, and get results with interpretation.

> Roadmap hooks are included for R backend; Stata/SPSS would be added later via server-side calls.

## 1) Prerequisites
- Python 3.9+ installed (`python --version`)
- (Optional for future) R 4.1+ if/when you enable the R backend

## 2) Quick Start
```bash
# (Windows PowerShell or macOS/Linux terminal)
python -m venv .venv
# Windows:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Streamlit will open in your browser (usually http://localhost:8501).
Upload your CSV/Excel, select options, and run.

## 3) What’s Inside
- `app.py` — Streamlit UI (upload, choose tool, pick analysis)
- `analysis_python.py` — Python implementations (EDA, t-test, OLS, Logistic)
- `requirements.txt` — Python dependencies

## 4) Notes
- CSV/Excel only. Clean your column names (no weird characters) for best results.
- Python backend uses `pandas`, `numpy`, `statsmodels`, and `scikit-learn`.
- Plots use `matplotlib`.

## 5) Roadmap
- Add R backend (via `subprocess` calling `Rscript`) — hook already in UI.
- Add Stata/SPSS backends (requires licensed server/VM or user-provided executables).
- Add auto-generated report (PDF/Word) and shareable links.
- Add LLM layer to translate natural language into analysis plans & code.
