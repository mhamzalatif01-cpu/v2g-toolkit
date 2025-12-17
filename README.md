# V2G Techno-Economic Toolkit (Streamlit Prototype)

This is a lightweight, GUI-based prototype inspired by HOMER workflows, designed for PV + BESS + EV (V2G) techno-economic analysis.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (Option B)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, choose:
   - Repo: your repo
   - Branch: main
   - Main file: app.py
3. Deploy.

## Input CSV formats
### Load (kW)
Either:
- one numeric column (8760 rows), or
- columns: `datetime, value`

### PV
Either:
- PV profile CSV (kW), same format as load, OR
- Solar GHI profile (W/m^2) as `value` (8760 rows). Prototype estimates PV with a simple model.

### EV availability (optional)
CSV with numeric column `available` (0/1), or a single numeric column (0/1), 8760 rows.
