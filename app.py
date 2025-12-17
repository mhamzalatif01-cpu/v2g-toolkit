
import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="V2G Techno-Economic Toolkit (Prototype)", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def read_timeseries_csv(upload, value_col_hint=None):
    """
    Accepts either:
      - a CSV with columns: datetime, value (or any single numeric column)
      - OR an 8760 single-column CSV (value only)
    Returns: pandas Series indexed by hour 0..n-1 and a DataFrame with hour index.
    """
    df = pd.read_csv(upload)
    # Strip whitespace columns
    df.columns = [c.strip() for c in df.columns]

    # If there's a datetime column, parse and sort
    dt_col = None
    for c in df.columns:
        if c.lower() in ["datetime", "time", "timestamp", "date_time", "date"]:
            dt_col = c
            break

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col]).sort_values(dt_col)
        df = df.reset_index(drop=True)

    # Choose value column
    val_cols = [c for c in df.columns if c != dt_col]
    num_cols = [c for c in val_cols if pd.api.types.is_numeric_dtype(df[c])]

    if value_col_hint and value_col_hint in df.columns:
        vcol = value_col_hint
    elif len(num_cols) == 1:
        vcol = num_cols[0]
    elif "value" in [c.lower() for c in df.columns]:
        vcol = [c for c in df.columns if c.lower() == "value"][0]
    else:
        # last resort: pick first numeric
        if len(num_cols) == 0:
            raise ValueError("No numeric column found. Please include a numeric column like 'value'.")
        vcol = num_cols[0]

    series = df[vcol].astype(float).copy()
    series = series.reset_index(drop=True)

    # Build hour index 0..n-1
    out = pd.DataFrame({"hour": np.arange(len(series)), "value": series.values})
    return out

def build_tou_vector(n_hours, peak_hours, shoulder_hours, offpeak_hours, peak_rate, shoulder_rate, offpeak_rate):
    rates = np.full(n_hours, offpeak_rate, dtype=float)

    def set_hours(hours, rate):
        for h in hours:
            if 0 <= h <= 23:
                rates[np.arange(n_hours) % 24 == h] = rate

    set_hours(offpeak_hours, offpeak_rate)
    set_hours(shoulder_hours, shoulder_rate)
    set_hours(peak_hours, peak_rate)
    return rates

def crf(discount_rate, years):
    if years <= 0:
        return np.nan
    r = discount_rate
    if abs(r) < 1e-12:
        return 1.0 / years
    return (r * (1 + r) ** years) / ((1 + r) ** years - 1)

def present_worth_factor(discount_rate, year):
    return 1.0 / ((1.0 + discount_rate) ** year)

# ----------------------------
# Sidebar inputs
# ----------------------------
st.sidebar.title("Inputs")

st.sidebar.subheader("Simulation Horizon")
n_hours = st.sidebar.selectbox("Time resolution (hours)", options=[8760, 35040], index=0,
                               help="8760=hourly, 35040=15-min (as hours*4). Prototype works best with 8760.")

st.sidebar.subheader("Upload Time Series")
load_file = st.sidebar.file_uploader("Building Load (kW) CSV", type=["csv"])
solar_file = st.sidebar.file_uploader("Solar Resource CSV (GHI W/m²) (optional)", type=["csv"])
pv_profile_file = st.sidebar.file_uploader("PV Power Profile (kW) CSV (optional, overrides GHI)", type=["csv"])

st.sidebar.caption("Tip: If you upload PV Power Profile, it will be used directly. Otherwise PV is estimated from GHI.")

# PV
st.sidebar.subheader("PV System")
pv_kw = st.sidebar.number_input("PV capacity (kWdc)", min_value=0.0, value=300.0, step=10.0)
pv_derate = st.sidebar.slider("PV derate factor", 0.70, 1.00, 0.88, 0.01)
inverter_kw = st.sidebar.number_input("Inverter capacity (kWac)", min_value=0.0, value=300.0, step=10.0)
inv_eff = st.sidebar.slider("Inverter efficiency", 0.85, 0.99, 0.96, 0.01)

# Stationary BESS
st.sidebar.subheader("Stationary BESS")
bess_kwh = st.sidebar.number_input("BESS energy (kWh)", min_value=0.0, value=1000.0, step=50.0)
bess_kw = st.sidebar.number_input("BESS power (kW)", min_value=0.0, value=250.0, step=10.0)
bess_eff_rt = st.sidebar.slider("BESS round-trip efficiency", 0.70, 0.99, 0.90, 0.01)
bess_min_soc = st.sidebar.slider("BESS min SOC", 0.0, 0.5, 0.1, 0.01)
bess_init_soc = st.sidebar.slider("BESS initial SOC", 0.0, 1.0, 0.5, 0.01)

# EV Fleet (aggregated for v1)
st.sidebar.subheader("EV Fleet (Aggregated V2G)")
use_ev = st.sidebar.checkbox("Enable EV fleet module", value=True)
ev_kwh = st.sidebar.number_input("EV aggregated energy (kWh)", min_value=0.0, value=300.0, step=10.0, disabled=not use_ev)
ev_kw = st.sidebar.number_input("EV aggregated power (kW)", min_value=0.0, value=100.0, step=10.0, disabled=not use_ev)
ev_eff_rt = st.sidebar.slider("EV round-trip efficiency", 0.70, 0.99, 0.90, 0.01, disabled=not use_ev)
ev_min_soc = st.sidebar.slider("EV min SOC (mobility reserve)", 0.0, 0.8, 0.4, 0.01, disabled=not use_ev)
ev_init_soc = st.sidebar.slider("EV initial SOC", 0.0, 1.0, 0.6, 0.01, disabled=not use_ev)
ev_avail_file = st.sidebar.file_uploader("EV Availability CSV (optional)", type=["csv"], disabled=not use_ev)
st.sidebar.caption("EV availability CSV: one numeric column (0/1 availability) OR column 'available'.")

# Tariff
st.sidebar.subheader("Tariff (Malaysia-style TOU Prototype)")
peak_rate = st.sidebar.number_input("Peak import rate (RM/kWh)", min_value=0.0, value=0.60, step=0.01)
shoulder_rate = st.sidebar.number_input("Shoulder import rate (RM/kWh)", min_value=0.0, value=0.45, step=0.01)
offpeak_rate = st.sidebar.number_input("Off-peak import rate (RM/kWh)", min_value=0.0, value=0.30, step=0.01)
export_rate = st.sidebar.number_input("Export (feed-out) rate (RM/kWh)", min_value=0.0, value=0.10, step=0.01)

peak_hours = st.sidebar.multiselect("Peak hours (0–23)", options=list(range(24)), default=[12,13,14,15,16,17,18])
shoulder_hours = st.sidebar.multiselect("Shoulder hours (0–23)", options=list(range(24)), default=[8,9,10,11,19,20,21])
offpeak_hours = [h for h in range(24) if h not in set(peak_hours) and h not in set(shoulder_hours)]
st.sidebar.caption(f"Off-peak hours auto: {offpeak_hours}")

# Finance
st.sidebar.subheader("Finance")
project_years = st.sidebar.number_input("Project life (years)", min_value=1, value=25, step=1)
discount_rate = st.sidebar.number_input("Discount rate (real)", min_value=0.0, value=0.06, step=0.005, format="%.3f")
pv_capex_per_kw = st.sidebar.number_input("PV CAPEX (RM/kW)", min_value=0.0, value=3500.0, step=100.0)
inv_capex_per_kw = st.sidebar.number_input("Inverter CAPEX (RM/kW)", min_value=0.0, value=800.0, step=50.0)
bess_capex_per_kwh = st.sidebar.number_input("BESS CAPEX (RM/kWh)", min_value=0.0, value=1500.0, step=50.0)
ev_capex_per_kwh = st.sidebar.number_input("EV degr. cost proxy (RM/kWh throughput)", min_value=0.0, value=0.08, step=0.01,
                                           help="Operational degradation cost proxy; NOT an upfront cost.")
fixed_om_per_year = st.sidebar.number_input("Fixed O&M (RM/year)", min_value=0.0, value=25000.0, step=1000.0)
pv_life = st.sidebar.number_input("PV lifetime (years)", min_value=1, value=25, step=1)
inv_life = st.sidebar.number_input("Inverter lifetime (years)", min_value=1, value=12, step=1)
bess_life = st.sidebar.number_input("BESS lifetime (years)", min_value=1, value=10, step=1)

# Run
st.sidebar.subheader("Run")
run_btn = st.sidebar.button("Run Simulation")

# ----------------------------
# Main
# ----------------------------
st.title("V2G Techno-Economic Toolkit (Prototype)")
st.caption("A lightweight, open-source-style prototype that mimics HOMER-like flows but adds EV fleet flexibility. "
           "Rule-based dispatch in v1; designed to be extended to MILP dispatch later.")

colA, colB = st.columns([1.2, 1.0], gap="large")

with colA:
    st.subheader("1) Data Preview")
    if load_file is None:
        st.info("Upload a Building Load (kW) CSV to begin.")
    else:
        load_df = read_timeseries_csv(load_file)
        if len(load_df) != n_hours:
            st.warning(f"Load has {len(load_df)} rows. You selected {n_hours}. The app will use the uploaded length.")
            n = len(load_df)
        else:
            n = n_hours

        st.write("Load (first 10 rows)")
        st.dataframe(load_df.head(10), use_container_width=True)

        # PV profile
        pv_power = None
        if pv_profile_file is not None:
            pv_df = read_timeseries_csv(pv_profile_file)
            if len(pv_df) != n:
                st.warning(f"PV profile has {len(pv_df)} rows; load has {n}. Will align to min length.")
            m = min(n, len(pv_df))
            n = m
            load_df = load_df.iloc[:n].reset_index(drop=True)
            pv_df = pv_df.iloc[:n].reset_index(drop=True)
            pv_power = pv_df["value"].to_numpy(dtype=float)
            st.write("PV Power Profile (first 10 rows)")
            st.dataframe(pv_df.head(10), use_container_width=True)

        else:
            # Estimate PV from GHI if provided, else use a simple synthetic daily curve (for demo)
            if solar_file is not None:
                ghi_df = read_timeseries_csv(solar_file)
                if len(ghi_df) != n:
                    st.warning(f"GHI has {len(ghi_df)} rows; load has {n}. Will align to min length.")
                m = min(n, len(ghi_df))
                n = m
                load_df = load_df.iloc[:n].reset_index(drop=True)
                ghi_df = ghi_df.iloc[:n].reset_index(drop=True)
                ghi = ghi_df["value"].clip(lower=0).to_numpy(dtype=float)  # W/m2
            else:
                # synthetic, for demo only
                t = np.arange(n)
                hod = t % 24
                ghi = (np.maximum(0, np.sin((hod-6)/12*np.pi)) * 800.0).astype(float)

            pv_dc = pv_kw * (ghi / 1000.0) * pv_derate  # kWdc
            pv_ac = np.minimum(pv_dc, inverter_kw) * inv_eff  # kWac delivered
            pv_power = pv_ac

        # EV availability
        if use_ev:
            if ev_avail_file is not None:
                avdf = pd.read_csv(ev_avail_file)
                avdf.columns = [c.strip() for c in avdf.columns]
                if "available" in [c.lower() for c in avdf.columns]:
                    colname = [c for c in avdf.columns if c.lower() == "available"][0]
                    avail = avdf[colname].astype(float).to_numpy()
                else:
                    # first numeric column
                    num_cols = [c for c in avdf.columns if pd.api.types.is_numeric_dtype(avdf[c])]
                    if not num_cols:
                        st.warning("EV availability file has no numeric column. Assuming always available.")
                        avail = np.ones(n)
                    else:
                        avail = avdf[num_cols[0]].astype(float).to_numpy()
                if len(avail) != n:
                    st.warning(f"EV availability has {len(avail)} rows; using min length with load.")
                m = min(n, len(avail))
                n = m
                load_df = load_df.iloc[:n].reset_index(drop=True)
                pv_power = pv_power[:n]
                avail = avail[:n]
            else:
                # default: weekdays 9–17 available
                t = np.arange(n)
                hod = t % 24
                avail = ((hod >= 9) & (hod <= 17)).astype(float)
        else:
            avail = np.zeros(n)

        st.success(f"Using {n} timesteps.")

with colB:
    st.subheader("2) Quick Inputs Summary")
    if load_file is not None:
        st.markdown(
            f"""
- **PV**: {pv_kw:.1f} kWdc, derate {pv_derate:.2f}, inverter {inverter_kw:.1f} kWac @ {inv_eff:.2f}
- **BESS**: {bess_kwh:.1f} kWh, {bess_kw:.1f} kW, RT eff {bess_eff_rt:.2f}
- **EV**: {"ON" if use_ev else "OFF"} ({ev_kwh:.1f} kWh, {ev_kw:.1f} kW, RT eff {ev_eff_rt:.2f})
- **Tariff**: peak {peak_rate:.2f}, shoulder {shoulder_rate:.2f}, off-peak {offpeak_rate:.2f} RM/kWh; export {export_rate:.2f}
- **Finance**: {project_years} years, discount {discount_rate:.3f}
            """
        )
    else:
        st.info("Upload data to see summary.")

# ----------------------------
# Simulation core (rule-based dispatch)
# ----------------------------
def simulate_rule_based(load_kw, pv_kw_ac, import_rate_vec, export_rate,
                        bess_kwh, bess_kw, bess_eff_rt, bess_min_soc, bess_init_soc,
                        ev_enabled, ev_kwh, ev_kw, ev_eff_rt, ev_min_soc, ev_init_soc, ev_avail,
                        ev_deg_cost_per_kwh_throughput):
    n = len(load_kw)
    dt = 1.0  # hours

    # Split RT efficiency into charge/discharge (sqrt model)
    bess_eff = math.sqrt(max(bess_eff_rt, 1e-6))
    ev_eff = math.sqrt(max(ev_eff_rt, 1e-6))

    # SOC states (energy in kWh)
    bess_e = np.zeros(n+1)
    ev_e = np.zeros(n+1)
    bess_e[0] = bess_init_soc * bess_kwh
    ev_e[0] = ev_init_soc * ev_kwh

    # Outputs
    pv_to_load = np.zeros(n)
    pv_to_bess = np.zeros(n)
    pv_to_ev = np.zeros(n)
    pv_export = np.zeros(n)

    bess_dis = np.zeros(n)  # kW to load
    bess_chg = np.zeros(n)  # kW from PV
    ev_dis = np.zeros(n)
    ev_chg = np.zeros(n)

    grid_import = np.zeros(n)
    grid_export = np.zeros(n)  # equals pv_export for v1

    unmet = np.zeros(n)

    # Throughput for degradation cost
    ev_throughput_kwh = 0.0

    for t in range(n):
        load = max(load_kw[t], 0.0)
        pv = max(pv_kw_ac[t], 0.0)

        # Step 1: PV serves load
        pv_served = min(pv, load)
        pv_to_load[t] = pv_served
        load_rem = load - pv_served
        pv_rem = pv - pv_served

        # Step 2: Charge stationary BESS with remaining PV
        if bess_kwh > 0 and bess_kw > 0 and pv_rem > 0:
            bess_room = max(0.0, (1.0 - bess_min_soc) * bess_kwh - bess_e[t])  # max energy above min? Actually room to full
            bess_room = max(0.0, bess_kwh - bess_e[t])
            max_chg_kw = min(bess_kw, pv_rem)
            # energy added = chg_kw * eff * dt, limited by room
            chg_kw = min(max_chg_kw, bess_room / (bess_eff * dt + 1e-12))
            bess_chg[t] = chg_kw
            pv_to_bess[t] = chg_kw
            bess_e[t+1] = bess_e[t] + chg_kw * bess_eff * dt
            pv_rem -= chg_kw
        else:
            bess_e[t+1] = bess_e[t]

        # Step 3: Charge EV fleet with remaining PV (if available)
        if ev_enabled and ev_kwh > 0 and ev_kw > 0 and pv_rem > 0 and ev_avail[t] > 0.5:
            ev_room = max(0.0, ev_kwh - ev_e[t])
            max_chg_kw = min(ev_kw, pv_rem)
            chg_kw = min(max_chg_kw, ev_room / (ev_eff * dt + 1e-12))
            ev_chg[t] = chg_kw
            pv_to_ev[t] = chg_kw
            ev_e[t+1] = ev_e[t] + chg_kw * ev_eff * dt
            pv_rem -= chg_kw
            ev_throughput_kwh += chg_kw * dt  # charging throughput
        else:
            ev_e[t+1] = ev_e[t]

        # Step 4: Export remaining PV
        pv_export[t] = pv_rem
        grid_export[t] = pv_rem

        # Now meet remaining load with BESS then EV then grid
        # Discharge BESS
        if load_rem > 1e-9 and bess_kwh > 0 and bess_kw > 0:
            bess_min_e = bess_min_soc * bess_kwh
            available_e = max(0.0, bess_e[t+1] - bess_min_e)  # use updated after charge
            max_dis_kw = min(bess_kw, load_rem)
            dis_kw = min(max_dis_kw, available_e * bess_eff / (dt + 1e-12))  # discharge power limited by available energy
            bess_dis[t] = dis_kw
            bess_e[t+1] = bess_e[t+1] - (dis_kw / (bess_eff + 1e-12)) * dt
            load_rem -= dis_kw

        # Discharge EV
        if load_rem > 1e-9 and ev_enabled and ev_kwh > 0 and ev_kw > 0 and ev_avail[t] > 0.5:
            ev_min_e = ev_min_soc * ev_kwh
            available_e = max(0.0, ev_e[t+1] - ev_min_e)
            max_dis_kw = min(ev_kw, load_rem)
            dis_kw = min(max_dis_kw, available_e * ev_eff / (dt + 1e-12))
            ev_dis[t] = dis_kw
            ev_e[t+1] = ev_e[t+1] - (dis_kw / (ev_eff + 1e-12)) * dt
            load_rem -= dis_kw
            ev_throughput_kwh += dis_kw * dt  # discharging throughput

        # Grid import for remainder
        if load_rem > 1e-9:
            grid_import[t] = load_rem
            load_rem = 0.0

        # Unmet (should be 0 in grid-connected case)
        if load_rem > 1e-6:
            unmet[t] = load_rem

    # Billing
    import_cost = float(np.sum(grid_import * import_rate_vec))
    export_revenue = float(np.sum(grid_export) * export_rate)

    ev_deg_cost = float(ev_throughput_kwh * ev_deg_cost_per_kwh_throughput)

    results = pd.DataFrame({
        "hour": np.arange(n),
        "load_kW": load_kw[:n],
        "pv_kW": pv_kw_ac[:n],
        "pv_to_load_kW": pv_to_load,
        "pv_to_bess_kW": pv_to_bess,
        "pv_to_ev_kW": pv_to_ev,
        "grid_import_kW": grid_import,
        "grid_export_kW": grid_export,
        "bess_chg_kW": bess_chg,
        "bess_dis_kW": bess_dis,
        "ev_chg_kW": ev_chg,
        "ev_dis_kW": ev_dis,
        "bess_energy_kWh": bess_e[1:],
        "ev_energy_kWh": ev_e[1:],
    })

    summary = {
        "import_kWh": float(np.sum(grid_import)),
        "export_kWh": float(np.sum(grid_export)),
        "pv_gen_kWh": float(np.sum(pv_kw_ac)),
        "load_kWh": float(np.sum(load_kw)),
        "import_cost_RM": import_cost,
        "export_revenue_RM": export_revenue,
        "ev_degradation_cost_RM": ev_deg_cost
    }
    return results, summary

# ----------------------------
# Finance computations
# ----------------------------
def compute_finance(summary, pv_kw, inverter_kw, bess_kwh,
                    pv_capex_per_kw, inv_capex_per_kw, bess_capex_per_kwh,
                    fixed_om_per_year,
                    project_years, discount_rate,
                    pv_life, inv_life, bess_life,
                    baseline_import_cost_per_year):
    # CAPEX
    capex_pv = pv_kw * pv_capex_per_kw
    capex_inv = inverter_kw * inv_capex_per_kw
    capex_bess = bess_kwh * bess_capex_per_kwh
    capex_total = capex_pv + capex_inv + capex_bess

    # Replacement present worth (simple: replace at multiples of lifetime within project)
    def repl_pw(component_capex, life_years):
        pw = 0.0
        y = life_years
        while y < project_years + 1e-9:
            pw += component_capex * (1.0) * present_worth_factor(discount_rate, int(round(y)))
            y += life_years
        return pw

    repl = repl_pw(capex_inv, inv_life) + repl_pw(capex_bess, bess_life)  # PV life usually equals project
    # O&M present worth (uniform series)
    om_pw = 0.0
    for y in range(1, project_years + 1):
        om_pw += fixed_om_per_year * present_worth_factor(discount_rate, y)

    # Grid net cost per year in "first year" terms from simulation summary
    net_grid_cost = summary["import_cost_RM"] - summary["export_revenue_RM"]
    annual_ev_deg = summary.get("ev_degradation_cost_RM", 0.0)

    # Assume same each year (prototype)
    net_operating_pw = 0.0
    for y in range(1, project_years + 1):
        net_operating_pw += (net_grid_cost + annual_ev_deg) * present_worth_factor(discount_rate, y)

    npc = capex_total + repl + om_pw + net_operating_pw

    # LCOE (annualized NPC / annual load served)
    annual_load_kwh = summary["load_kWh"]
    ann = npc * crf(discount_rate, project_years)
    lcoe = ann / max(annual_load_kwh, 1e-9)

    # Payback (simple): capex / annual savings vs baseline
    annual_savings = max(0.0, baseline_import_cost_per_year - (net_grid_cost + annual_ev_deg + fixed_om_per_year))
    payback = (capex_total / annual_savings) if annual_savings > 1e-9 else np.inf

    return {
        "CAPEX_RM": capex_total,
        "CAPEX_PV_RM": capex_pv,
        "CAPEX_Inverter_RM": capex_inv,
        "CAPEX_BESS_RM": capex_bess,
        "Replacement_PW_RM": repl,
        "OM_PW_RM": om_pw,
        "Operating_PW_RM": net_operating_pw,
        "NPC_RM": npc,
        "Annualized_Cost_RM_per_year": ann,
        "LCOE_RM_per_kWh": lcoe,
        "Simple_Payback_years": payback,
        "Baseline_Grid_Cost_RM_per_year": baseline_import_cost_per_year,
        "Net_Grid_Cost_RM_per_year": net_grid_cost
    }

# ----------------------------
# Run
# ----------------------------
if run_btn and load_file is not None:
    load_kw = load_df["value"].to_numpy(dtype=float)[:n]
    pv_kw_ac = pv_power[:n]

    import_rate_vec = build_tou_vector(n, peak_hours, shoulder_hours, offpeak_hours,
                                       peak_rate, shoulder_rate, offpeak_rate)

    # Baseline (no PV/BESS/EV): all load imported
    baseline_import_cost = float(np.sum(load_kw * import_rate_vec))

    results, summary = simulate_rule_based(
        load_kw=load_kw,
        pv_kw_ac=pv_kw_ac,
        import_rate_vec=import_rate_vec,
        export_rate=export_rate,
        bess_kwh=bess_kwh, bess_kw=bess_kw, bess_eff_rt=bess_eff_rt,
        bess_min_soc=bess_min_soc, bess_init_soc=bess_init_soc,
        ev_enabled=use_ev,
        ev_kwh=ev_kwh, ev_kw=ev_kw, ev_eff_rt=ev_eff_rt,
        ev_min_soc=ev_min_soc, ev_init_soc=ev_init_soc,
        ev_avail=avail[:n],
        ev_deg_cost_per_kwh_throughput=ev_capex_per_kwh
    )

    fin = compute_finance(
        summary=summary,
        pv_kw=pv_kw,
        inverter_kw=inverter_kw,
        bess_kwh=bess_kwh,
        pv_capex_per_kw=pv_capex_per_kw,
        inv_capex_per_kw=inv_capex_per_kw,
        bess_capex_per_kwh=bess_capex_per_kwh,
        fixed_om_per_year=fixed_om_per_year,
        project_years=int(project_years),
        discount_rate=float(discount_rate),
        pv_life=int(pv_life),
        inv_life=int(inv_life),
        bess_life=int(bess_life),
        baseline_import_cost_per_year=baseline_import_cost
    )

    st.subheader("3) Key Results (Annual)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("PV Generation (kWh/yr)", f"{summary['pv_gen_kWh']:,.0f}")
    k2.metric("Grid Import (kWh/yr)", f"{summary['import_kWh']:,.0f}")
    k3.metric("Grid Export (kWh/yr)", f"{summary['export_kWh']:,.0f}")
    k4.metric("Net Grid Bill (RM/yr)", f"{(summary['import_cost_RM']-summary['export_revenue_RM']):,.0f}")

    f1, f2, f3, f4 = st.columns(4)
    f1.metric("CAPEX (RM)", f"{fin['CAPEX_RM']:,.0f}")
    f2.metric("NPC (RM)", f"{fin['NPC_RM']:,.0f}")
    f3.metric("LCOE (RM/kWh)", f"{fin['LCOE_RM_per_kWh']:.3f}")
    pb = fin["Simple_Payback_years"]
    f4.metric("Simple Payback (yr)", "∞" if (pb == np.inf or pb > 1e6) else f"{pb:.1f}")

    with st.expander("Show detailed finance breakdown"):
        st.json(fin)

    st.subheader("4) Plots")
    pcol1, pcol2 = st.columns(2)

    # Plot 1: Power balance (first 168 hours)
    window = min(168, len(results))
    x = results["hour"].iloc[:window]
    fig1 = plt.figure()
    plt.plot(x, results["load_kW"].iloc[:window], label="Load")
    plt.plot(x, results["pv_kW"].iloc[:window], label="PV")
    plt.plot(x, results["grid_import_kW"].iloc[:window], label="Grid Import")
    plt.plot(x, results["grid_export_kW"].iloc[:window], label="Grid Export")
    plt.legend()
    plt.xlabel("Hour")
    plt.ylabel("kW")
    plt.title("Power balance (first week)")
    pcol1.pyplot(fig1, use_container_width=True)

    # Plot 2: SOC energies (first week)
    fig2 = plt.figure()
    plt.plot(x, results["bess_energy_kWh"].iloc[:window], label="BESS Energy (kWh)")
    if use_ev:
        plt.plot(x, results["ev_energy_kWh"].iloc[:window], label="EV Energy (kWh)")
    plt.legend()
    plt.xlabel("Hour")
    plt.ylabel("kWh")
    plt.title("Storage energy (first week)")
    pcol2.pyplot(fig2, use_container_width=True)

    st.subheader("5) Download Results")
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button("Download hourly results CSV", data=csv_bytes, file_name="hourly_results.csv", mime="text/csv")

    summary_df = pd.DataFrame([summary | fin])
    st.dataframe(summary_df.T.rename(columns={0: "value"}), use_container_width=True)
    st.download_button("Download summary CSV", data=summary_df.to_csv(index=False).encode("utf-8"),
                       file_name="summary_results.csv", mime="text/csv")

else:
    st.warning("Upload Load CSV, set inputs, then click **Run Simulation** in the sidebar.")

st.markdown("---")
st.caption("Prototype notes: Rule-based dispatch; EV fleet aggregated; simplified PV model from GHI or user PV profile. "
           "Designed for easy extension to MILP dispatch, detailed EV schedules, demand charges, and Malaysia tariff modules.")
