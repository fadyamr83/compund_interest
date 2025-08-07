
import math
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# -------------------------------
# Core math
# -------------------------------

@dataclass
class CIParams:
    principal: float
    annual_rate: float                # decimal, e.g., 0.08 for 8%
    years: float
    compounds_per_year: int           # n
    contribution: float               # C
    contribution_freq: Literal["monthly","quarterly","yearly"]
    contribution_timing: Literal["end","begin"]

def frequency_to_periods_per_year(freq: str) -> int:
    if freq == "monthly":
        return 12
    if freq == "quarterly":
        return 4
    if freq == "yearly":
        return 1
    raise ValueError("Unknown contribution frequency.")

def lcm(a: int, b: int) -> int:
    import math
    return abs(a*b) // math.gcd(a, b)

def compound_interest_growth(params: CIParams) -> Tuple[pd.DataFrame, float, float]:
    P = float(params.principal)
    r = float(params.annual_rate)
    n = int(params.compounds_per_year)
    t_years = float(params.years)
    contrib = float(params.contribution)
    contrib_freq = frequency_to_periods_per_year(params.contribution_freq)
    timing = params.contribution_timing

    base_per_year = lcm(n, contrib_freq)
    total_periods = int(round(t_years * base_per_year))
    effective_rate_per_base = r / base_per_year  # nominal spread across base periods

    balance = P
    rows = []
    contributions_cum = 0.0
    interest_cum = 0.0

    for k in range(1, total_periods + 1):
        deposit = 0.0
        if (k % (base_per_year // contrib_freq) == 0):
            deposit = contrib

        if timing == "begin" and deposit > 0:
            balance += deposit
            contributions_cum += deposit

        interest = balance * effective_rate_per_base
        balance += interest
        interest_cum += interest

        if timing == "end" and deposit > 0:
            balance += deposit
            contributions_cum += deposit

        time_years = k / base_per_year
        rows.append({
            "period_index": k,
            "time_years": time_years,
            "deposit": deposit,
            "interest_earned": interest,
            "balance": balance,
            "contributions_cum": contributions_cum,
            "interest_cum": interest_cum,
        })

    df = pd.DataFrame(rows)
    total_contributions = contributions_cum
    total_interest = interest_cum
    return df, total_contributions, total_interest

# Closed-form helpers (approximate when compounding frequency != contribution frequency)
def future_value_principal_only(P: float, r: float, n: int, t: float) -> float:
    return P * (1 + r / n) ** (n * t)

def future_value_annuity(C: float, r: float, m: int, t: float, timing: str = "end") -> float:
    i = r / m
    if i == 0:
        fv = C * m * t
    else:
        fv = C * ((1 + i) ** (m * t) - 1) / i
        if timing == "begin":
            fv *= (1 + i)
    return fv

# -------------------------------
# Plots (matplotlib, single chart per figure, no explicit colors)
# -------------------------------

def plot_balance(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(df["time_years"], df["balance"])
    ax.set_xlabel("Years")
    ax.set_ylabel("Balance")
    ax.set_title("Total Balance Over Time")
    ax.grid(True)
    st.pyplot(fig)

def plot_components(df: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(df["time_years"], df["contributions_cum"], label="Contributions (cum)")
    ax.plot(df["time_years"], df["interest_cum"], label="Interest (cum)")
    ax.set_xlabel("Years")
    ax.set_ylabel("Amount")
    ax.set_title("Contributions vs. Interest (Cumulative)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# -------------------------------
# UI
# -------------------------------

st.set_page_config(page_title="Compound Interest — Interactive Lesson", layout="wide")

st.title("Compound Interest — Interactive Lesson")

st.markdown(
    """
**Core ideas**  
- Compound interest grows your money because interest earns interest.  
- Nominal annual rate: `r`, compounding `n` times per year for `t` years.  
- Future Value (principal only):  
  \\[ \\text{FV} = P (1 + \\tfrac{r}{n})^{n t} \\]  
- With regular contributions (annuity), `m` payments per year:  
  \\[ \\text{FV}_{\\text{annuity,end}} = C \\cdot \\frac{(1 + \\tfrac{r}{m})^{m t} - 1}{\\tfrac{r}{m}} \\]  
  For payments at the **beginning** of each period (annuity due), multiply by \\((1 + \\tfrac{r}{m})\\).
"""
)

with st.sidebar:
    st.header("Inputs")
    P = st.number_input("Principal (P)", min_value=0.0, value=1000.0, step=100.0)
    r = st.number_input("Annual rate r (decimal, e.g., 0.08)", min_value=0.0, max_value=1.0, value=0.08, step=0.005, format="%.3f")
    t = st.number_input("Years (t)", min_value=0.5, max_value=100.0, value=10.0, step=0.5)
    n = st.selectbox("Compounds per year (n)", options=[1,2,4,12,365], index=3)
    C = st.number_input("Contribution amount (C)", min_value=0.0, value=100.0, step=50.0)
    c_freq_label = st.selectbox("Contribution frequency", options=["monthly","quarterly","yearly"], index=0)
    timing = st.radio("Contribution timing", options=["end","begin"], index=0, horizontal=True)

params = CIParams(
    principal=P,
    annual_rate=r,
    years=t,
    compounds_per_year=int(n),
    contribution=C,
    contribution_freq=c_freq_label,
    contribution_timing=timing,
)

df, tot_c, tot_i = compound_interest_growth(params)
final_balance = float(df["balance"].iloc[-1])

# Totals + closed-form check
left, right = st.columns([2,1])
with left:
    st.subheader("Totals")
    st.metric("Final Balance", f"{final_balance:,.2f}")
    st.write(
        f"**Contributions (cum):** {tot_c:,.2f} &nbsp;&nbsp; | "
        f"**Interest (cum):** {tot_i:,.2f}"
    )
with right:
    m = frequency_to_periods_per_year(params.contribution_freq)
    fv_closed = future_value_principal_only(params.principal, params.annual_rate, params.compounds_per_year, params.years) + \
                future_value_annuity(params.contribution, params.annual_rate, m, params.years, params.contribution_timing)
    st.caption(f"Closed‑form approximation (principal + annuity): **{fv_closed:,.2f}**")

st.divider()

c1, c2 = st.columns(2)
with c1:
    plot_balance(df)
with c2:
    plot_components(df)

st.divider()

st.subheader("Amortization-like Schedule (sampled)")
step = max(1, len(df) // 300)
df_view = df.iloc[::step, :].copy()
df_view = df_view[["time_years","deposit","interest_earned","contributions_cum","interest_cum","balance"]].round(2)
st.dataframe(df_view, use_container_width=True)

csv = df.round(6).to_csv(index=False).encode("utf-8")
st.download_button("Download full schedule as CSV", csv, file_name="compound_interest_schedule.csv", mime="text/csv")

st.info(
    "Teaching tip: Ask students to switch timing from **End** to **Begin**, or change compounding from monthly to daily, "
    "and predict which line (interest or contributions) will grow faster before updating the charts."
)
