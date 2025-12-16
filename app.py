import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm

# --- 1. SET PAGE CONFIG & CUSTOM CSS ---
st.set_page_config(page_title="AI Strategic Decision Simulator", layout="wide", page_icon="🎯")

# Custom CSS for Bold Inputs and Professional UI
st.markdown("""
    <style>
    .stNumberInput label, .stSlider label, .stSelectbox label, .stRadio label {
        font-weight: bold !important;
        color: #1E1E1E !important;
        font-size: 1.1rem !important;
    }
    div[data-testid="stMetric"] {
        background-color: #F8F9FA;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True) # FIXED: Changed from unsafe_base_with_rows

# --- 2. DATA & AI ENGINE ---
@st.cache_data
def get_sim_data():
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=730, freq="D")
    prices = 100 + np.random.normal(0, 2, len(dates)).cumsum()
    prices = np.clip(prices, 50, 250)
    # Simulate historical relationship
    log_q = 10 - 1.6 * np.log(prices) + np.random.normal(0, 0.1, len(dates))
    quantity = np.exp(log_q).astype(int)
    return pd.DataFrame({"Price": prices, "Quantity": quantity})

def get_elasticity(df):
    log_q, log_p = np.log(df['Quantity']), np.log(df['Price'])
    model = sm.OLS(log_q, sm.add_constant(log_p)).fit()
    return model.params.iloc[1]

# --- 3. INPUT UI ---
st.title("🚀 AI Strategic Decision Simulator")
st.markdown("### 1. Define Business Constraints")

col_in1, col_in2, col_in3 = st.columns(3)
with col_in1:
    action = st.radio("Strategic Direction", ["Increase Price 📈", "Decrease Price 📉"])
    change_pct_val = st.slider("Magnitude of Change (%)", 0.0, 50.0, 10.0) / 100
with col_in2:
    unit_cogs = st.number_input("Direct Unit Cost (COGS) $", value=45.0)
    fixed_costs = st.number_input("Monthly Overhead (Fixed) $", value=15000.0)
with col_in3:
    horizon = st.number_input("Simulation Horizon (Months)", 1, 12, 6)

st.markdown("---")
run_analysis = st.button("📊 GENERATE FULL STRATEGIC ANALYSIS", use_container_width=True, type="primary")

if run_analysis:
    # --- 4. CALCULATIONS ---
    df = get_sim_data()
    elasticity = get_elasticity(df)
    
    # Adjust change based on direction
    change_pct = -change_pct_val if "Decrease" in action else change_pct_val
    
    curr_p = df['Price'].iloc[-30:].mean()
    curr_q = df['Quantity'].iloc[-30:].mean()
    
    # Scenario Stats
    new_p = curr_p * (1 + change_pct)
    new_q = curr_q * (1 + (elasticity * change_pct))
    
    # Financials (Monthly)
    base_profit = ((curr_p - unit_cogs) * curr_q * 30) - fixed_costs
    new_profit = ((new_p - unit_cogs) * new_q * 30) - fixed_costs
    profit_delta = new_profit - base_profit
    
    # --- 5. EXECUTIVE SHOCK OUTPUTS ---
    st.markdown("## 🚨 Executive Shock Analysis")
    s1, s2, s3 = st.columns(3)
    
    with s1:
        color = "normal" if profit_delta > 0 else "inverse"
        label = "Daily Profit Leakage" if profit_delta < 0 else "Daily Opportunity Gain"
        st.metric(label, f"${abs(profit_delta/30):,.2f}", f"${abs(profit_delta):,.0f} / month", delta_color=color)
    
    with s2:
        # Effort required to keep same profit after a price drop
        multiplier = (base_profit + fixed_costs) / ((new_p - unit_cogs) * curr_q * 30) if change_pct < 0 else 1.0
        effort_text = f"{abs(multiplier-1)*100:.1f}% More" if change_pct < 0 else "Efficiency Gained"
        st.metric("Sales Effort Required", effort_text, "Operational Strain")

    with s3:
        # Distance to zero profit
        safety_buffer = new_profit / (new_p - unit_cogs) / 30 if (new_p - unit_cogs) > 0 else 0
        st.metric("Margin of Safety (Daily)", f"{int(max(0, safety_buffer))} Units", "Distance to Cliff")

    # --- 6. VISUALS & SOLUTIONS ---
    st.divider()
    tab1, tab2 = st.tabs(["📈 Financial Projection", "💡 Strategic Solutions"])
    
    with tab1:
        # Clean Profit Chart
        fig = go.Figure()
        x_axis = [f"Month {i+1}" for i in range(int(horizon))]
        fig.add_trace(go.Scatter(x=x_axis, y=[base_profit]*int(horizon), name="Status Quo", line=dict(dash='dash', color='gray')))
        fig.add_trace(go.Scatter(x=x_axis, y=[new_profit]*int(horizon), name="AI Prediction", fill='tonexty', line=dict(width=4, color='blue')))
        fig.update_layout(
            title="Monthly Net Profit Shift", 
            yaxis_title="Net Profit ($)",
            xaxis_title="Timeline",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("### Final AI Verdict")
        if new_profit > base_profit:
            st.success(f"### ✅ RECOMMENDATION: PROCEED\nEvery 1% price change is only losing {abs(elasticity):.1f}% volume. Your margin power is stronger than customer resistance.")
        else:
            st.error(f"### ❌ RECOMMENDATION: ABORT\nCustomers are too sensitive to this price (Elasticity: {elasticity:.2f}). The volume drop will kill your net profit.")

        st.info(f"**Strategic Solution:** To survive this move, you must reduce fixed costs by **${abs(profit_delta):,.2f}** per month just to reach your old baseline.")

else:
    st.info("👈 Set your price strategy and costs, then click the **GENERATE FULL STRATEGIC ANALYSIS** button above.")
