import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime, timedelta

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Agentic AutoCare AI",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# SESSION STATE
# =====================================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "failure_prob" not in st.session_state:
    st.session_state.failure_prob = 0.0

# =====================================================
# GLOBAL DARK UI CSS
# =====================================================
st.markdown("""
<style>
body, .stApp {
    background: radial-gradient(circle at top, #0b1220, #020617);
    color: #e5e7eb;
}
h1,h2,h3 {
    color: #e5e7eb;
}
.top-nav {
    display: flex;
    align-items: center;
    padding: 14px 24px;
    background: rgba(2,6,23,0.95);
    border-radius: 14px;
    margin-bottom: 24px;
}
.nav-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
}
.panel {
    background: rgba(15,23,42,0.75);
    border-radius: 18px;
    padding: 22px;
    border: 1px solid rgba(148,163,184,0.15);
    margin-bottom: 22px;
}
.metric-big {
    font-size: 2.3rem;
    font-weight: 700;
    color: #38bdf8;
}
.good { color: #22c55e; }
.warn { color: #facc15; }
.bad { color: #ef4444; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# LOGIN PAGE (SINGLE LOGIN)
# =====================================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align:center;'>🧠 Agentic AutoCare AI</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;color:#94a3b8;'>Autonomous Vehicle Maintenance Platform</h4>", unsafe_allow_html=True)

    st.divider()
    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Please enter credentials")

    st.stop()

# =====================================================
# TOP NAVIGATION BAR
# =====================================================
left, right = st.columns([2,6])

with left:
    st.markdown("<div class='top-nav'><div class='nav-title'>🚗 AutoCare AI</div></div>", unsafe_allow_html=True)

with right:
    nav_cols = st.columns(5)
    pages = ["Dashboard", "Vehicle", "Scheduling", "Manufacturing", "Security"]

    for col, name in zip(nav_cols, pages):
        if col.button(name):
            st.session_state.page = name

page = st.session_state.page

# =====================================================
# DATA + MODEL
# =====================================================
np.random.seed(42)
fleet = pd.DataFrame({
    "engine_temp": np.random.normal(85, 10, 500),
    "battery_voltage": np.random.normal(12.2, 0.6, 500),
    "brake_wear": np.random.randint(10, 90, 500),
    "oil_pressure": np.random.normal(40, 8, 500),
    "coolant_level": np.random.uniform(60, 100, 500),
    "rpm": np.random.randint(700, 4500, 500)
})

fleet["failure"] = (
    (fleet["engine_temp"] > 95) |
    (fleet["battery_voltage"] < 11.8) |
    (fleet["brake_wear"] > 70) |
    (fleet["oil_pressure"] < 30) |
    (fleet["coolant_level"] < 65)
).astype(int)

X = fleet.drop("failure", axis=1)
y = fleet["failure"]

@st.cache_resource
def load_model():
    model = RandomForestClassifier(n_estimators=120, max_depth=8)
    model.fit(X, y)
    return model

model = load_model()

# =====================================================
# DASHBOARD
# =====================================================
if page == "Dashboard":
    st.subheader("🛰️ Fleet Command Dashboard")
    c1, c2, c3 = st.columns(3)

    c1.markdown("<div class='panel'><div class='metric-big'>500</div>Fleet Size</div>", unsafe_allow_html=True)
    c2.markdown("<div class='panel'><div class='metric-big'>91%</div>Average Health</div>", unsafe_allow_html=True)
    c3.markdown("<div class='panel'><div class='metric-big'>37</div>Predicted Failures</div>", unsafe_allow_html=True)

# =====================================================
# VEHICLE HEALTH INTELLIGENCE
# =====================================================
elif page == "Vehicle":
    st.subheader("🚗 Vehicle Health Intelligence")

    col1, col2 = st.columns([1,2])

    with col1:
        engine_temp = st.slider("Engine Temp (°C)", 60, 120, 92)
        battery_voltage = st.slider("Battery Voltage (V)", 10.0, 13.5, 12.0)
        brake_wear = st.slider("Brake Wear (%)", 0, 100, 55)
        oil_pressure = st.slider("Oil Pressure (psi)", 20, 60, 38)
        coolant_level = st.slider("Coolant Level (%)", 40, 100, 75)
        rpm = st.slider("RPM", 600, 5000, 2500)

    input_df = pd.DataFrame([{
        "engine_temp": engine_temp,
        "battery_voltage": battery_voltage,
        "brake_wear": brake_wear,
        "oil_pressure": oil_pressure,
        "coolant_level": coolant_level,
        "rpm": rpm
    }])

    prob = model.predict_proba(input_df)[0][1]
    st.session_state.failure_prob = prob

    def score(v, lo, hi):
        return 90 if lo <= v <= hi else 50

    scores = {
        "Engine Temp": score(engine_temp, 70, 95),
        "Battery Voltage": score(battery_voltage, 11.8, 13.2),
        "Brake Wear": score(brake_wear, 0, 70),
        "Oil Pressure": score(oil_pressure, 30, 55),
        "Coolant Level": score(coolant_level, 65, 100),
        "RPM": score(rpm, 700, 4000)
    }

    health = int(sum(scores.values()) / len(scores))

    with col2:
        st.dataframe(pd.DataFrame({
            "Parameter": scores.keys(),
            "Health (%)": scores.values()
        }), use_container_width=True)

        st.metric("Overall Vehicle Health", f"{health}%")

        if prob > 0.7:
            st.error("⚠ Proactive Maintenance Required")
        else:
            st.success("✔ Vehicle Operating Normally")

# =====================================================
# AUTONOMOUS SCHEDULING
# =====================================================
elif page == "Scheduling":
    st.subheader("📅 Autonomous Scheduling")

    if st.session_state.failure_prob > 0.7:
        st.success("Service Automatically Scheduled")
        st.write("📍 Hyderabad Service Hub")
        st.write("🕙 In 3 days")
    else:
        st.info("No service required")

# =====================================================
# MANUFACTURING FEEDBACK
# =====================================================
elif page == "Manufacturing":
    st.subheader("🏭 Manufacturing Feedback")
    st.bar_chart({
        "Brakes": 42,
        "Battery": 28,
        "Cooling": 19,
        "Engine": 11
    })

# =====================================================
# SECURITY
# =====================================================
elif page == "Security":
    st.subheader("🔐 Behavioral Security (UEBA)")
    st.warning("Scheduling Agent override anomaly detected")
