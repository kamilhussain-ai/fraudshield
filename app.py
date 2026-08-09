import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0a0e1a; }

[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #a8b8d8 !important; }
[data-testid="stSidebar"] h2 { color: #ffffff !important; font-size: 16px !important; }

.header {
    background: linear-gradient(135deg, #0f1629 0%, #1a237e 50%, #0d47a1 100%);
    border: 1px solid #1e3a6e;
    border-radius: 20px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(0,150,255,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.header::after {
    content: '';
    position: absolute;
    bottom: -60%;
    left: 20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(100,0,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.header h1 {
    color: #ffffff;
    font-size: 32px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.header p {
    color: #7eb3ff;
    font-size: 13px;
    margin: 8px 0 0 0;
    font-weight: 400;
}
.badge {
    display: inline-block;
    background: rgba(0,150,255,0.2);
    border: 1px solid rgba(0,150,255,0.4);
    color: #7eb3ff;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 8px;
    margin-top: 12px;
}

.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    transition: all 0.3s ease;
}
.glass-card:hover { border-color: rgba(0,150,255,0.3); }
.glass-card .label {
    color: #5a7a9e;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin: 0;
}
.glass-card .value {
    color: #ffffff;
    font-size: 26px;
    font-weight: 700;
    margin: 8px 0 0 0;
}
.glass-card .accent { color: #3b9eff; }

.fraud-card {
    background: linear-gradient(135deg, rgba(220,38,38,0.15), rgba(153,27,27,0.1));
    border: 1px solid rgba(220,38,38,0.4);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
}
.safe-card {
    background: linear-gradient(135deg, rgba(22,163,74,0.15), rgba(15,118,53,0.1));
    border: 1px solid rgba(22,163,74,0.4);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
}
.result-icon { font-size: 56px; margin-bottom: 12px; }
.result-title { font-size: 24px; font-weight: 800; margin: 0; }
.result-subtitle { font-size: 13px; margin: 8px 0 20px 0; }
.prob-badge {
    display: inline-block;
    padding: 12px 28px;
    border-radius: 12px;
    font-size: 32px;
    font-weight: 800;
}

.section-label {
    color: #5a7a9e;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 28px 0 14px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

.idle-card {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 80px;
    text-align: center;
    margin-top: 8px;
}

.stButton > button {
    background: linear-gradient(135deg, #1565c0, #1976d2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    width: 100% !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 20px rgba(21,101,192,0.4) !important;
}
.stButton > button:hover {
    box-shadow: 0 6px 28px rgba(21,101,192,0.6) !important;
}

div[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('xgb_smote_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# Header
st.markdown("""
<div class="header">
    <h1>🛡️ FraudShield</h1>
    <p>Real-time Credit Card Fraud Detection System</p>
    <div style="margin-top:14px;">
        <span class="badge">XGBoost + SMOTE</span>
        <span class="badge">AUC 0.9779</span>
        <span class="badge">SMIU FYP 2025</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Metric Cards
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("Algorithm", "XGBoost", "accent"),
    ("ROC-AUC Score", "0.9779", "accent"),
    ("Fraud Recall", "87%", "accent"),
    ("Training Samples", "284,807", "accent"),
]
for col, (label, value, cls) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="glass-card">
        <p class="label">{label}</p>
        <p class="value {cls}">{value}</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## 🛡️ FraudShield")
st.sidebar.markdown("---")
st.sidebar.markdown("**Transaction Details**")
amount = st.sidebar.number_input("Amount ($)", min_value=0.0, max_value=100000.0, value=100.0)
time_val = st.sidebar.number_input("Time (seconds)", min_value=0.0, value=50000.0)
st.sidebar.markdown("---")
st.sidebar.markdown("**PCA Features**")
v_values = {}
for i in range(1, 29):
    v_values[f'V{i}'] = st.sidebar.number_input(f"V{i}", value=0.0, format="%.4f")

predict_btn = st.sidebar.button("⚡ Analyze Transaction")

if predict_btn:
    amount_scaled = (amount - 88.35) / 250.12
    time_scaled = (time_val - 94813.86) / 47488.15

    input_data = {f'V{i}': [v_values[f'V{i}']] for i in range(1, 29)}
    input_data['Amount_Scaled'] = [amount_scaled]
    input_data['Time_Scaled'] = [time_scaled]
    input_df = pd.DataFrame(input_data)

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])

    with left:
        if prediction == 1:
            st.markdown(f"""
            <div class="fraud-card">
                <div class="result-icon">🚨</div>
                <p class="result-title" style="color:#f87171;">FRAUD DETECTED</p>
                <p class="result-subtitle" style="color:#9ca3af;">This transaction has been flagged as high risk</p>
                <div class="prob-badge" style="background:rgba(220,38,38,0.2); color:#f87171; border:1px solid rgba(220,38,38,0.4);">
                    {probability*100:.1f}%
                </div>
                <p style="color:#6b7280; font-size:12px; margin-top:8px;">Fraud Probability</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-card">
                <div class="result-icon">✅</div>
                <p class="result-title" style="color:#4ade80;">TRANSACTION SAFE</p>
                <p class="result-subtitle" style="color:#9ca3af;">No fraudulent activity detected</p>
                <div class="prob-badge" style="background:rgba(22,163,74,0.2); color:#4ade80; border:1px solid rgba(22,163,74,0.4);">
                    {probability*100:.1f}%
                </div>
                <p style="color:#6b7280; font-size:12px; margin-top:8px;">Fraud Probability</p>
            </div>
            """, unsafe_allow_html=True)

    with right:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 40, 'color': '#ffffff'}},
            title={'text': "Risk Score", 'font': {'size': 13, 'color': '#5a7a9e'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#5a7a9e', 'tickfont': {'color': '#5a7a9e'}},
                'bar': {'color': "#ef4444" if prediction == 1 else "#22c55e", 'thickness': 0.25},
                'bgcolor': "rgba(0,0,0,0)",
                'bordercolor': "rgba(0,0,0,0)",
                'steps': [
                    {'range': [0, 30], 'color': "rgba(34,197,94,0.15)"},
                    {'range': [30, 70], 'color': "rgba(234,179,8,0.15)"},
                    {'range': [70, 100], 'color': "rgba(239,68,68,0.15)"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="white",
            height=280,
            margin=dict(t=40, b=0, l=30, r=30)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature Chart
    st.markdown('<div class="section-label">Key Feature Analysis</div>', unsafe_allow_html=True)
    top_features = {f'V{i}': v_values[f'V{i}'] for i in [14, 4, 8, 12, 17, 11, 1, 3]}
    colors = ['#ef4444' if v < 0 else '#3b9eff' for v in top_features.values()]

    fig2 = go.Figure(go.Bar(
        x=list(top_features.keys()),
        y=list(top_features.values()),
        marker_color=colors,
        marker_line_width=0,
        text=[f"{v:.2f}" for v in top_features.values()],
        textposition='outside',
        textfont=dict(color='#a8b8d8', size=11)
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color='#a8b8d8'),
        height=260,
        margin=dict(t=20, b=20, l=10, r=10),
        xaxis=dict(showgrid=False, tickfont=dict(color='#5a7a9e')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)', tickfont=dict(color='#5a7a9e')),
        bargap=0.3
    )
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.markdown("""
    <div class="idle-card">
        <div style="font-size:56px; margin-bottom:16px;">🛡️</div>
        <p style="color:#ffffff; font-size:20px; font-weight:700; margin:0;">Ready to Analyze</p>
        <p style="color:#5a7a9e; font-size:14px; margin:10px 0 0 0;">Enter transaction details in the sidebar and click <b style="color:#3b9eff;">Analyze Transaction</b></p>
    </div>
    """, unsafe_allow_html=True)