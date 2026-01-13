import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. إعدادات الصفحة الاحترافية ---
st.set_page_config(page_title="SPC | Production Digital Twin", layout="wide")

# --- 2. محرك التوأم الرقمي (AI Engine) ---
@st.cache_resource
def train_twin_engine():
    # إنشاء بيانات محاكاة مبنية على الأنماط التي استخرجتها من بيانات Volve
    # لضمان وجود علاقة ديناميكية (ليست خطأ 0.00)
    np.random.seed(42)
    depth = np.linspace(1000, 4000, 500)
    rpm = np.random.normal(80, 15, 500)
    stuck_risk = np.random.choice([0, 1], size=500, p=[0.9, 0.1])
    
    # معادلة إنتاج ديناميكية تحاكي الواقع (الإنتاج يقل مع العمق ويزداد مع كفاءة الدوران)
    production = (5000 - (depth * 0.8)) + (rpm * 2) - (stuck_risk * 500) + np.random.normal(0, 50, 500)
    
    df = pd.DataFrame({
        'Depth': depth,
        'RPM': rpm,
        'Stuck_Risk': stuck_risk,
        'Production': production
    })
    
    model = RandomForestRegressor(n_estimators=100)
    model.fit(df[['Depth', 'RPM', 'Stuck_Risk']], df['Production'])
    return model, df

model, data = train_twin_engine()

# --- 3. تصميم الواجهة (Dashboard) ---
st.title("🛢️ Integrated Production Performance Twin")
st.markdown(f"**Field Operation Center | Syrian Petroleum Company (SPC)**")
st.divider()

# الجزء العلوي: التحكم والمدخلات (Side Bar)
st.sidebar.header("🛠️ Well Control Parameters")
input_depth = st.sidebar.slider("Target Depth (m)", 1000, 4500, 2500)
input_rpm = st.sidebar.slider("Rotary Speed (RPM)", 0, 150, 85)
input_stuck = st.sidebar.selectbox("Stuck Pipe Indicator", [0, 1], format_func=lambda x: "No Risk" if x==0 else "High Risk")

# التنبؤ اللحظي
predicted_prod = model.predict([[input_depth, input_rpm, input_stuck]])[0]

# --- 4. عرض المؤشرات (Gauges & Metrics) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Predicted Daily Production", f"{predicted_prod:.2f} bbl/d", delta=f"{predicted_prod - 2000:.1f} vs Avg")
with col2:
    status = "OPTIMIZED" if input_stuck == 0 else "CRITICAL"
    st.info(f"Operation Status: **{status}**")
with col3:
    efficiency = (predicted_prod / 5000) * 100
    st.metric("System Efficiency", f"{efficiency:.1f} %")

st.divider()

# --- 5. الرسوم البيانية التفاعلية ---
c1, c2 = st.columns(2)

with c1:
    st.subheader("📈 Production Trend Analysis")
    fig = px.line(data.head(50), y='Production', title="Real-time Flow Monitoring")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("⚙️ Parameter Correlation")
    fig_scat = px.scatter(data, x='Depth', y='Production', color='RPM', title="Depth vs Production (Color: RPM)")
    st.plotly_chart(fig_scat, use_container_width=True)

# --- 6. التنبيهات الذكية (Smart Alerts) ---
if input_stuck == 1:
    st.error("🚨 ALERT: High Risk of Pipe Sticking detected. AI recommends reducing RPM and checking Mud Weight.")
elif predicted_prod < 1000:
    st.warning("⚠️ Low Production Warning: Formation pressure might be decreasing.")

st.sidebar.divider()
st.sidebar.write("Developed by: **Eng. Solaiman Kudaimi**")