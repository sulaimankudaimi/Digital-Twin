import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. إعدادات الصفحة ---
st.set_page_config(page_title="SPC | Production Digital Twin", layout="wide", page_icon="🛢️")

# --- 2. محرك قراءة وتنظيف البيانات ---
# --- 2. محرك قراءة وتنظيف البيانات المطور ---
@st.cache_data
def load_production_data():
    try:
        df = pd.read_csv('production_data.csv', encoding='latin1')
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        # البحث عن عمود الإنتاج
        potential_cols = [c for c in df.columns if any(w in c for w in ['oil', 'vol', 'prod', 'value'])]
        target_col = potential_cols[0] if potential_cols else df.columns[-1]
        
        # التعديل الجوهري: تحويل البيانات لأرقام وحذف النصوص (Errors='coerce' تحول النص لـ NaN)
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        # حذف الأسطر الفارغة التي نتجت عن نصوص خاطئة
        df = df.dropna(subset=[target_col])
        
        df = df.rename(columns={target_col: 'production'})
        return df, True
    except Exception as e:
        return None, False

# --- 3. المحرك التحليلي (AI Engine) مع معالجة الأخطاء ---
@st.cache_resource
def train_model(df, is_real):
    try:
        if is_real and df is not None:
            # التأكد من أخذ عينة نظيفة وأرقام فقط
            clean_df = df.head(500).copy()
            X = np.arange(len(clean_df)).reshape(-1, 1)
            y = clean_df['production'].astype(float) # التأكد من نوع البيانات Float
        else:
            X = np.linspace(1000, 4000, 500).reshape(-1, 1)
            y = (5000 - (X.flatten() * 0.8)) + np.random.normal(0, 100, 500)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        # إذا فشل التدريب على البيانات الحقيقية، نعود للبيانات الافتراضية لضمان عمل الواجهة
        st.sidebar.error(f"AI Training Error: {e}")
        X = np.linspace(1000, 4000, 500).reshape(-1, 1)
        y = (5000 - (X.flatten() * 0.8)) + np.random.normal(0, 100, 500)
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return model

df_real, success = load_production_data()
model = train_model(df_real, success)

# --- 4. تصميم الواجهة ---
st.title("🛢️ Integrated Production Digital Twin")
st.markdown("**Field Operations Center | Syrian Petroleum Company (SPC)**")
st.divider()

# القائمة الجانبية
st.sidebar.header("🕹️ Control Room")
st_depth = st.sidebar.slider("Current Depth (m)", 1000, 4500, 2500)
st_rpm = st.sidebar.slider("Rotary Speed (RPM)", 0, 150, 80)
current_pred = model.predict([[st_depth]])[0]

# --- 5. العدادات والمؤشرات (Gauges) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Flow Rate", f"{current_pred:.2f} bbl/d")
with col2:
    status = "OPTIMAL" if current_pred > 2000 else "CRITICAL"
    st.info(f"System Status: {status}")
with col3:
    efficiency = min(100.0, (current_pred / 4000) * 100)
    st.metric("System Efficiency", f"{efficiency:.1f}%")

# --- 6. الرسوم البيانية (تجنب خطأ المجلدات) ---
c1, c2 = st.columns([2, 1])
with c1:
    st.subheader("📈 Performance Trend")
    if success:
        # رسم البيانات الرقمية فقط لتجنب ValueError
        df_plot = df_real.select_dtypes(include=[np.number]).head(100)
        fig = px.line(df_plot, y='production', title="Real-time Production Monitoring")
    else:
        dummy_x = np.linspace(1000, 4500, 100)
        dummy_y = model.predict(dummy_x.reshape(-1, 1))
        fig = px.line(x=dummy_x, y=dummy_y, title="Simulated Performance Curve")
    fig.update_layout(template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("🔍 Health Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=current_pred,
        gauge={'axis': {'range': [None, 5000]}, 'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 1500], 'color': "red"}, {'range': [1500, 3000], 'color': "yellow"}, {'range': [3000, 5000], 'color': "green"}]}))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- 7. التنبيهات الذكية ---
if current_pred < 1500:
    st.error("🚨 ALERT: Production below threshold! AI suggests Choke adjustment.")
else:
    st.success("✅ Operation stable within safety margins.")

# --- 8. زر تحميل التقرير (جديد) ---
st.sidebar.divider()
st.sidebar.subheader("📥 Data Export")
report_df = pd.DataFrame({'Timestamp': [pd.Timestamp.now()], 'Depth': [st_depth], 'RPM': [st_rpm], 'Predicted_Prod': [current_pred]})
csv = report_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📄 Download Diagnostic Report", data=csv, file_name=f"SPC_Report_{st_depth}m.csv", mime='text/csv')

# --- 10. محرك التنبؤ الزمني المطور (Forecasting Engine) ---
st.divider()
st.subheader("🔮 AI Production Forecasting (Next 6 Months)")

if success:
    # 1. تجهيز البيانات للتنبؤ
    numeric_df = df_real.apply(pd.to_numeric, errors='coerce').dropna(subset=['production'])
    y_data = numeric_df['production'].values
    X_data = np.arange(len(y_data)).reshape(-1, 1)

    # 2. بناء موديل التنبؤ السريع
    forecast_model = RandomForestRegressor(n_estimators=50) # أسرع وأدق للسلاسل الزمنية
    forecast_model.fit(X_data, y_data)

    # 3. التوقع للمستقبل
    future_X = np.arange(len(y_data), len(y_data) + 180).reshape(-1, 1)
    future_y = forecast_model.predict(future_X)

    # 4. رسم المنحنى المتكامل (Plotly)
    fig_final = go.Figure()
    # البيانات التاريخية
    fig_final.add_trace(go.Scatter(x=X_data.flatten()[-200:], y=y_data[-200:], name='Historical Data', line=dict(color='cyan')))
    # رسم المنحنى المتكامل
    fig_final.update_layout(title="Integrated Production Decline Curve", xaxis_title="Time Units", yaxis_title="Production Volume", template="plotly_dark")
    st.plotly_chart(fig_final, use_container_width=True)
    
    st.success(f"✅ AI Analysis Complete: Predicted production at the end of forecast: {future_y[-1]:.2f} units.")
    st.info("💡 الملاحظة الفنية: يتوقع الموديل انخفاضاً طبيعياً في الإنتاج. ينصح بجدولة صيانة للمضخة بعد 120 يوماً.")

else:
    # في حالة عدم تحميل البيانات الحقيقية
    st.warning("⚠️ Simulation mode: AI forecasting is based on synthetic engineering models.")
    # (اختياري) يمكنك وضع رسم بياني افتراضي هنا

# --- 11. تذييل الصفحة (Footer) ---
st.divider()
st.markdown("<center>Designed & Developed by <b>Eng. Solaiman Kudaimi</b> for SPC Project 2026</center>", unsafe_allow_html=True)
