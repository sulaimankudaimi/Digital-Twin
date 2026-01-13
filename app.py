import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor

# --- 1. إعدادات الصفحة الاحترافية ---
st.set_page_config(page_title="SPC | Production Digital Twin", layout="wide", page_icon="🛢️")

# --- 2. وظيفة قراءة البيانات (Data Ingestion) ---
@st.cache_data
def load_production_data():
    try:
        # 1. محاولة قراءة الملف مع تجاوز أخطاء الترميز
        df = pd.read_csv('production_data.csv', encoding='latin1')
        
        # 2. تنظيف أسماء الأعمدة (إزالة المسافات وتحويلها لنص صغير)
        df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
        
        # 3. محاولة العثور على عمود الإنتاج تلقائياً
        # سنبحث عن كلمات دلالية مثل (oil, volume, value, production)
        potential_target_cols = [c for c in df.columns if any(word in c for word in ['oil', 'vol', 'prod', 'value'])]
        
        if potential_target_cols:
            # إعادة تسمية العمود المكتشف إلى 'production' لسهولة استخدامه في الكود
            df = df.rename(columns={potential_target_cols[0]: 'production'})
            return df, True
        else:
            # إذا لم يجد عموداً مناسباً، سنعتبر آخر عمود هو الإنتاج
            df = df.rename(columns={df.columns[-1]: 'production'})
            return df, True
            
    except Exception as e:
        st.sidebar.error(f"Error details: {e}")
        return None, False

# --- 3. بناء وتدريب محرك التوأم الرقمي (AI Engine) ---
@st.cache_resource
def train_twin_engine(df, is_real):
    if is_real:
        # استخدام أعمدة حقيقية من بيانات Volve
        # سنحاول العثور على أعمدة الضغط أو العمق، وإذا لم توجد سنعتمد على الترتيب الزمني
        X = np.arange(len(df)).reshape(-1, 1) # كبديل للزمن
        y = df.iloc[:, -1] # نفترض أن العمود الأخير هو الإنتاج
    else:
        # بيانات محاكاة هندسية دقيقة في حال غياب الملف
        X = np.linspace(1000, 4000, 500).reshape(-1, 1) # العمق
        y = (5000 - (X.flatten() * 0.8)) + np.random.normal(0, 100, 500)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

# --- 4. تشغيل العمليات الخلفية ---
df_real, success = load_production_data()
model = train_twin_engine(df_real, success)
# --- 9. تصدير التقارير (Exporting Reports) ---
st.sidebar.divider()
st.sidebar.subheader("📥 Export Results")

# تحضير بيانات التقرير الحالي بناءً على مدخلات المستخدم
report_data = pd.DataFrame({
    'Parameter': ['Target Depth', 'Operating Speed', 'Stuck Risk Status', 'Predicted Production'],
    'Value': [f"{st_depth} m", f"{st_rpm} RPM", status, f"{current_pred:.2f} bbl/d"]
})

# دالة لتحويل الـ DataFrame إلى ملف CSV للتحميل
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# زر تحميل تقرير الحالة الحالية
csv_report = convert_df(report_data)
st.sidebar.download_button(
    label="📄 Download Diagnostic Report",
    data=csv_report,
    file_name=f'Well_Diagnostic_Report_{st_depth}m.csv',
    mime='text/csv',
)

# زر تحميل البيانات التاريخية المعالجة (إذا نجحت القراءة)
if success:
    csv_full = convert_df(df_real.head(100))
    st.sidebar.download_button(
        label="📊 Download Cleaned Field Data",
        data=csv_full,
        file_name='Cleaned_Volve_Data.csv',
        mime='text/csv',
    )
# --- 5. تصميم واجهة المستخدم (The Dashboard) ---
st.title("🛢️ Production Performance Digital Twin")
st.markdown(f"**Field Monitoring & Optimization Center | Syrian Petroleum Company (SPC)**")
st.divider()

# القائمة الجانبية للتحكم
st.sidebar.header("🕹️ Simulation Controls")
if success:
    st.sidebar.success("✅ Real Field Data Loaded")
else:
    st.sidebar.warning("⚠️ Using Engineering Simulation Mode")

st_depth = st.sidebar.slider("Target Depth (m)", 1000, 4500, 2500)
st_rpm = st.sidebar.slider("Operating Speed (RPM)", 0, 150, 80)

# التنبؤ باستخدام الموديل
current_pred = model.predict([[st_depth]])[0]

# --- 6. عرض المؤشرات الرئيسية (KPIs) ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Production", f"{current_pred:.2f} bbl/d", delta="Real-time Prediction")
with col2:
    st.metric("Operational Stability", "94%", delta="Optimal Range")
with col3:
    st.metric("Risk Factor", "Low", delta_color="inverse")

st.divider()

# --- 7. الرسوم البيانية (Visual Analytics) ---
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("📊 Production Trends & AI Forecasting")
    if success:
        # --- التعديل هنا لضمان رسم الأعمدة الرقمية فقط ---
        # نختار أول 100 صف ونأخذ الأعمدة الرقمية فقط (مثل الإنتاج، الضغط)
        df_numeric = df_real.select_dtypes(include=[np.number]).iloc[:100]
        
        if not df_numeric.empty:
            # نرسم عمود 'production' الذي أنشأناه في دالة التحميل
            if 'production' in df_numeric.columns:
                fig = px.line(df_numeric, y='production', title="Real-time Flow Monitoring (bbl/d)")
            else:
                # إذا لم يجد عمود بهذا الاسم، يرسم أول عمود رقمي يجده
                fig = px.line(df_numeric, y=df_numeric.columns[0], title="Field Metric Monitoring")
        else:
            st.error("No numeric data found to plot.")
            fig = go.Figure() # شكل فارغ لمنع الانهيار
    else:
        # رسم بيانات المحاكاة (هذا الجزء سليم عادة)
        dummy_x = np.linspace(1000, 4500, 100)
        dummy_y = model.predict(dummy_x.reshape(-1, 1))
        fig = px.line(x=dummy_x, y=dummy_y, title="Production vs Depth Model")
    
    fig.update_layout(template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
with c2:
    st.subheader("🔍 Diagnostics")
    # إضافة رادار أو عداد سرعة (Gauge)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = current_pred,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Borehole Efficiency"},
        gauge = {'axis': {'range': [None, 5000]},
                 'steps' : [
                     {'range': [0, 2000], 'color': "red"},
                     {'range': [2000, 3500], 'color': "yellow"},
                     {'range': [3500, 5000], 'color': "green"}]}
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

# --- 8. التنبيهات الذكية ---
st.subheader("🔔 Intelligent Alerts")
if current_pred < 1500:
    st.error("🚨 Critical Production Drop: AI suggests immediate well stimulation or choke adjustment.")
else:
    st.success("✅ Operations are within the safe and profitable zone.")

st.divider()
st.markdown("<center>Designed & Developed by <b>Eng. Solaiman Kudaimi</b> for SPC Project 2026</center>", unsafe_allow_html=True)
