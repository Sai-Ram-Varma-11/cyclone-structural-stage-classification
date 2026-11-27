from pathlib import Path
from datetime import date, datetime
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "indian_cyclone_cleaned_engineered.csv"
MODEL_FILE = BASE / "models" / "indian_cyclone_model.joblib"

st.set_page_config(page_title="Indian Cyclone Intelligence", page_icon="🌀", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: #0B1220; color: #F8FAFC; }
[data-testid="stSidebar"] { background: #111C30; border-right: 1px solid #263B56; }
[data-testid="stSidebar"] * { color: #EAF2FA !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1500px; }
h1,h2,h3,h4,p, label, .stMarkdown { color: #F8FAFC; }
small, .stCaption, [data-testid="stCaptionContainer"] { color: #B8C5D6 !important; }
div[data-testid="stMetric"] { background: #192840; border: 1px solid #2A405D; padding: 15px 17px; border-radius: 13px; }
[data-testid="stMetricLabel"] { color: #B8C5D6 !important; }
[data-testid="stMetricValue"] { color: #22D3EE !important; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #2A405D !important; }
.stButton > button { background: #22D3EE; color: #07111F; border: 0; border-radius: 9px; font-weight: 700; }
.stButton > button:hover { background: #67E8F9; color: #07111F; }
div[data-baseweb="select"] > div, input, textarea { background-color: #192840 !important; color: #F8FAFC !important; }
[data-testid="stDataFrame"] { border: 1px solid #2A405D; border-radius: 10px; }
hr { border-color: #2A405D; }
.hero { background: linear-gradient(110deg,#14243B,#193B50); border: 1px solid #2A506A; padding: 24px; border-radius: 16px; margin-bottom: 18px; }
.hero p { color: #C5D5E5; margin-bottom: 0; }
.pill { display:inline-block; padding:4px 10px; border-radius:20px; background:#123B4A; color:#67E8F9; font-size:12px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

LABEL_NAMES = {
    "D": "Depression", "DD": "Deep Depression", "CS": "Cyclonic Storm",
    "SCS": "Severe Cyclonic Storm", "VSCS": "Very Severe Cyclonic Storm",
    "ESCS": "Extremely Severe Cyclonic Storm", "SuCS": "Super Cyclonic Storm"
}
LABEL_ORDER = ["D", "DD", "CS", "SCS", "VSCS", "ESCS", "SuCS"]

@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        return None
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_data():
    if not DATA_FILE.exists():
        return None
    d = pd.read_csv(DATA_FILE)
    d["time"] = pd.to_datetime(d["time"], errors="coerce")
    return d

bundle = load_model()
data = load_data()
if bundle is None:
    st.error("Trained model not found. From the project folder, run: python train_model.py")
    st.stop()
if data is None:
    st.error("Dataset not found. Check data/indian_cyclone_cleaned_engineered.csv")
    st.stop()

summary = bundle["summary"]
st.sidebar.markdown("## 🌀 Indian Cyclone")
st.sidebar.caption("North Indian Ocean · BOB + ARB")
page = st.sidebar.radio("NAVIGATION", [
    "Home Dashboard", "Cyclone Prediction", "Model Performance",
    "Dataset Analysis", "Indian Cyclone Map"
])
st.sidebar.divider()
st.sidebar.caption(f"Dataset: {summary['observations']:,} observations")
st.sidebar.caption(f"Storms: {summary['storms']:,}")
st.sidebar.caption("Academic decision-support project")

if page == "Home Dashboard":
    st.markdown('<div class="hero"><span class="pill">INDIA-SPECIFIC ML PROJECT</span><h1>Indian Cyclone Intelligence</h1><p>Historical analysis and machine-learning classification of cyclone intensity across the Bay of Bengal and Arabian Sea.</p></div>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Observations", f"{summary['observations']:,}")
    c2.metric("Unique cyclones", f"{summary['storms']:,}")
    c3.metric("Bay of Bengal", f"{summary['bob_records']:,}")
    c4.metric("Arabian Sea", f"{summary['arb_records']:,}")
    st.caption(f"Dataset time coverage: {summary['first_year']}–{summary['last_year']}. Counts are based on the supplied CSV.")
    left,right = st.columns([1.25,1])
    with left:
        st.subheader("Intensity category distribution")
        counts = data["grade"].value_counts().reindex(LABEL_ORDER, fill_value=0).rename_axis("Grade").reset_index(name="Observations")
        fig = px.bar(counts, x="Grade", y="Observations", color="Grade", text="Observations",
                     color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Records by basin")
        basin = data["basin"].value_counts().rename_axis("Basin").reset_index(name="Records")
        fig = px.pie(basin, names="Basin", values="Records", hole=.58,
                     color_discrete_sequence=["#22D3EE","#3B82F6"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    st.info(f"Current deployed classifier: **{bundle['best_name']}**. The model uses observed wind and other inputs, so this is current-intensity classification, not a forecast of future intensity.")

elif page == "Cyclone Prediction":
    st.markdown('<div class="hero"><h1> Cyclone Intensity Prediction</h1><p>Enter an observed or hypothetical set of cyclone conditions to classify its intensity grade.</p></div>', unsafe_allow_html=True)
    defaults = bundle["defaults"]
    with st.form("prediction_form"):
        a,b,c = st.columns(3)
        basin = a.selectbox("Basin", options=["BOB","ARB"], format_func=lambda x: "Bay of Bengal (BOB)" if x=="BOB" else "Arabian Sea (ARB)")
        lat = b.number_input("Latitude (°)", min_value=-5.0, max_value=35.0, value=float(np.clip(defaults["lat"],-5,35)), step=0.1)
        lon = c.number_input("Longitude (°)", min_value=40.0, max_value=110.0, value=float(np.clip(defaults["lon"],40,110)), step=0.1)
        d,e,f = st.columns(3)
        wind = d.number_input("Maximum sustained wind (knots)", min_value=0.0, max_value=200.0, value=float(defaults["wind"]), step=1.0)
        pressure = e.number_input("Central pressure (hPa)", min_value=850.0, max_value=1100.0, value=float(defaults["pressure"]) if np.isfinite(defaults["pressure"]) else 1000.0, step=1.0)
        pressure_drop = f.number_input("Pressure drop (hPa)", min_value=0.0, max_value=200.0, value=float(defaults["pressure_drop"]), step=1.0)
        g,h = st.columns(2)
        obs_date = g.date_input("Observation date", value=date.today())
        hour = h.slider("Observation hour (24-hour clock)", 0, 23, 12)
        submitted = st.form_submit_button("Classify cyclone intensity", use_container_width=True)
    if submitted:
        day_of_year = obs_date.timetuple().tm_yday
        angle = 2*np.pi*day_of_year/365.25
        row = pd.DataFrame([{
            "lat":lat, "lon":lon, "wind":wind, "pressure":pressure,
            "pressure_drop":pressure_drop, "month":obs_date.month, "hour":hour,
            "day_of_year_sin":np.sin(angle), "day_of_year_cos":np.cos(angle),
            "basin":basin
        }])[bundle["features"]]
        pred = bundle["model"].predict(row)[0]
        st.divider()
        st.success(f"Predicted intensity: {LABEL_NAMES.get(pred, pred)} ({pred})")
        st.markdown(f"### {LABEL_NAMES.get(pred, pred)}")
        st.write(f"Category code: **{pred}**")
        st.write(f"Model: **{bundle['best_name']}**")
        st.write(f"Input basin: **{'Bay of Bengal' if basin=='BOB' else 'Arabian Sea'}**")
        st.caption("No confidence percentage is displayed because prediction probabilities have not been calibrated and validated for this project.")

elif page == "Model Performance":
    st.markdown('<div class="hero"><h1>Model Performance</h1><p>Storm-aware evaluation with two feature experiments. Metrics below are calculated on a held-out set of complete cyclones.</p></div>', unsafe_allow_html=True)
    a = bundle["experiments"]["with_wind"]
    b = bundle["experiments"]["without_wind"]
    st.subheader("Experiment A · With wind")
    st.caption(f"Selected model by grouped cross-validation macro F1: {a['best_name']}")
    m = a["metrics"]
    cols=st.columns(4)
    for col,(label,key) in zip(cols,[("Accuracy","accuracy"),("Macro F1","f1_macro"),("Weighted F1","f1_weighted"),("Balanced accuracy","balanced_accuracy")]):
        col.metric(label, f"{m[key]:.3f}")
    st.subheader("Experiment B · Without wind")
    st.caption(f"Selected model by grouped cross-validation macro F1: {b['best_name']}")
    mb=b["metrics"]
    cols=st.columns(4)
    for col,(label,key) in zip(cols,[("Accuracy","accuracy"),("Macro F1","f1_macro"),("Weighted F1","f1_weighted"),("Balanced accuracy","balanced_accuracy")]):
        col.metric(label, f"{mb[key]:.3f}")
    st.info("Wind is strongly related to the cyclone intensity grade. Experiment A is therefore a current-intensity classification experiment when wind is observed. Experiment B shows performance without wind. Neither experiment should be described as future forecasting.")
    comp_path = BASE/"reports"/"model_comparison.csv"
    if comp_path.exists():
        st.subheader("Model comparison (training-only grouped CV)")
        comp=pd.read_csv(comp_path)
        st.dataframe(comp[["Experiment","Model","CV Macro F1"]].sort_values(["Experiment","CV Macro F1"],ascending=[True,False]), use_container_width=True, hide_index=True)
    st.caption(f"Storm-aware split: {summary['train_storms']} training storms and {summary['test_storms']} test storms; no storm is shared between partitions.")
    st.caption("Detailed classification reports and confusion matrices are saved in the reports/ folder.")

elif page == "Dataset Analysis":
    st.markdown('<div class="hero"><h1>Dataset Analysis</h1><p>Explore the supplied cleaned and engineered Indian cyclone observations.</p></div>', unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    c1.metric("Rows",f"{len(data):,}"); c2.metric("Columns",len(data.columns)); c3.metric("Unique storms",f"{data['storm_id'].nunique():,}")
    st.subheader("Dataset preview")
    st.dataframe(data.head(30),use_container_width=True,hide_index=True)
    left,right=st.columns(2)
    with left:
        st.subheader("Missing values")
        miss=data.isna().sum().rename("Missing").to_frame()
        miss["Percent"]=100*miss["Missing"]/len(data)
        st.dataframe(miss,use_container_width=True)
    with right:
        st.subheader("Intensity by basin")
        cross=pd.crosstab(data["grade"],data["basin"]).reindex(LABEL_ORDER).fillna(0)
        st.dataframe(cross,use_container_width=True)
    left,right=st.columns(2)
    with left:
        st.subheader("Wind distribution by intensity")
        fig=px.box(data,x="grade",y="wind",category_orders={"grade":LABEL_ORDER},color="grade",color_discrete_sequence=px.colors.sequential.Teal)
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.subheader("Historical observations by year")
        yearly=data.groupby("year").size().rename("Records").reset_index()
        fig=px.line(yearly,x="year",y="Records",markers=True)
        fig.update_traces(line_color="#22D3EE")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    st.subheader("Pressure distribution")
    fig=px.box(data,x="grade",y="pressure",category_orders={"grade":LABEL_ORDER},color="grade",color_discrete_sequence=px.colors.sequential.Teal)
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",showlegend=False)
    st.plotly_chart(fig,use_container_width=True)

else:
    st.markdown('<div class="hero"><h1>Indian Cyclone Map</h1><p>Recorded historical positions from the uploaded dataset. Points are observations, not interpolated storm tracks.</p></div>', unsafe_allow_html=True)
    map_data=data.dropna(subset=["lat","lon"]).copy()
    basin_filter=st.multiselect("Show basin",["BOB","ARB"],default=["BOB","ARB"],format_func=lambda x:"Bay of Bengal" if x=="BOB" else "Arabian Sea")
    map_data=map_data[map_data["basin"].isin(basin_filter)]
    fig=px.scatter_geo(map_data,lat="lat",lon="lon",color="basin",hover_name="storm_id",
        hover_data={"time":True,"grade":True,"wind":True,"basin":True,"lat":False,"lon":False},
        color_discrete_map={"BOB":"#22D3EE","ARB":"#F59E0B"})
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#58708B",
        showland=True, landcolor="#17283D", showocean=True, oceancolor="#0B192B",
        lataxis_range=[-5,35], lonaxis_range=[40,110])
    fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",geo_bgcolor="#0B1220",height=620,legend_title_text="Basin")
    st.plotly_chart(fig,use_container_width=True)
    st.caption(f"Showing {len(map_data):,} actual coordinate observations. The map does not create missing coordinates or infer a path between observations.")
