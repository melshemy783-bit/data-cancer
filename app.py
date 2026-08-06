import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_breast_cancer

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# CUSTOM CSS (background + cards)
# ============================================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0f1117 0%, #1a1d29 100%);
    }
    section[data-testid="stSidebar"] {
        background-color: #12141c;
    }
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #2c2f3f;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL + DATA
# ============================================================
model = joblib.load("svm_model.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity",
    "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error",
    "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension"
]
mean_features = feature_names[0:10]
error_features = feature_names[10:20]
worst_features = feature_names[20:30]

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

# ============================================================
# SIDEBAR: About + Predict only
# ============================================================
with st.sidebar:
    st.header("ℹ️ About")
    st.write(
        "This app uses an **SVM model** to predict whether a tumor is "
        "**Benign** or **Malignant** based on 30 cell nuclei features."
    )
    st.write("1️⃣ Fill in the patient's features in the tabs on the main page.")
    st.write("2️⃣ Come back here and click **Predict**.")
    st.markdown("---")
    st.caption("Model: SVM  •  Dataset: sklearn breast cancer")
    st.markdown("---")
    predict_clicked = st.button("🔍 Predict", use_container_width=True, type="primary")

# ============================================================
# MAIN PAGE HEADER
# ============================================================
st.title("🩺 Breast Cancer Prediction")
st.write("Explore the dataset, fill in the patient's features below, then click **Predict** in the sidebar.")

# ============================================================
# KPI CARDS (Dataset quick stats)
# ============================================================
st.markdown("## 📌 Dataset at a Glance")

total = len(df)
malignant_count = (df["diagnosis"] == "Malignant").sum()
benign_count = (df["diagnosis"] == "Benign").sum()
avg_radius = df["mean radius"].mean()
avg_area = df["mean area"].mean()

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

kpi1.metric("🧬 Total Samples", f"{total}")
kpi2.metric("🔴 Malignant", f"{malignant_count}", f"{malignant_count/total*100:.1f}%")
kpi3.metric("🟢 Benign", f"{benign_count}", f"{benign_count/total*100:.1f}%")
kpi4.metric("📏 Avg Mean Radius", f"{avg_radius:.2f}")
kpi5.metric("📐 Avg Mean Area", f"{avg_area:.1f}")

st.markdown("---")

# ============================================================
# OVERVIEW VISUALIZATIONS
# ============================================================
st.markdown("## 📊 Dataset Overview")

overview_col1, overview_col2, overview_col3 = st.columns(3)

with overview_col1:
    fig_pie = px.pie(
        df, names="diagnosis", hole=0.45,
        title="Class Distribution",
        color="diagnosis",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_pie.update_layout(height=320)
    st.plotly_chart(fig_pie, use_container_width=True)

with overview_col2:
    fig_hist = px.histogram(
        df, x="mean radius", color="diagnosis",
        barmode="overlay", nbins=35,
        title="Mean Radius Distribution",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_hist.update_layout(height=320)
    st.plotly_chart(fig_hist, use_container_width=True)

with overview_col3:
    fig_box = px.box(
        df, x="diagnosis", y="mean concavity", color="diagnosis",
        title="Mean Concavity by Diagnosis",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_box.update_layout(height=320)
    st.plotly_chart(fig_box, use_container_width=True)

overview_col4, overview_col5 = st.columns(2)

with overview_col4:
    fig_scatter = px.scatter(
        df, x="mean radius", y="mean texture", color="diagnosis",
        title="Mean Radius vs Mean Texture",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with overview_col5:
    corr = df[mean_features].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title="Correlation Heatmap (Mean Features)",
        color_continuous_scale="RdBu_r"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ============================================================
# MORE INSIGHTS
# ============================================================
st.markdown("## 🔬 More Insights")

insight_col1, insight_col2 = st.columns(2)

with insight_col1:
    fig_violin = px.violin(
        df, x="diagnosis", y="worst area", color="diagnosis", box=True,
        title="Worst Area Distribution by Diagnosis",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_violin.update_layout(height=350)
    st.plotly_chart(fig_violin, use_container_width=True)

with insight_col2:
    fig_3d = px.scatter_3d(
        df, x="mean radius", y="mean texture", z="mean concavity",
        color="diagnosis",
        title="Mean Radius vs Texture vs Concavity",
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_3d.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

# Feature correlation with diagnosis (which features matter most)
st.markdown("### 🏆 Features Most Correlated with Diagnosis")
corr_with_target = df[feature_names + ["target"]].corr()["target"].drop("target")
corr_with_target = corr_with_target.abs().sort_values(ascending=False).head(10)

fig_corr_bar = px.bar(
    x=corr_with_target.values,
    y=corr_with_target.index,
    orientation="h",
    title="Top 10 Features Correlated with Diagnosis",
    labels={"x": "Correlation (absolute)", "y": "Feature"},
    color=corr_with_target.values,
    color_continuous_scale="Reds"
)
fig_corr_bar.update_layout(height=400, yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_corr_bar, use_container_width=True)

st.markdown("---")

# ============================================================
# PATIENT INPUTS — organized in tabs
# ============================================================
st.markdown("## 📝 Patient Features")

tab1, tab2, tab3 = st.tabs(["Mean Values", "Error Values", "Worst Values"])

user_input = {}

with tab1:
    cols = st.columns(2)
    for i, feature in enumerate(mean_features):
        with cols[i % 2]:
            user_input[feature] = st.number_input(feature, value=0.0, format="%.4f", key=feature)

with tab2:
    cols = st.columns(2)
    for i, feature in enumerate(error_features):
        with cols[i % 2]:
            user_input[feature] = st.number_input(feature, value=0.0, format="%.4f", key=feature)

with tab3:
    cols = st.columns(2)
    for i, feature in enumerate(worst_features):
        with cols[i % 2]:
            user_input[feature] = st.number_input(feature, value=0.0, format="%.4f", key=feature)

st.markdown("---")

# ============================================================
# PREDICTION SECTION
# ============================================================
if predict_clicked:

    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]

    st.markdown("## 🔍 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns([1, 1, 1.3])

    with result_col1:
        if prediction == 0:
            st.error("🔴 Malignant\n(Cancer)")
        else:
            st.success("🟢 Benign\n(Not Cancer)")

    with result_col2:
        st.metric("Malignant", f"{probability[0]*100:.2f}%")
        st.metric("Benign", f"{probability[1]*100:.2f}%")

    with result_col3:
        gauge_value = probability[1] * 100
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={'text': "Benign Probability %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if gauge_value > 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "#3a1f1f"},
                    {'range': [50, 100], 'color': "#1f3a24"}
                ],
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")

    bar_col, radar_col = st.columns(2)

    with bar_col:
        fig_bar = go.Figure(data=[
            go.Bar(
                x=["Malignant", "Benign"],
                y=[probability[0]*100, probability[1]*100],
                marker_color=["#ff4b4b", "#2ecc71"],
                text=[f"{probability[0]*100:.2f}%", f"{probability[1]*100:.2f}%"],
                textposition="auto"
            )
        ])
        fig_bar.update_layout(title="Probability Comparison", yaxis_title="Probability (%)", height=380)
        st.plotly_chart(fig_bar, use_container_width=True)

    with radar_col:
        radar_features = mean_features[:6]
        patient_vals = [user_input[f] for f in radar_features]
        max_vals = [df[f].max() for f in radar_features]
        avg_vals = [df[f].mean() for f in radar_features]

        patient_norm = [p / m if m else 0 for p, m in zip(patient_vals, max_vals)]
        avg_norm = [a / m if m else 0 for a, m in zip(avg_vals, max_vals)]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=patient_norm, theta=radar_features, fill='toself', name='Patient'))
        fig_radar.add_trace(go.Scatterpolar(r=avg_norm, theta=radar_features, fill='toself', name='Dataset Average'))
        fig_radar.update_layout(
            title="Patient vs Dataset Average (normalized)",
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Where does the patient fall in the overall distribution?
    st.markdown("### 📍 Patient Position vs Dataset (Mean Radius)")
    fig_pos = px.histogram(
        df, x="mean radius", color="diagnosis", barmode="overlay", nbins=35,
        color_discrete_map={"Malignant": "#ff4b4b", "Benign": "#2ecc71"}
    )
    fig_pos.add_vline(
        x=user_input["mean radius"], line_width=3, line_dash="dash", line_color="white",
        annotation_text="Patient", annotation_position="top"
    )
    fig_pos.update_layout(height=350)
    st.plotly_chart(fig_pos, use_container_width=True)

else:
    st.info("👈 Fill in the patient's features above, then click **Predict** in the sidebar.")