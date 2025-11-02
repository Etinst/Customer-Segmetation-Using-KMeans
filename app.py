import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="Machine Learning Project Dashboard", layout="wide")

st.sidebar.title("📂 Data Controls")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data uploaded successfully!")
else:
    st.sidebar.info("ℹ️ Using default dataset (Mall_Customers.csv if available).")
    try:
        df = pd.read_csv("Mall_Customers.csv")
    except Exception:
        st.error("No dataset found. Please upload a CSV to continue.")
        st.stop()

st.title("📊 Machine Learning Project Visualization Dashboard")
tab1, tab2, tab3 = st.tabs(["Data Overview", "Model Results", "Visualizations"])

with tab1:
    st.header("🔍 Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.write("Summary Statistics:")
    st.dataframe(df.describe())

    numeric_df = df.select_dtypes(include=["number"])

    if not numeric_df.empty:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available for heatmap.")

with tab2:
    st.header("🤖 Model Summary & Performance")
    accuracy = 0.89
    silhouette = 0.71
    cluster_count = 5

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Silhouette Score", f"{silhouette:.2f}")
    col3.metric("Clusters Formed", cluster_count)

with tab3:
    st.header("📈 Visualization Section")
    chart_type = st.selectbox("Choose Visualization Type", ["Scatter Plot", "Bar Chart", "Distribution", "Custom Plot"])

    if chart_type == "Scatter Plot":
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if len(numeric_cols) >= 2:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0)
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1)
            color_col = st.selectbox("Color (optional)", [None] + numeric_cols)
            fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Not enough numeric columns for scatter plot.")

    elif chart_type == "Bar Chart":
        cols = df.columns.tolist()
        x_col = st.selectbox("X-axis", cols)
        y_col = st.selectbox("Y-axis", cols)
        fig = px.bar(df, x=x_col, y=y_col)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Distribution":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            col = st.selectbox("Select column", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numeric columns to plot distribution.")

    elif chart_type == "Custom Plot":
        code = st.text_area(
            "Python code",
            "plt.figure(figsize=(6,4))\nsns.boxplot(x='Gender', y='Spending Score (1-100)', data=df)\nst.pyplot()"
        )
        try:
            exec(code)
        except Exception as e:
            st.error(f"Error running code: {e}")

st.markdown("---")
st.caption("© 2025 Streamlit Visualization Framework for ML Projects")
