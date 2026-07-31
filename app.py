import streamlit as st
import pandas as pd

# ------------------- Page Configuration -------------------
st.set_page_config(
    page_title="Student Grade Prediction System",
    page_icon="🎓",
    layout="wide"
)

# ------------------- Sidebar -------------------
st.sidebar.title("🎓 Student Grade Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "📂 Upload Dataset",
        "📊 Dataset Analysis",
        "🤖 Train Model",
        "ℹ About"
    ]
)

# ------------------- Dashboard -------------------
if page == "🏠 Dashboard":

    st.title("🎓 Student Grade Prediction System")
    st.write("### Machine Learning Based Student Performance Prediction")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Students", "379")
    col2.metric("Input Features", "SEM1 - SEM4")
    col3.metric("Target", "SEM5")

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.info("📂 Upload Dataset")

    with c2:
        st.success("🤖 Train Machine Learning Model")

    st.subheader("Project Overview")

    st.write("""
This project predicts **Semester 5 CGPA**
using previous semester performances.
""")

# ------------------- Upload -------------------
elif page == "📂 Upload Dataset":

    st.title("📂 Upload Dataset")

    uploaded = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded is not None:

        df = pd.read_csv(uploaded)

        st.success("Dataset Uploaded Successfully!")

        st.write(df.head())

# ------------------- Dataset Analysis -------------------
elif page == "📊 Dataset Analysis":

    st.title("📊 Dataset Analysis")

    try:

        df = pd.read_csv("student grade pred data set.csv")

        st.subheader("Dataset")

        st.dataframe(df)

        st.subheader("Statistics")

        st.write(df.describe())

    except:

        st.error("Dataset not found.")

# ------------------- Train Model -------------------
elif page == "🤖 Train Model":

    st.title("🤖 Train Model")

    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    try:
        df = pd.read_csv("student grade pred data set.csv")

        X = df[['SEM 1', 'SEM 2', 'SEM 3', 'SEM 4']]
        y = df['SEM 5']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        st.success("✅ Model Trained Successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("MAE", f"{mae:.4f}")
            st.metric("MSE", f"{mse:.4f}")

        with col2:
            st.metric("RMSE", f"{rmse:.4f}")
            st.metric("R² Score", f"{r2:.4f}")

    except Exception as e:
        st.error(f"Error: {e}")

# ------------------- About -------------------
elif page == "ℹ About":

    st.title("ℹ About Project")

    st.write("""
### Student Grade Prediction System

**Developed Using**

- Python
- Streamlit
- Pandas
- Scikit-Learn
- Machine Learning

This project predicts Semester 5 CGPA based on Semester 1–4 CGPA.
""")