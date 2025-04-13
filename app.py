# ------------------------------
# Streamlit App: NIBRS Compliance Predictor (Final Stats Edition)
# ------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer

# ------------------------------
# Setup
# ------------------------------
st.set_page_config(page_title="NIBRS Predictor", layout="wide")

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.title("📊 About This App")
    st.markdown("""
This app predicts whether law enforcement agencies are **NIBRS compliant** using a trained machine learning model.

#### How to Use:
1. 📂 Upload your `agencies.csv`  
2. ✅ View predictions & confidence  
3. 🌍 Explore the U.S. agency map  
4. 📥 Download full results  
""")

# ------------------------------
# Load model and encoders
# ------------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")  # Optional if needed
label_encoders = joblib.load("label_encoders.pkl")

# ------------------------------
# Main UI
# ------------------------------
st.title("🔍 NIBRS Compliance Predictor")
st.markdown("Upload a **raw agency CSV** (like `agencies.csv`) to predict NIBRS compliance status.")

uploaded_file = st.file_uploader("📂 Upload raw agency CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Data Preview")
        st.dataframe(df.head(10))

        # ------------------------------
        # Preprocessing
        # ------------------------------
        df = df.drop(columns=['ori', 'agency_name', 'nibrs_start_date', 'nibrs_start_year'], errors='ignore')
        df['state_abbr'] = df['state_abbr'].str.upper().str.strip()
        df['county'] = df['county'].str.upper().str.strip()
        df['agency_type'] = df['agency_type'].fillna(df['agency_type'].mode()[0])
        df[['latitude', 'longitude']] = SimpleImputer(strategy='mean').fit_transform(df[['latitude', 'longitude']])
        df['agency_density_per_county'] = df['county'].map(df['county'].value_counts())

        # Encode categoricals
        categorical_columns = ['state', 'state_abbr', 'agency_type', 'county']
        unk_count_report = []

        for col in categorical_columns:
            le = label_encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "<UNK>")
            if "<UNK>" not in le.classes_:
                le.classes_ = np.append(le.classes_, "<UNK>")
            df[col] = le.transform(df[col])
            unk_count_report.append(f"{col}: {(df[col] == le.transform(['<UNK>'])[0])} unknowns")

        # Required features
        required_cols = ['county', 'latitude', 'longitude', 'state_abbr', 'state', 'agency_type', 'agency_density_per_county']
        if not all(col in df.columns for col in required_cols):
            st.error("❌ Uploaded CSV is missing required columns.")
        else:
            df_input = df[required_cols].copy()
            preds = model.predict(df_input)
            proba = model.predict_proba(df_input)[:, 1]

            df['NIBRS_Prediction'] = preds
            df['Probability'] = proba

            st.success("✅ Prediction complete!")

            # ------------------------------
            # 📊 Enhanced Summary Dashboard
            # ------------------------------
            st.markdown("### 📊 Summary Dashboard")

            total = len(df)
            compliant = int(preds.sum())
            non_compliant = total - compliant

            low_conf = df[df['Probability'] < 0.4].shape[0]
            mid_conf = df[df['Probability'].between(0.4, 0.6)].shape[0]
            high_conf = df[df['Probability'] > 0.6].shape[0]

            avg_prob = proba.mean()
            std_prob = proba.std()
            min_prob = proba.min()
            max_prob = proba.max()
            median_prob = np.median(proba)

            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Compliant", compliant)
            col2.metric("❌ Non-Compliant", non_compliant)
            col3.metric("📦 Total Agencies", total)

            col4, col5, col6 = st.columns(3)
            col4.metric("📉 Min Confidence", f"{min_prob:.2f}")
            col5.metric("📈 Max Confidence", f"{max_prob:.2f}")
            col6.metric("📌 Std Dev", f"{std_prob:.2f}")

            col7, col8, col9 = st.columns(3)
            col7.metric("⚖️ Median", f"{median_prob:.2f}")
            col8.metric("🧠 Average", f"{avg_prob:.2%}")
            col9.metric("⚠️ Low Confidence (0.4–0.6)", mid_conf)

            with st.expander("🔍 Confidence Buckets Breakdown"):
                st.markdown(f"""
- 🔴 **Low (< 0.4):** `{low_conf}` agencies  
- 🟡 **Medium (0.4–0.6):** `{mid_conf}` agencies  
- 🟢 **High (> 0.6):** `{high_conf}` agencies  
                """)

            # ------------------------------
            # 📈 Histogram of Confidence
            # ------------------------------
            st.markdown("### 📈 Prediction Confidence Histogram")
            fig_hist, ax = plt.subplots(figsize=(10, 6))
            df['Probability'].plot.hist(bins=20, color='#4CAF50', edgecolor='white', ax=ax)
            ax.set_xlabel("Probability of Compliance")
            ax.set_ylabel("Agency Count")
            ax.set_title("Confidence Distribution")
            st.pyplot(fig_hist)

            # ------------------------------
            # 🌍 US Map
            # ------------------------------
            st.markdown("### 🌍 U.S. Map: Agency Prediction Distribution")
            fig_map = px.scatter_geo(
                df,
                lat='latitude',
                lon='longitude',
                color='NIBRS_Prediction',
                color_discrete_map={0: 'red', 1: 'green'},
                hover_name='county',
                scope='usa',
                title="Agency Locations Across the U.S.",
                opacity=0.7,
                height=500,
            )
            fig_map.update_traces(marker=dict(size=6))
            st.plotly_chart(fig_map, use_container_width=True)

            # ------------------------------
            # 🔍 Detailed Prediction Table
            # ------------------------------
            st.markdown("### 🔍 Detailed Predictions")
            st.dataframe(df[['NIBRS_Prediction', 'Probability'] + required_cols].head(20))

            # ------------------------------
            # 📥 Download
            # ------------------------------
            csv_output = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Full Predictions CSV", csv_output, "nibrs_predictions.csv")

            # Warn for unknowns
            if any("unknowns" in line for line in unk_count_report):
                st.warning("⚠️ Some unseen categories were encoded as '<UNK>':")
                for line in unk_count_report:
                    st.text(line)

    except Exception as e:
        st.error(f"🚨 Error processing file: {e}")
else:
    st.info("🕒 Upload a CSV file to begin.")
