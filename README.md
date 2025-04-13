
# 🔍 NIBRS Compliance Prediction

This project predicts whether law enforcement agencies are NIBRS-compliant using a trained machine learning model. It includes data preprocessing, model training, evaluation, and a fully functional Streamlit dashboard for real-time CSV analysis and visualization.

---

## 📁 Project Structure

```
📦 NIBRS-Prediction/
├── best_model.pkl
├── scaler.pkl
├── label_encoders.pkl
├── app.py
├── agencies.csv
├── nibrsPrediction.ipynb
└── README.md
```

---

## 💡 Features

- Predicts compliance using Random Forest Classifier
- Handles raw agency CSV input
- Dashboard includes:
  - 📊 Confidence distribution
  - 🌍 U.S. map of agencies
  - 📈 Class-wise stats and insights
  - 📥 Downloadable results
- Fully interactive via Streamlit

---

## ⚙️ Tools & Tech

- Python, Pandas, NumPy, Scikit-learn
- Streamlit, Plotly, Seaborn, Matplotlib
- imbalanced-learn (SMOTE)
- Pretrained model + encoders (saved as .pkl)

---

## 🚀 How to Run

```bash

# Run the Streamlit app
streamlit run app.py
```

---

## 🧪 Sample CSV

Make sure your CSV includes the following columns:

```
ori, county, latitude, longitude, state_abbr, state,
agency_name, agency_type, is_nibrs, nibrs_start_date
```

---

## 📊 Model Performance

- Model: RandomForestClassifier
- Accuracy: ~92%
- ROC AUC Score: 0.92
- Trained on SMOTE-balanced data
- Feature Importance: Latitude, Longitude, Agency Type, County Density

---

## 🙏 Acknowledgment

Created by **Manglam Srivastav and Jasraj Singh Khalsa**  
University of Arizona – MS in Information Science & Machine Learning

