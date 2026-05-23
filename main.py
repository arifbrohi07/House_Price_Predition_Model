import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="California Housing Prediction",
    layout="wide"
)

st.title("🏠 California Housing Price Prediction")
st.write("Machine Learning Model using Random Forest Regressor")

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("housing.csv")
    return df

df = load_data()

# =====================================
# DATA OVERVIEW
# =====================================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("📌 Dataset Shape")
st.write(df.shape)

# =====================================
# HANDLE MISSING VALUES
# =====================================
df['total_bedrooms'] = df['total_bedrooms'].fillna(
    df['total_bedrooms'].median()
)

# =====================================
# FEATURES & TARGET
# =====================================
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# =====================================
# NUMERICAL & CATEGORICAL COLUMNS
# =====================================
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# =====================================
# PREPROCESSING
# =====================================
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_transformer, num_cols),
    ('cat', cat_transformer, cat_cols)
])

# =====================================
# TRAIN TEST SPLIT
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================
# MODEL
# =====================================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# =====================================
# TRAIN MODEL
# =====================================
model.fit(X_train, y_train)

# =====================================
# PREDICTIONS
# =====================================
y_pred = model.predict(X_test)

# =====================================
# EVALUATION
# =====================================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# =====================================
# METRICS
# =====================================
st.subheader("📈 Model Accuracy")

col1, col2, col3, col4 = st.columns(4)

col1.metric("MAE", f"{mae:.2f}")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("R² Score", f"{r2:.2f}")

# =====================================
# FEATURE IMPORTANCE
# =====================================
regressor = model.named_steps['regressor']

encoded_cat_features = model.named_steps['preprocessor'] \
    .transformers_[1][1] \
    .named_steps['encoder'] \
    .get_feature_names_out(cat_cols)

all_features = list(num_cols) + list(encoded_cat_features)

importance = regressor.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': importance
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

# =====================================
# FEATURE IMPORTANCE GRAPH
# =====================================
st.subheader(" Top 10 Important Features")

fig, ax = plt.subplots(figsize=(10, 6))

top_features = feature_importance.head(10)

ax.barh(
    top_features['Feature'],
    top_features['Importance']
)

ax.set_xlabel("Importance")
ax.set_ylabel("Features")
ax.set_title("Top 10 Feature Importance")

st.pyplot(fig)

# =====================================
# USER INPUT SECTION
# =====================================
st.subheader("🏡 Predict House Price")

longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

housing_median_age = st.number_input(
    "Housing Median Age",
    value=41
)

total_rooms = st.number_input(
    "Total Rooms",
    value=880
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    value=129
)

population = st.number_input(
    "Population",
    value=322
)

households = st.number_input(
    "Households",
    value=126
)

median_income = st.number_input(
    "Median Income",
    value=8.3252
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    df['ocean_proximity'].unique()
)

# =====================================
# PREDICT BUTTON
# =====================================
if st.button("Predict House Price"):

    input_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [housing_median_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

    prediction = model.predict(input_data)

    st.success(
        f"🏠 Predicted House Price: ${prediction[0]:,.2f}"
    )

# =====================================
# FOOTER
# =====================================
st.markdown("---")
st.write("Developed using Streamlit & Scikit-learn")