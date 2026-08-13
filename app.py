import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction")
st.write("Enter car details to predict the price.")

# Load dataset
df = pd.read_csv("car_price.csv")

# Remove ID
if "ID" in df.columns:
    df = df.drop("ID", axis=1)

# Clean data
df["Levy"] = pd.to_numeric(df["Levy"], errors="coerce")

df["Mileage"] = (
    df["Mileage"].astype(str)
    .str.replace(" km", "", regex=False)
    .str.replace(",", "", regex=False)
)
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")

df["Engine volume"] = pd.to_numeric(
    df["Engine volume"].astype(str).str.extract(r"([0-9.]+)")[0],
    errors="coerce"
)

df["Prod. year"] = pd.to_numeric(df["Prod. year"], errors="coerce")

df = df.dropna()

# Features and target
X = df.drop("Price", axis=1)
y = df["Price"]

categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("num", "passthrough", numeric)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

model.fit(X, y)

# Input
st.subheader("Enter Car Details")

levy = st.number_input("Levy", min_value=0, value=500)

manufacturer = st.selectbox(
    "Manufacturer",
    sorted(df["Manufacturer"].unique())
)

car_model = st.selectbox(
    "Model",
    sorted(df["Model"].unique())
)

year = st.number_input(
    "Production Year",
    min_value=1990,
    max_value=2026,
    value=2018
)

category = st.selectbox(
    "Category",
    sorted(df["Category"].unique())
)

leather = st.selectbox(
    "Leather Interior",
    sorted(df["Leather interior"].unique())
)

fuel = st.selectbox(
    "Fuel Type",
    sorted(df["Fuel type"].unique())
)

engine = st.number_input(
    "Engine Volume",
    min_value=0.5,
    max_value=10.0,
    value=2.0
)

mileage = st.number_input(
    "Mileage (km)",
    min_value=0,
    value=100000
)

if st.button("🔮 Predict Price"):

    input_data = pd.DataFrame({
        "Levy": [levy],
        "Manufacturer": [manufacturer],
        "Model": [car_model],
        "Prod. year": [year],
        "Category": [category],
        "Leather interior": [leather],
        "Fuel type": [fuel],
        "Engine volume": [engine],
        "Mileage": [mileage]
    })

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Estimated Car Price: ${prediction:,.2f}")