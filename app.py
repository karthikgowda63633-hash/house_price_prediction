import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

numeric_features = [
    "Overall Qual", "Overall Cond", "Gr Liv Area", "Garage Cars", "Garage Area",
    "Total Bsmt SF", "1st Flr SF", "2nd Flr SF", "Year Built", "Year Remod/Add",
    "Full Bath", "Half Bath", "Bedroom AbvGr", "Kitchen AbvGr", "TotRms AbvGrd",
    "Fireplaces", "Lot Area", "Mas Vnr Area", "BsmtFin SF 1", "Wood Deck SF",
    "Open Porch SF", "Bsmt Full Bath"
]
categorical_features = [
    "Neighborhood", "MS Zoning", "Bldg Type", "House Style", "Exter Qual",
    "Kitchen Qual", "Bsmt Qual", "Heating QC", "Central Air", "Foundation",
    "Sale Condition", "Bsmt Exposure", "Garage Finish", "Fireplace Qu"
]
features = numeric_features + categorical_features
size_features = ["Gr Liv Area", "Garage Area", "Total Bsmt SF", "1st Flr SF", "Lot Area"]


@st.cache_resource
def train_model():
    df = pd.read_csv("housing_data.csv").dropna(subset=["SalePrice"]).drop_duplicates().copy()
    X = df[features].copy()
    y = df["SalePrice"].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    imputer = SimpleImputer(strategy="median")
    limits = {}
    for col in size_features:
        filled = X_train[col].fillna(X_train[col].median())
        q1, q3 = filled.quantile(0.25), filled.quantile(0.75)
        iqr = q3 - q1
        limits[col] = (max(0, q1 - 1.5 * iqr), q3 + 1.5 * iqr)

    imputer.fit(X_train[numeric_features])
    dummy_cols = pd.get_dummies(
        X_train[categorical_features].fillna("None"), drop_first=True
    ).columns

    def prepare(X_raw):
        num = pd.DataFrame(imputer.transform(X_raw[numeric_features]),
                           columns=numeric_features, index=X_raw.index)
        for col, (lo, hi) in limits.items():
            num[col] = num[col].clip(lo, hi)
        cat = pd.get_dummies(X_raw[categorical_features].fillna("None"), drop_first=True)
        cat = cat.reindex(columns=dummy_cols, fill_value=0).astype(float)
        return pd.concat([num, cat], axis=1)

    model = LinearRegression().fit(prepare(X_train), y_train)
    pred = model.predict(prepare(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    return model, prepare, df, rmse, r2


model, prepare, df, rmse, r2 = train_model()

st.title("House Price Prediction (Linear Regression)")
c1, c2 = st.columns(2)
c1.metric("Test RMSE", f"{rmse:,.0f}")
c2.metric("Test R-squared", f"{r2:.3f}")

main_features = [
    "1st Flr SF", "2nd Flr SF", "Overall Qual", "Overall Cond", "Neighborhood",
    "Total Bsmt SF", "Year Built", "Bldg Type", "Kitchen Qual", "Bsmt Qual"
]
optional_features = [
    "Bsmt Exposure", "Exter Qual", "Lot Area", "Garage Cars", "Fireplaces"
]

# every feature starts at its default (median for numeric, most common for categorical)
inputs = {}
for col in numeric_features:
    inputs[col] = int(round(df[col].median()))
for col in categorical_features:
    inputs[col] = df[col].fillna("None").astype(str).mode()[0]


def show_input(container, col):
    if col in numeric_features:
        s = df[col]
        inputs[col] = container.number_input(
            col, min_value=int(s.min()), max_value=int(s.max()), value=inputs[col], step=1
        )
    else:
        options = sorted(df[col].fillna("None").astype(str).unique())
        inputs[col] = container.selectbox(col, options, index=options.index(inputs[col]))


st.subheader("Main inputs")
cols = st.columns(2)
for i, col in enumerate(main_features):
    show_input(cols[i % 2], col)

with st.expander("Optional inputs (defaults are used if you leave them)"):
    cols = st.columns(2)
    for i, col in enumerate(optional_features):
        show_input(cols[i % 2], col)

st.caption("Gr Liv Area is calculated as 1st Flr SF + 2nd Flr SF. All other features not shown are set to their default values.")

if st.button("Predict SalePrice"):
    inputs["Gr Liv Area"] = inputs["1st Flr SF"] + inputs["2nd Flr SF"]
    row = pd.DataFrame([inputs])[features]
    price = model.predict(prepare(row))[0]
    st.success(f"Predicted SalePrice: ${price:,.0f}")