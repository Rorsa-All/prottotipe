import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# App title
st.title("Simple App for Model Training and Predictions")

# Step 1: Upload data
st.header("Step 1: Upload your data")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data successfully loaded!")
    st.write(data.head())
    
    # Step 2: Select target and features
    st.header("Step 2: Select the target (Y) and features (X)")
    
    target_column = st.selectbox("Select the target variable (Y)", data.columns)
    feature_columns = st.multiselect("Select the feature columns (X)", data.columns, default=data.columns[:-1])

    # Ensure selections are not empty before proceeding
    if target_column and len(feature_columns) > 0:
        X = data[feature_columns]
        y = data[target_column]

        # Step 3: Train the model
        st.header("Step 3: Train the model")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.write("Model trained successfully!")

        # Step 4: Make predictions
        st.header("Step 4: Make predictions on the test data")
        y_pred = model.predict(X_test)
        
        st.write("Actual values vs Predictions:")
        st.write(pd.DataFrame({"Actual values": y_test, "Predictions": y_pred}).head())
    else:
        st.write("Please select both a target and at least one feature.")
