import joblib
import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import os
import streamlit as st

MODEL_DIR = 'saved_models'

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    st.write(f"Directory created: {MODEL_DIR}")
else:
    st.write(f"Directory already exists: {MODEL_DIR}")

def save_model(model, model_name):
    if not model_name.endswith('.pkl'):
        model_name += '.pkl'
    filename = os.path.abspath(os.path.join(MODEL_DIR, model_name))
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"Model successfully saved as {filename}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

def main():
    st.title("Model Save Test")

    if st.button("Create and Train Model"):
        iris = load_iris()
        X, y = iris.data, iris.target

        model = LogisticRegression(max_iter=200)
        model.fit(X, y)

        model_name = st.text_input("Enter a name to save your model (e.g., 'my_model.pkl'):")

        if st.button("Save Model"):
            if model_name:
                save_model(model, model_name)
            else:
                st.error("Please provide a valid model name.")

if __name__ == "__main__":
    main()
