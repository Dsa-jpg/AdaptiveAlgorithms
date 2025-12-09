import streamlit as st
from app.core.model import HONU, MLP

def select_model():
    st.title("2) Model Selection")


    model_type = st.selectbox("Select Model:", ["HONU", "MLP"])
    st.session_state['model_type'] = model_type


    if model_type == "HONU":
        degree = st.slider("Polynomial degree (1-3):", min_value=1, max_value=3, value=1)
        st.session_state['degree'] = degree

        st.session_state['model'] = HONU(degree)
        st.write(f"HONU model with degree {degree} initialized.")
    elif model_type == "MLP":
        layers = st.slider("Number of hidden layers:", min_value=1, max_value=5, value=2)
        neurons = st.slider("Neurons per hidden layer:", min_value=1, max_value=50, value=10)
        activation = st.selectbox("Activation function:", ["tanh", "relu", "sigmoid"])
        st.session_state['layers'] = layers
        st.session_state['neurons'] = neurons
        st.session_state['activation'] = activation

        st.session_state['model'] = MLP(layers, neurons, activation)
        st.write(f"MLP model with {layers} layers, {neurons} neurons, activation {activation} initialized.")

def main():
    select_model()
    st.info("Model parameters are stored in session_state and ready for training.")

if __name__ == "__main__":
    main()
