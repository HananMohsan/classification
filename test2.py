import streamlit as st
from transformers import pipeline

# Load pre-trained text classification model from Hugging Face
classifier = pipeline("sentiment-analysis")

# Streamlit web app
def main():
    st.title("Text Classification App")

    # User input: Text
    text_input = st.text_area("Enter your text:")

    if st.button("Get Sentiment"):
        if text_input:
            # Make prediction
            prediction = classifier(text_input)

            # Display result
            st.subheader("Sentiment Prediction:")
            st.write(f"Text: {text_input}")
            st.write(f"Sentiment: {prediction[0]['label']} with confidence: {prediction[0]['score']:.4f}")
        else:
            st.warning("Please enter some text.")

if __name__ == "__main__":
    main()
