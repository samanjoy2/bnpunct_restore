import streamlit as st
from transformers import pipeline
import torch

# Load the text-to-text generation pipeline
pipe = pipeline("text2text-generation", model="samanjoy2/bnpunct_banglat5_seq2seq_finetuned")

st.title("Bangla Punctutation Restoration")

# User input for text generation
input_text = st.text_area("Enter Bangla text for restoration:", max_chars=500)

def generate_text(input_text):
    # Use the pipeline to generate text
    generated_text = pipe(input_text, max_length=512, batch_size=64)[0]['generated_text']
    return generated_text

if st.button("Restore Punctuations"):
    if input_text:
        # Generate text using the pipeline
        generated_text = generate_text(input_text)

        # Display the generated text
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter text for generation.")
