import streamlit as st
from transformers import pipeline
import torch

# Load the text-to-text generation pipeline
pipe = pipeline("text2text-generation", model="samanjoy2/bnpunct_banglat5_seq2seq_finetuned", device=0)

st.title("Bangla Punctutation Restoration")

# User input for text generation
input_text = st.text_area("Enter Bangla text for restoration:", max_chars=450)

if st.button("Restore Punctuations"):
    if input_text:
        # Generate text using the pipeline
        generated_text = pipe(input_text, max_length=512, batch_size=1)[0]['generated_text']

        # Display the generated text
        st.subheader("Generated Text:")
        st.write(generated_text)
    else:
        st.warning("Please enter text for generation.")
