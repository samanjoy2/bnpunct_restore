import streamlit as st
from transformers import pipeline

# Load the text-to-text generation pipeline
pipe = pipeline("text2text-generation", model="samanjoy2/bnpunct_banglat5_seq2seq_finetuned", device=0)

def main():
    st.title("Bangla Punctutation Restoration")

    # User input for text generation
    input_text = st.text_area("Enter Bangla text for restoration:", max_chars=500)

    if st.button("Restore Punctuations"):
        if input_text:
            # Generate text using the pipeline
            generated_text = generate_text(input_text)

            # Display the generated text
            st.subheader("Generated Text:")
            st.write(generated_text)
        else:
            st.warning("Please enter text for generation.")

def generate_text(input_text):
    # Use the pipeline to generate text
    generated_text = pipe(input_text)[0]['generated_text']
    return generated_text

if __name__ == "__main__":
    main()
