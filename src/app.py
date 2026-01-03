import streamlit as st
from transformers import AutoTokenizer

# Header
st.title("🔢 Token Counter App")
st.markdown(
    "Easily count tokens for your text using Hugging Face tokenizers! "
    "Select your model, input your text, and get insights into tokenization."
)

# Sidebar - Model Selector
st.sidebar.header("Model Settings")
default_model = "bert-base-uncased"  # Default model
model_name = st.sidebar.text_input("Model Name", default_model)

# Tokenizer Loading
@st.cache_resource
def load_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

try:
    tokenizer = load_tokenizer(model_name)
    st.sidebar.success(f"Model '{model_name}' loaded successfully! 🎉")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# Input Area
st.subheader("Input your text:")
input_text = st.text_area("Enter text here:", "", height=150)

# Tokenization Logic
if st.button("Count Tokens"):
    if not input_text.strip():
        st.warning("Please input some text before counting tokens.")
    else:
        try:
            # Tokenize and count tokens
            tokenized = tokenizer(input_text, return_tensors="pt", truncation=True)
            token_count = len(tokenized["input_ids"][0])

            # Display Results
            st.success("Tokenization Complete!")
            st.markdown(f"**Token Count:** `{token_count}`")
            st.text_area(
                "Tokenized Output (IDs):",
                tokenizer.convert_ids_to_tokens(tokenized["input_ids"][0]),
                height=200,
            )
        except Exception as e:
            st.error(f"An error occurred during tokenization: {e}")

# Footer
st.markdown("---")
st.markdown(
    "Developed with ❤️ "
    "Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)."
)
