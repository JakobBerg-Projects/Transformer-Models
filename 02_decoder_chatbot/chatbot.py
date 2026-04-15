import torch
import streamlit as st
from tokenizers import Tokenizer

from config import config
from model import TransformerModel
from inference import sample_sequence, tokenize_input, decode_output

# Hack to prevent streamlit error 
torch.classes.__path__ = []

@st.cache_resource
def load_model(config):
    model = TransformerModel(config)
    model = model.to(config.device)
    model.load_state_dict(torch.load(config.model_filename, weights_only=True, map_location=config.device))
    return model

@st.cache_resource
def load_tokenizer(config):
    return Tokenizer.from_file(config.tokenizer_filename)

# A simple streamlit chatbot
st.title("Cursed Chatbot")

# Load model and tokenizer
status_text = st.text("Loading model...")
model = load_model(config)
status_text.text("Loading tokenizer...")
tokenizer = load_tokenizer(config)
status_text.empty()

sep_id = tokenizer.token_to_id(config.sep_token)
end_id = tokenizer.token_to_id(config.end_token)

with st.sidebar:
    strategy = st.selectbox("Sampling strategy", ["greedy", "top-p"], index=1)
    temperature = st.slider("Temperature", 0.1, 2.0, 0.7)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.95)
    clear_chat = st.button("Clear chat history")

if clear_chat:
    st.session_state.messages = []

# Based on code from the Streamlit docs

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# React to user input
if prompt := st.chat_input("Type your question...", max_chars=100):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Sample answer from model 
    input_sequence = tokenize_input(tokenizer, prompt, sep_id)
    answer = sample_sequence(input_sequence, model, strategy, config.max_len, config.device, end_id, p=top_p, temperature=temperature)
    answer_text = decode_output(tokenizer, answer)

    st.chat_message("assistant").write(answer_text)
    st.session_state.messages.append({"role": "assistant", "content": answer_text})


