import streamlit as st
import google.generativeai as genai
import time
import random

st.set_page_config(
    page_title="Chat with Gemini Pro",
    page_icon="🔥"
)

st.title("Chat with Gemini Pro")
st.caption("A Chatbot Powered by Google Gemini Pro")

if "app_key" not in st.session_state:
    app_key = st.text_input("Please enter your Gemini API Key", type='password')
    if app_key:
        st.session_state.app_key = app_key

if "history" not in st.session_state:
    st.session_state.history = []

try:
    genai.configure(api_key = st.session_state.app_key)
except AttributeError as e:
    st.warning("Please Put Your Gemini API Key First")

model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat(history = st.session_state.history)

with st.sidebar:
    if st.button("Clear Chat Window", use_container_width=True, type="primary"):
        st.session_state.history = []
        st.rerun()

for message in chat.history:
    role ="assistant" if message.role == 'model' else message.role
    with st.chat_message(role):
        st.markdown(message.parts[0].text)

if "app_key" in st.session_state:
    if prompt := st.chat_input(""):
        prompt = prompt.replace('\n', ' \n')
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            try:
                full_response = ""
                for chunk in chat.send_message(prompt, stream=True):
                    word_count = 0
                    random_int = random.randint(5,10)
                    for word in chunk.text:
                        full_response+=word
                        word_count+=1
                        if word_count == random_int:
                            time.sleep(0.05)
                            message_placeholder.markdown(full_response + "_")
                            word_count = 0
                            random_int = random.randint(5,10)
                message_placeholder.markdown(full_response)
            except genai.types.generation_types.BlockedPromptException as e:
                st.exception(e)
            except Exception as e:
                st.exception(e)
            st.session_state.history = chat.history

