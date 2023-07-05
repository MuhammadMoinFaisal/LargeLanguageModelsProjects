from langchain.llms import OpenAI
import streamlit as st
import os
from openaiapikey import openai_key

os.environ['OPENAI_API_KEY'] = openai_key

st.title('Lang Chain Demo with Open AI')

input_text = st.text_input("Search the topic you want")


#Open AI LLMS
llm = OpenAI(temperature=0.8)

if input_text:
    st.write(llm(input_text))