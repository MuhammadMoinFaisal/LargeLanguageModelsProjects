from langchain.llms import OpenAI
import streamlit as st
import os
from openaiapikey import openai_key
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain



os.environ['OPENAI_API_KEY'] = openai_key

st.title('Book Summary')

input_text = st.text_input("Search the book you want")
#Prompt Template
first_input_prompt = PromptTemplate(input_variables = ['name'],
                                    template="Provide me a summary of the book {name}"
                                    )
#Open AI LLMS
llm = OpenAI(temperature=0.8)

#LLM Chain
chain1  = LLMChain(llm=llm, prompt = first_input_prompt, verbose=True, output_key = 'summaryofbook')
#Prompt Template

second_input_prompt = PromptTemplate(input_variables = ['summaryofbook'],
                                    template="when was the {summaryofbook} published"
                                    )
#LLM Chain

chain2  = LLMChain(llm=llm, prompt = second_input_prompt, verbose=True, output_key = 'bookpublishdate')


#Prompt Template

third_input_prompt = PromptTemplate(input_variables = ['summaryofbook'],
                                    template="Please tell me about the authors of the {summaryofbook}"
                                    )
#LLM Chain

chain3  = LLMChain(llm=llm, prompt = third_input_prompt, verbose=True, output_key = 'authorsofthebook')

parent_chain = SequentialChain(chains = [chain1, chain2, chain3], input_variables = ['name'], output_variables = ['summaryofbook', 'bookpublishdate','authorsofthebook'], verbose = True)

if input_text:
    st.write(parent_chain({'name':input_text}))