#Import All the Required Libraries
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain_core.output_parsers import StrOutputParser

#Setup the Environment
load_dotenv()

#Initialize and Load the Llama 3.1 Model
llm = ChatGroq(model = "llama-3.1-70b-versatile")

#Extract Text from PDF Files
#file_path = "PDFs/2207.02696v1.pdf"
#reader = PdfReader(file_path)
#number_of_pages = len(reader.pages)
#print("Number of Pages", number_of_pages)
#page = reader.pages[14]
#print(page.extract_text())

#Create a streamlit application
st.set_page_config(page_title="PDF Sorter")
st.title("PDF Sorter")

#Upload Files
files = st.file_uploader("Upload Your Files", type = "pdf", accept_multiple_files= True)

if st.button("Organize PDFs"):
    with st.spinner("Working on PDFs"):
        for i, file in enumerate(files):
            #Read the first page of the document
            reader = PdfReader(file)
            number_of_pages = len(reader.pages)
            first_page = reader.pages[0]
            raw_text = first_page.extract_text()
            #st.write(raw_text)
            #Output Format
            output_format = 'title - keyword - keyword - keyword'
            #Input Prompt
            prompt = f"""
                     Below is the text of a research paper. Using the text, generate the name of the paper as well as 
                     two or three keywords. Give raw text as the output. Please make sure to follow the following
                     output format: {output_format}.
                     Here is the raw text of the research paper: {raw_text}
                     """

            result = llm.invoke(prompt)
            parser = StrOutputParser()
            response = parser.invoke(result)
            response_cleaned_text  = ''.join(c for c in response if c.isalnum() or c in [' ', '-', '_'])
            st.subheader(f"PDF: {i + 1}")
            text_area_placeholder = st.markdown("", unsafe_allow_html = True)
            if text_area_placeholder:
                text_area_placeholder.markdown(response_cleaned_text, unsafe_allow_html = True)

            #Save the files
            output_folder = "Organized"
            os.makedirs(output_folder, exist_ok= True)
            new_file_path = f"{output_folder}/{response_cleaned_text}.pdf"

            #Write the uploaded file to new location
            with open(new_file_path, "wb") as f:
                f.write(file.getbuffer())
