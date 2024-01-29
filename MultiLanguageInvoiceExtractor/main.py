#Import All the Required Libraries
import streamlit as st
import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#Load the Gemini Pro Vision Model
model = genai.GenerativeModel('gemini-pro-vision')

def get_gemini_respone(input_prompt, image, user_input_prompt):
    response = model.generate_content([input_prompt, image[0], user_input_prompt])
    return response.text

def input_image_bytes(uploaded_file):
    if uploaded_file is not None:
        #Convert the Uploaded File into bytes
        bytes_data = uploaded_file.getvalue()
        image_parts = [
            {
                "mime_type": uploaded_file.type,
                "data": bytes_data
            }
        ]
        return  image_parts
    else:
        raise FileNotFoundError("No File Uploaded")

# Initialize the Streamlit App
st.set_page_config(page_title="MultiLanguage Invoice Extractor")
input_prompt = """
You are an expert in understanding invoices. Please try to answer the question using the information from the uploaded
invoice.
"""
user_input_prompt = st.text_input("User Input Prompt", key="input")
upload_image_file = st.file_uploader("Choice an Image of the Invoice", type=["jpg", "jpeg", "png"])
if upload_image_file is not None:
    image = Image.open(upload_image_file)
    st.image(image, caption = "Uploaded Image", use_column_width=True)

submit = st.button("Find the Answer from the Invoice")
if submit:
    input_image_data = input_image_bytes(upload_image_file)
    response = get_gemini_respone(input_prompt, input_image_data, user_input_prompt)
    st.subheader("Response")
    st.write(response)