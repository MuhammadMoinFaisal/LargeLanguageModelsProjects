from flask import Flask, jsonify, request
from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import CTransformers
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from urllib.parse import unquote
from langchain import OpenAI

import os
os.environ["OPENAI_API_KEY"] = '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'

app=Flask(__name__)
#1. Extract Data From the Website
def extract_data_website(url):
    loader=SeleniumURLLoader([url])
    data=loader.load()
    text=""
    for page in data:
        text +=page.page_content + " "
        return text
#2. Generate a Summary of the Text
def split_text_chunks_and_summary_generator(text):
    text_splitter=CharacterTextSplitter(separator='\n',
                                        chunk_size=1000,
                                        chunk_overlap=20)
    text_chunks=text_splitter.split_text(text)
    print(len(text_chunks))

    #llm = CTransformers(model='models\llama-2-7b-chat.ggmlv3.q4_0.bin',
    #                    model_type='llama',
    #                    config={'max_new_tokens': 128,
    #                            'temperature': 0.01}
    #                    )
    llm = OpenAI()

    docs = [Document(page_content=t) for t in text_chunks]
    chain=load_summarize_chain(llm=llm, chain_type='map_reduce', verbose=True)
    summary = chain.run(docs)
    return summary


@app.route('/', methods=['GET', 'POST'])
def home():
    return "Summary Generator"

@app.route('/summary_generate', methods=['GET', 'POST'])
def summary_generator():
    encode_url=unquote(unquote(request.args.get('url')))
    if not encode_url:
        return jsonify({'error':'URL is required'}), 400
    text=extract_data_website(encode_url)
    #text_chunks=split_text_chunks(text)
    #print(len(text_chunks))
    summary=split_text_chunks_and_summary_generator(text)
    print("Here is the Complete Summary", summary)
    response= {
        'submitted_url': encode_url,
        'summary': summary
    }
    return jsonify(response)
if __name__ == '__main__':
    app.run(debug=True)

