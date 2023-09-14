from langchain.llms import CTransformers
from langchain.chains import LLMChain
from langchain import PromptTemplate

prompt_template = """
You are an AI coding assistant and your task to solve the coding problems, and return coding snippets based on the
Query: {query}

You just return helpful answer and nothing else
Helpful Answer: 
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['query'])


llm = CTransformers(model = "model/codellama-7b-instruct.ggmlv3.Q4_0.bin",
                    model_type = "llama",
                    max_new_tokens=512,
                    temperature=0.2
                    )

llm_chain = LLMChain(prompt=prompt, llm=llm)


llm_response = llm_chain.run({"query": "Write a python code to load a CSV file using pandas library"})

print(llm_response)