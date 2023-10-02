import os
from langchain.document_loaders import PyPDFLoader

os.environ["OPENAI_API_KEY"] = "sk-"

pdf_loader = PyPDFLoader('./docs/RachelGreenCV.pdf')
documents = pdf_loader.load()

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# we are specifying that OpenAI is the LLM that we want to use in our chain
chain = load_qa_chain(llm=OpenAI())
query = 'Who is the CV about?'
response = chain.run(input_documents=documents, question=query)
print(response)