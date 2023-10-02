import gradio as gr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
import os
import fitz
from PIL import Image

with gr.Blocks() as demo:
    # Create a Gradio block

    with gr.Column():
        with gr.Row():
            with gr.Column(scale=8):
                api_key = gr.Textbox(
                    placeholder='Enter OpenAI API key',
                    label=None,
                    interactive=True
                )
            with gr.Column(scale=2):
                change_api_key = gr.Button('Change Key')

        with gr.Row():
            chatbot = gr.Chatbot(value=[], elem_id='chatbot', height=650)
            show_img = gr.Image(label='Upload PDF', tool='select', height=680)

    with gr.Row():
        with gr.Column(scale=7):
            txt = gr.Textbox(
                label=None,
                placeholder="Enter text and press enter"
            )

        with gr.Column(scale=1):
            submit_btn = gr.Button('Submit')

        with gr.Column(scale=1):
            btn = gr.UploadButton("📁 Upload a PDF", file_types=[".pdf"])