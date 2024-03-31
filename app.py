import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain import LLMChain
import shutil

from audio_recorder_streamlit import audio_recorder

import torch
from TTS.api import TTS

import whisper


from speech_to_text import speech_to_text
from text_to_speech import text_to_speech
from local_model import Chat, CustomLLM

# embedding模型
import os


if "init" not in st.session_state:
    st.session_state["init"] = True

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    if os.path.exists("audio"):
        shutil.rmtree("audio")
    os.mkdir("audio")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading tts model...")
    tts = TTS("tts_models/zh-CN/baker/tacotron2-DDC-GST").to(device)
    st.session_state["tts"] = tts
    print("finish loading")

    print("loading stt model...")
    st.session_state["stt"] = whisper.load_model("base")
    print("finish loading")

    print("loading llm model...")
    model = "/home/slx/CodeTangent_backup/text-generation-webui/models/Qwen1.5-14B-Chat"
    chat = Chat(model)
    llm = CustomLLM(client=chat,n=32000)
    st.session_state["llm"] = llm
    print("finish loading")


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name='/home/slx/embedding_models/acge_text_embedding')
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = st.session_state["llm"]
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("please upload some files")
        return
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            text_to_speech(st.session_state["tts"], message.content, i)

def main():
    load_dotenv()


    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    speech='(Please say something to the recorder)'
    audio_bytes = audio_recorder()
    if audio_bytes:
        speech = speech_to_text(st.session_state["stt"], audio_bytes)
        st.write("语音: ", speech)
    user_question = st.text_input("Ask a question about your documents:", placeholder=speech)
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
