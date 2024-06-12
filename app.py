import langchain
from pydantic import SecretStr
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder

from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from htmlTemplates import css, bot_template, user_template

State = st.session_state

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
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceEmbeddings(model_name="../acge_text_embedding")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_retriever_chain(vectorstore: FAISS):
    llm: ChatOpenAI = State.llm
    retriever = vectorstore.as_retriever()
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    prompt_search_query = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)
    return retriever_chain

def get_qa_chain():
    llm: ChatOpenAI = State.llm
    qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
    prompt_get_answer = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human","{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, prompt_get_answer)
    return qa_chain

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in State.store:
        State.store[session_id] = ChatMessageHistory()
    return State.store[session_id]

def handle_userinput(user_question):
    if State.conversation is None:
        st.write(
            bot_template.replace("{{MSG}}", "Please first input some PDFs and then click the process button. It may also fail to load the model."),
            unsafe_allow_html=True
        )
        return
    sid = "sid"
    State.conversation.invoke(
        {"input": user_question},
        config={
            "configurable": {"session_id": sid}
        }
    )

    message: BaseMessage
    for message in State.store[sid].messages:
        if isinstance(message, AIMessage):
            st.write(bot_template.replace(
                "{{MSG}}", str(message.content)), unsafe_allow_html=True)
        elif isinstance(message, HumanMessage):
            st.write(user_template.replace(
                "{{MSG}}", str(message.content)), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in State:
        State.conversation = None
    if "chat_history" not in State:
        State.chat_history = None
    if "llm" not in State:
        State.llm = ChatOpenAI(model="chatglm3-6b", base_url="http://127.0.0.1:8000/v1", api_key="test")
        # State.llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    if "store" not in State:
        State.store = {}

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
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
                if raw_text == "":
                    st.write(
                        "Please input some PDFs first."
                    )
                    return

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create retriever, qa, conversation chain
                history_aware_chain = get_retriever_chain(vectorstore)
                qa_chain = get_qa_chain()

                rag_chain = create_retrieval_chain(history_aware_chain, qa_chain)
                
                State.conversation = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )

                st.write("Process finished!")


if __name__ == '__main__':
    main()
