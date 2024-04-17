

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except PdfReadError:
            st.error("Error reading PDF file. Please check if the file is properly formatted and try again.")
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")
    return text

def get_vtt_text(vtt_docs):
    text = ""
    for vtt in vtt_docs:
        text += vtt.read().decode("utf-8")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

prompt_options = {
    "Sample Prompt": """
    Generate detailed discussion items of the following meeting transcript do not include minutes of meeting and next steps.
    Do not use any previous information, Use the following format:

    Meeting Overview: <Provide the overview of the meeting>

    Discussion Item 1: <item_name_1>
    - <detailed_description_point_1>
    - <detailed_description_point_2>

    ..

    Discussion Item 2: <item_name_2>
    - <detailed_description_point_1>
    - <detailed_description_point_2>
    ..

    Discussion Item 3: <item_name_3>
    - <detailed_description_point_1>
    - <detailed_description_point_2>

    ..

    Action Item: <item_name_1>
    - <Action_point_1>
    - <Action_point_2>

    ###

    The input text will be appended here: """
}

def get_gemini_response(input, prompt):
    model = genai.GenerativeModel("gemini-pro")
    if prompt == "Sample Prompt":
        response = model.generate_content(prompt + input)
    else:
        response = model.generate_content(prompt + "\n" + input)
    return response.text

def main():
    st.set_page_config(layout="centered")
    st.title("Meeting Summarizer")
    st.header("Upload your meetings to get a summary")

    file_uploader = st.file_uploader("Upload your files", accept_multiple_files=True, type=["txt", "vtt", "pdf"])

    prompt_selection = st.selectbox("Select a prompt", list(prompt_options.keys()) + ["Custom"])

    # Add a new input field for custom prompt
    custom_prompt = st.text_input("Custom Prompt", "") if prompt_selection == "Custom" else None

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            pdf_docs = [file for file in file_uploader if file.name.endswith(".pdf")]
            txt_docs = [file for file in file_uploader if file.name.endswith(".txt")]
            vtt_docs = [file for file in file_uploader if file.name.endswith(".vtt")]

            raw_text = get_pdf_text(pdf_docs) + get_txt_text(txt_docs) + get_vtt_text(vtt_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done")

            # Use the custom prompt when "Custom" is selected
            selected_prompt = custom_prompt if prompt_selection == "Custom" else prompt_options[prompt_selection]

            summary = get_gemini_response(raw_text, selected_prompt)
            st.write(summary)

if __name__ == "__main__":
    main()

