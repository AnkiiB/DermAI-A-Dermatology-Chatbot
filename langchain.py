import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import tiktoken
import openai

# Load environment variables
openai_api_key = os.getenv('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError(
        "No OpenAI API key found. Set your OpenAI API key in the environment variables or use st.secrets for sensitive data.")

# Define the file path for saving the FAISS index
openai_api_key = "your-api-key'
# Define the file path for saving the FAISS index
FAISS_INDEX_PATH = "faiss_index"


# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_paths):
    text_data = []
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        text_data.extend([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return " ".join(text_data)


# Function to split text into smaller chunks based on token limits
def split_text_into_chunks(text, max_tokens=2000):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(tokenizer.decode(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(tokenizer.decode(current_chunk))

    return chunks


# Function to create or load the LangChain model
def create_langchain_model(docs):
    if os.path.exists(FAISS_INDEX_PATH):
        # Load the existing FAISS index
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        # Create a new FAISS index from the document chunks
        chunks = split_text_into_chunks(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        # Save the FAISS index to disk
        vectorstore.save_local(FAISS_INDEX_PATH)

    # Create the QA chain using the loaded or newly created FAISS index
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4", openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    return qa_chain


# Load PDF content and prepare LangChain model
pdf_files = ['Dermatology_An_Illustrated_Colour_Textbo.pdf']
text_content = extract_text_from_pdfs(pdf_files)
qa_chain = create_langchain_model(text_content)

# Streamlit UI setup
st.set_page_config(page_title="Skin Disease Expert System", page_icon=":microscope:")
st.title('Skin Disease Expert System')

# Image uploader for processing images via API
uploaded_image = st.file_uploader("Upload an image of the skin condition:", type=["png", "jpg", "jpeg"])
if uploaded_image is not None:
    try:
        # Sending the uploaded image to the external API
        bytes_data = uploaded_image.getvalue()
        response = requests.post(
            'http://127.0.0.1:5000/predict',
            files={'pic': (uploaded_image.name, bytes_data, 'image/jpeg')}
        )

        # Checking the API response
        if response.status_code == 200:
            result = response.json()
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            st.write(f"**Diagnosis:** {result['class_name']}")
            st.write(f"Confidence: {result['class_probability']:.2f}%")
            st.write(f"**Description:** {result['description']}")
            st.write(f"**Disclaimer:** Please consult a professional.")
        else:
            st.error("Failed to process image. Please try again.")
    except Exception as e:
        st.error(f"Failed to process image: {e}")

# Text query input for handling questions using LangChain
user_query = st.text_input("Ask any question about skin diseases:")
if user_query:
    try:
        # Using the LangChain model to answer the query
        response = qa_chain.invoke(user_query)
        st.write("**Answer:**",
                 response if response else "I'm sorry, I do not have enough information to answer this question.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
