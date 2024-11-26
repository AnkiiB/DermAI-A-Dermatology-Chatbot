import os
import streamlit as st
import requests
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
import tiktoken
from gtts import gTTS
import speech_recognition as sr
import openai

# OpenAI API Key
openai_api_key = "OpenAI_Key"
FAISS_INDEX_PATH = "faiss_index"

# Initialize conversation history in Streamlit session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = split_text_into_chunks(docs)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        vectorstore.save_local(FAISS_INDEX_PATH)

    retriever = vectorstore.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_api_key),
        retriever=retriever
    )
    return qa_chain

# Load PDF content and prepare LangChain model
pdf_files = ['Dermatology_An_Illustrated_Colour_Textbo.pdf']
text_content = extract_text_from_pdfs(pdf_files)
qa_chain = create_langchain_model(text_content)

# Streamlit UI setup
st.set_page_config(page_title="Skin Disease Expert System", page_icon=":microscope:")
st.title('Skin Disease Expert System')

# Sidebar for displaying conversation
st.sidebar.title("Chat History")
if st.session_state.chat_history:
    for i, (user, bot) in enumerate(st.session_state.chat_history, 1):
        st.sidebar.markdown(f"*You {i}:* {user}")
        st.sidebar.markdown(f"*Chatbot {i}:* {bot}")

# Function to handle voice input
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for your query... Speak now!")
        try:
            audio_data = recognizer.listen(source, timeout=5)
            query = recognizer.recognize_google(audio_data)
            st.success(f"You said: {query}")
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand your speech. Please try again.")
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
    return None

# Main chat interface
st.subheader("Chat with the Skin Disease Expert")
user_query = st.text_input("Ask any question about skin diseases (or use voice):", key="user_query", placeholder="Type your question here...")
voice_input = st.button("Use Voice Input")

if voice_input:
    user_query = speech_to_text()

if user_query:
    try:
        response = qa_chain.invoke({"question": user_query, "chat_history": st.session_state.chat_history})
        answer = response["answer"]

        # Append to chat history
        st.session_state.chat_history.append((user_query, answer))

        # Generate and play audio response using a temporary file
        import tempfile

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                tts = gTTS(answer)
                tts.save(temp_audio_file.name)

                # Play the chatbot's response as audio
                with open(temp_audio_file.name, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Error generating or playing audio: {e}")

        # Display chat in real time
        for i, (user, bot) in enumerate(st.session_state.chat_history, 1):
            st.markdown(f"*You {i}:* {user}")
            st.markdown(f"*Chatbot {i}:* {bot}")

    except Exception as e:
        st.error(f"An error occurred: {e}")

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
            st.write(f"*Diagnosis:* {result['class_name']}")
            st.write(f"Confidence: {result['class_probability']:.2f}%")
            st.write(f"*Description:* {result['description']}")
            st.write(f"*Disclaimer:* Please consult a professional.")
        else:
            st.error("Failed to process image. Please try again.")
    except Exception as e:
        st.error(f"Failed to process image: {e}")
