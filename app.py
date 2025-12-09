import streamlit as st
import PyPDF2 
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
import textwrap
import time
from gtts import gTTS
import io

from test_inference import chatbot 
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    huggingface_api_key = st.text_input("Huggingface API Key", key="huggingface_api_key", type="password")
    "[Get a Huggingface API key](https://huggingface.co/settings/tokens)"

if "history" not in st.session_state:
    st.session_state.history = [] 

for role, message in st.session_state.history:
    st.chat_message(role).write(message) 

if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

if uploaded_file is not None:
   
    if st.session_state.text_chunks is None:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
 
        st.session_state.text_chunks = textwrap.wrap(text, width=1000, break_long_words=False)
else:
    st.warning("Please upload a file to start chatting.")


query = st.chat_input("Ask a question about the document")

if query:
    if st.session_state.text_chunks is None:
        st.error("Please upload and process a file first.")
    else:
        st.chat_message("user").write(query)
        st.session_state.history.append(("user", query))

        
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

       
        chunk_embeddings = embed_model.encode(st.session_state.text_chunks, convert_to_tensor=False)

       
        embedding_dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(chunk_embeddings).astype("float32"))
       
        query_embedding = embed_model.encode([query])
        
        distances, indices = index.search(np.array(query_embedding).astype("float32"), k=4)       
        retrieved_texts = [st.session_state.text_chunks[i] for i in indices[0]]
        context = " ".join(retrieved_texts)
        prompt = f"""
        You are a helpful assistant. Use ONLY the context provided below to answer the question. 
        If the answer is not in the context, respond with: 'I could not find an answer in the provided document.' 
        
        Context: {context}
        
        Question: {query}
        """
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            if not huggingface_api_key:
                assistant_response = "Please enter your Huggingface API key in the sidebar."
            else:
                try:
                    assistant_response = chatbot(prompt, huggingface_api_key)
                except Exception as e:
                    assistant_response = f"An error occurred: {e}"
            
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
           
            st.divider()
            try:
                tts = gTTS(full_response, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)
                st.audio(audio_buffer, format='audio/mp3')
            except Exception as e:
                st.error(f"Failed to generate audio: {e}")           
            st.session_state.history.append(("assistant", full_response))
