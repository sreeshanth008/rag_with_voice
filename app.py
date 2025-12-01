# import streamlit as st
# import PyPDF2 
# from sentence_transformers import SentenceTransformer 
# import faiss
# import numpy as np
# import textwrap
# import time
# from elevenlabs.client import ElevenLabs 
# from gtts import gTTS
# import io
# # NEW: Import ElevenLabs client

# # Assuming 'chatbot' is a function in a file named 'test_inference.py'
# # from test_inference import chatbot 
# # Since I don't have your file, I'll create a dummy function for demonstration
# from test_inference import chatbot 
# # --- Sidebar for Inputs ---
# with st.sidebar:
#     st.header("Configuration")
#     uploaded_file = st.file_uploader("Upload a file", type=["pdf","txt"])
#     huggingface_api_key = st.text_input("Huggingface API Key", key="huggingface_api_key", type="password")
    
#     # NEW: Added input for ElevenLabs API key
#     elevenlabs_api_key = st.text_input("ElevenLabs API Key", key="elevenlabs_api_key", type="password")

#     "[Get a Huggingface API key](https://huggingface.co/settings/tokens)"
#     "[Get an ElevenLabs API key](https://elevenlabs.io/subscription)"

# # --- Chat History Initialization ---
# if "history" not in st.session_state:
#     st.session_state.history = [] 

# for role, message in st.session_state.history:
#     st.chat_message(role).write(message) 

# # Initialize session state for text chunks
# if "text_chunks" not in st.session_state:
#     st.session_state.text_chunks = None

# # --- File Processing ---
# if uploaded_file is not None:
#     # Process file only once
#     if st.session_state.text_chunks is None:
       
#         if uploaded_file.type == "text/plain":
#             text = uploaded_file.read().decode("utf-8")
#         else:
#             pdf_reader = PyPDF2.PdfReader(uploaded_file)
#             text = ""
#             for page in pdf_reader.pages:
#                 text += page.extract_text() or ""
        
#         # FIXED: Use a more robust chunking strategy
#         st.session_state.text_chunks = textwrap.wrap(text, width=1000, break_long_words=False)
       
# else:
#     st.warning("Please upload a file to start chatting.")

# # --- Query Processing and RAG Implementation ---
# query = st.chat_input("Ask a question about the document")
# if query:
#     if st.session_state.text_chunks is None:
#         st.error("Please upload and process a file first.")
#     else:
#         st.chat_message("user").write(query)
#         st.session_state.history.append(("user", query))

#         # --- FIXED: Correct RAG and FAISS Logic ---
#         # This whole block was incorrect and has been fixed.
#         # 1. Initialize the embedding model
#         embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        
#         # 2. Encode the text chunks (not the whole document at once)
#         chunk_embeddings = embed_model.encode(st.session_state.text_chunks, convert_to_tensor=False)
        
#         # 3. Create the FAISS index
#         embedding_dim = chunk_embeddings.shape[1]
#         index = faiss.IndexFlatL2(embedding_dim)
#         index.add(np.array(chunk_embeddings).astype("float32"))
        
#         # 4. Embed the user's query
#         query_embedding = embed_model.encode([query])
        
#         # 5. Search the index for the most relevant chunks
#         distances, indices = index.search(np.array(query_embedding).astype("float32"), k=4)
        
#         # 6. Retrieve the context and build the prompt
#         retrieved_texts = [st.session_state.text_chunks[i] for i in indices[0]]
#         context = " ".join(retrieved_texts)
#         prompt = f"""You are a helpful assistant. Use ONLY the context provided below to answer the question. If the answer is not in the context, you must respond with: 'I could not find an answer in the provided document.' Do not try to make up an answer.
        
#         Context: {context}
        
#         Question: {query}"""
        
#         # --- Response Generation and Display ---
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             full_response = ""
            
#             if not huggingface_api_key:
#                 assistant_response = "Please enter your Huggingface API key in the sidebar."
#             else:
#                 try:
#                     assistant_response = chatbot(prompt, huggingface_api_key)
#                 except Exception as e:
#                     assistant_response = f"An error occurred: {e}"

#             # Simulate typing effect for the text response
#             for chunk in assistant_response.split():
#                 full_response += chunk + " "
#                 time.sleep(0.05)
#                 message_placeholder.markdown(full_response + "▌")
#             message_placeholder.markdown(full_response)
            
#             # --- NEW: Audio Generation and Playback ---
#             # --- NEW: Audio Generation and Playback (FIXED) ---
#             # --- TEXT TO SPEECH USING gTTS (Google Text-to-Speech) ---

# st.divider()
# try:
#     # Create a TTS object
#     tts = gTTS(full_response, lang='en', slow=False)

#     # Save the audio to a bytes buffer
#     audio_buffer = io.BytesIO()
#     tts.write_to_fp(audio_buffer)
#     audio_buffer.seek(0)

#     # Play audio in Streamlit
#     st.audio(audio_buffer, format='audio/mp3')

# except Exception as e:
#     st.error(f"Failed to generate audio: {e}")

#             # if elevenlabs_api_key:
#             #     st.divider() 
#             #     try:
#             #         client = ElevenLabs(api_key=elevenlabs_api_key)
                    
#             #         # 1. Generate the audio stream (which is a generator)
#             #         audio_stream = client.text_to_speech.convert(
#             #             text=full_response,
#             #             voice_id="pNInz6obpgDQGcFmaJgB",
#             #             model_id="eleven_multilingual_v2"
#             #         )
                    
#             #         # 2. Join all the chunks from the generator into a single bytes object
#             #         # This is the crucial fix.
#             #         audio_bytes = b"".join(audio_stream)
                    
#             #         # 3. Pass the complete audio data to st.audio
#             #         st.audio(audio_bytes, format="audio/mpeg")

#             #     except Exception as e:
#                     st.error(f"Failed to generate audio: {e}")
#                 st.session_state.history.append(("assistant", full_response))
import streamlit as st
import PyPDF2 
from sentence_transformers import SentenceTransformer 
import faiss
import numpy as np
import textwrap
import time
from gtts import gTTS
import io

# Assuming 'chatbot' is a function in a file named 'test_inference.py'
# You can import your actual chatbot function here
from test_inference import chatbot 

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    uploaded_file = st.file_uploader("Upload a file", type=["pdf", "txt"])
    huggingface_api_key = st.text_input("Huggingface API Key", key="huggingface_api_key", type="password")
    "[Get a Huggingface API key](https://huggingface.co/settings/tokens)"

# --- Chat History Initialization ---
if "history" not in st.session_state:
    st.session_state.history = [] 

for role, message in st.session_state.history:
    st.chat_message(role).write(message) 

# Initialize session state for text chunks
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# --- File Processing ---
if uploaded_file is not None:
    # Process file only once
    if st.session_state.text_chunks is None:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        else:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        
        # Use a more robust chunking strategy
        st.session_state.text_chunks = textwrap.wrap(text, width=1000, break_long_words=False)
else:
    st.warning("Please upload a file to start chatting.")

# --- Query Processing and RAG Implementation ---
query = st.chat_input("Ask a question about the document")

if query:
    if st.session_state.text_chunks is None:
        st.error("Please upload and process a file first.")
    else:
        st.chat_message("user").write(query)
        st.session_state.history.append(("user", query))

        # --- RAG with FAISS ---
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Encode text chunks
        chunk_embeddings = embed_model.encode(st.session_state.text_chunks, convert_to_tensor=False)

        # Create FAISS index
        embedding_dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(np.array(chunk_embeddings).astype("float32"))

        # Encode user query
        query_embedding = embed_model.encode([query])

        # Search for most relevant chunks
        distances, indices = index.search(np.array(query_embedding).astype("float32"), k=4)

        # Retrieve context
        retrieved_texts = [st.session_state.text_chunks[i] for i in indices[0]]
        context = " ".join(retrieved_texts)
        prompt = f"""
        You are a helpful assistant. Use ONLY the context provided below to answer the question. 
        If the answer is not in the context, respond with: 'I could not find an answer in the provided document.' 
        
        Context: {context}
        
        Question: {query}
        """

        # --- Response Generation ---
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

            # Simulate typing effect
            for chunk in assistant_response.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # --- TEXT TO SPEECH USING gTTS ---
            st.divider()
            try:
                tts = gTTS(full_response, lang='en', slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                st.audio(audio_buffer, format='audio/mp3')
            except Exception as e:
                st.error(f"Failed to generate audio: {e}")

            # Save chat to history
            st.session_state.history.append(("assistant", full_response))
