import streamlit as st
import requests

# 1. Set up the Streamlit page
st.set_page_config(page_title="MediAssist Chatbot", page_icon="🏥")
st.title("🏥 MediAssist Domain-Specific Assistant")
st.markdown("Ask me anything about anatomy, physiology, pharmacology, pathology, or clinical medicine!")

# Define the FastAPI backend URL
API_URL = "http://127.0.0.1:8000/generate"

# 2. Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Handle new user input
if prompt := st.chat_input("Ask a medical question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                # Send the request to the FastAPI backend
                response = requests.post(API_URL, json={"message": prompt})
                response.raise_for_status()  # Check for HTTP errors
                
                # Extract the reply
                reply = response.json().get("reply", "Error: Empty response")
                
            except requests.exceptions.RequestException as e:
                reply = f"**Backend connection error:** Make sure the FastAPI server is running. ({e})"
            
            # Display the result
            st.markdown(reply)
            
    # Add assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": reply})