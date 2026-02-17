import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="OCR + RAG Assistant", layout="wide")

st.title("🤖 OCR + Vector AI Assistant")

# -----------------------------
# SIDEBAR - PDF Upload
# -----------------------------

st.sidebar.header("📄 Upload PDF")

uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

if uploaded_file:

    if st.sidebar.button("Upload & Store"):

        with st.spinner("Uploading and processing PDF..."):

            response = requests.post(
                f"{API_URL}/upload-pdf",
                files={"file": uploaded_file}
            )

        if response.status_code == 200:
            data = response.json()
            st.sidebar.success(
                f"✅ Stored {data['chunks_stored']} chunks successfully!"
            )
        else:
            st.sidebar.error("❌ Upload failed")


# -----------------------------
# CHAT SECTION
# -----------------------------

st.subheader("💬 Chat with Your Documents")

# Store messages in session
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask something about your uploaded PDF...")

if user_input:

    # Add user message
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Call /chat endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            response = requests.get(
                f"{API_URL}/chats",
                json={"message": user_input}
            )

            if response.status_code == 200:
                reply = response.text
            else:
                reply = "❌ Server error"
        reply = reply.replace("\\n", "\n")
        st.markdown(reply)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )
