Here is a clean, simple, ready-to-copy **README.txt** with your exact run commands included.

---

OCR Agent System
Author: Shovan Saha

---

Project Overview

This project is an OCR-based AI Agent system that allows users to:

* Upload PDF documents
* Extract text using OCR
* Query the extracted content using a conversational AI interface

The system is built using:

* FastAPI (Backend)
* Streamlit (Frontend)
* Agent-based RAG pipeline

---

Architecture

User
-> Streamlit UI
-> FastAPI Backend
- /upload (PDF Upload Endpoint)
- /chat (Chat Endpoint)
-> Agent (RAG + LLM)
-> Response

---

How to Run the Project

Make sure your virtual environment is activated.

Start Backend:

uvicorn app.orchestration.backend.app:server --reload

Start Frontend:

streamlit run app/frontend/streamlit/app.py

Important:

* Run backend first
* Then run Streamlit
* Open the Streamlit URL in your browser (usually [http://localhost:8501](http://localhost:8501))

---

API Endpoints

POST /upload
Uploads a PDF and processes OCR.

POST /chat
Sends a query to the agent and returns an AI-generated response.

---

Current Limitations

Tracing / Observability system is NOT implemented.

There is:

* No request tracing
* No performance monitoring
* No token usage tracking
* No LLM call logging

This project focuses on core OCR -> RAG -> Agent functionality.

---

Future Improvements

* Add tracing and monitoring
* Add evaluation pipeline
* Improve retrieval quality
* Add caching layer
* Production deployment setup

---

Developed by
Shovan Saha

---


Production Date - From 20/01/2026 -- To  17/02/2026