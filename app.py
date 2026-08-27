import json
import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import PolicyRAG, chunk_pages, extract_pdf_pages

load_dotenv()

st.set_page_config(page_title="Policy2Course RAG", page_icon="📘", layout="wide")
st.title("📘 Policy2Course RAG")
st.caption("Upload a company policy PDF, index it with embeddings, and generate a grounded 5-module training course.")

with st.sidebar:
    st.header("Pipeline")
    st.write("PDF → text → chunks → embeddings → Chroma → retrieval → Gemini")
    st.code(f"LLM: {os.getenv('GEMINI_MODEL', 'gemini-3.7-flash')}\nEmbeddings: all-MiniLM-L6-v2\nVector DB: Chroma")

uploaded = st.file_uploader("Upload a company policy PDF", type=["pdf"])

if uploaded:
    with st.spinner("Extracting and indexing the policy..."):
        pages = extract_pdf_pages(uploaded.getvalue())
        chunks = chunk_pages(pages)
        rag = PolicyRAG()
        rag.index(chunks)

    st.success(f"Indexed {len(chunks)} chunks from {len(pages)} text-bearing pages.")
    tab1, tab2, tab3 = st.tabs(["Generate 5-module course", "Ask the policy", "Inspect retrieval"])

    with tab1:
        if st.button("Generate course", type="primary"):
            try:
                with st.spinner("Retrieving evidence and generating course..."):
                    st.session_state["course"] = rag.generate_course()
            except Exception as exc:
                st.error(str(exc))
        if "course" in st.session_state:
            course = st.session_state["course"]
            st.json(course)
            st.download_button("Download course JSON", json.dumps(course, indent=2), "generated_course.json", "application/json")

    with tab2:
        question = st.text_input("Ask a question", placeholder="What happens if an employee violates this policy?")
        if st.button("Answer question") and question:
            try:
                result = rag.answer_question(question)
                st.write(result["answer"])
                with st.expander("Retrieved evidence"):
                    for hit in result["retrieved"]:
                        st.markdown(f"**Page {hit['page']} · similarity {hit['similarity']}**")
                        st.write(hit["text"])
            except Exception as exc:
                st.error(str(exc))

    with tab3:
        retrieval_query = st.text_input("Search the policy semantically", value="employee responsibilities and compliance")
        k = st.slider("Top-k chunks", 1, 10, 5)
        if st.button("Retrieve chunks"):
            for i, hit in enumerate(rag.retrieve(retrieval_query, k=k), 1):
                st.markdown(f"### {i}. Page {hit['page']} (similarity {hit['similarity']})")
                st.write(hit["text"])
else:
    st.info("Upload a text-based PDF to start. Scanned PDFs would need OCR as an additional preprocessing step.")
