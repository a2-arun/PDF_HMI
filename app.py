"""Streamlit HMI: the student-facing chat interface.

This is intentionally a thin presentation layer — all retrieval and
generation logic lives in query_engine.py so it can be tested or reused
(e.g. from a future voice/robot front end) without Streamlit.
"""
import streamlit as st

import query_engine
import vector_store
from config import GRADE_PDFS

st.set_page_config(page_title="Enlighten Bot", page_icon="🤖", layout="centered")
st.title("🤖 Enlighten Bot")
st.caption("Ask a question about your school textbook — answers are grounded in your grade's curriculum.")

grade_labels = {name: name.replace("_", " ") for name in GRADE_PDFS}
selected_grade = st.selectbox(
    "Select your grade",
    options=list(grade_labels.keys()),
    format_func=lambda name: grade_labels[name],
)

if not vector_store.has_documents(selected_grade):
    st.warning(
        f"The knowledge base for **{grade_labels[selected_grade]}** hasn't been built yet. "
        f"Run this once from a terminal:\n\n"
        f"```\npython ingest.py --grade {selected_grade}\n```"
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.form("ask_form", clear_on_submit=True):
    question = st.text_input("Ask your question", placeholder="e.g. What is friction?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    with st.spinner("Thinking..."):
        answer = query_engine.answer_question(question, selected_grade)
    st.session_state.chat_history.append(
        {
            "grade": grade_labels[selected_grade],
            "question": question.strip(),
            "answer": answer.text,
            "sources": answer.sources,
        }
    )

st.divider()
for turn in reversed(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(f"**({turn['grade']})** {turn['question']}")
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        if turn["sources"]:
            page_list = ", ".join(str(p) for p in turn["sources"])
            st.caption(f"Textbook pages: {page_list}")

if not st.session_state.chat_history:
    st.info("Ask your first question above to get started.")
